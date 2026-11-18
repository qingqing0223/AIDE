from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audio_id_for(path: str | Path) -> str:
    path = Path(path)
    return f"{path.stem}_{sha256_file(path)[:10]}"


def normalize_waveform(waveform: torch.Tensor, peak: float = 0.5) -> torch.Tensor:
    """Apply AudioLDM's mean removal and peak normalization once before PGD."""
    waveform = waveform.float()
    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    scale = waveform.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return (waveform / scale * peak).clamp(-peak, peak)


def fit_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[-1] > target_samples:
        return waveform[..., :target_samples]
    if waveform.shape[-1] < target_samples:
        return F.pad(waveform, (0, target_samples - waveform.shape[-1]))
    return waveform


def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    duration_seconds: float = 10.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Load mono audio as [1,T] and fit AudioLDM's 102.4-frame/s input window."""
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(audio.T.copy()).mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if normalize:
        waveform = normalize_waveform(waveform)
    target_frames = int(round(duration_seconds * 102.4))
    waveform = fit_length(waveform, target_frames * 160)
    if normalize:
        waveform = normalize_waveform(waveform)
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Non-finite audio values in {path}")
    return waveform


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = waveform.detach().float().cpu()
    if value.ndim == 2:
        value = value[0]
    sf.write(str(path), value.numpy(), sample_rate, subtype="PCM_16")


def stft_magnitude(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 1024,
) -> torch.Tensor:
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    window = torch.hann_window(win_length, periodic=True, device=waveform.device)
    return torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode="reflect", return_complex=True
    ).abs()


def snr_db(clean: torch.Tensor, protected: torch.Tensor) -> float:
    noise = protected - clean
    signal_power = clean.square().mean().clamp_min(1e-12)
    noise_power = noise.square().mean().clamp_min(1e-12)
    return float((10.0 * torch.log10(signal_power / noise_power)).detach().cpu())


def lsd_db(clean: torch.Tensor, protected: torch.Tensor) -> float:
    a = stft_magnitude(clean).square().clamp_min(1e-9)
    b = stft_magnitude(protected).square().clamp_min(1e-9)
    value = 10.0 * torch.sqrt((torch.log10(a) - torch.log10(b)).square().mean())
    return float(value.detach().cpu())


def list_audio_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"})
    if not files:
        raise RuntimeError(f"No WAV/FLAC files found under {root}")
    return files


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
