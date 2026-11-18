from __future__ import annotations
from pathlib import Path
import torch
import torchaudio
import torch.nn.functional as F

def load_audio(path: str | Path, sample_rate: int = 16000, mono: bool = True) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if mono and wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav

def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    torchaudio.save(str(path), wav.detach().cpu().float().clamp(-1, 1), sample_rate)

def pad_or_trim(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] > length: return wav[..., :length]
    if wav.shape[-1] < length: return F.pad(wav, (0, length - wav.shape[-1]))
    return wav

def stft_magnitude(wav: torch.Tensor, n_fft=1024, hop=256, win=1024) -> torch.Tensor:
    x = wav.squeeze(1) if wav.dim() == 3 else wav
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    return torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, center=True, return_complex=True).abs()
