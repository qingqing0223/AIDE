# -*- coding: utf-8 -*-
# eval_fd_fad_offline.py
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    import torch
    import torchaudio
except Exception:
    torch = None
    torchaudio = None


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
        wav = wav.T
    return wav.mean(axis=-1)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return wav
    if torchaudio is None:
        x = np.arange(len(wav))
        xp = np.linspace(0, len(wav) - 1, int(len(wav) * target_sr / sr))
        return np.interp(xp, x, wav).astype(np.float32)
    w = torch.from_numpy(wav).float().unsqueeze(0)
    w2 = torchaudio.functional.resample(w, orig_freq=sr, new_freq=target_sr)
    return w2.squeeze(0).cpu().numpy().astype(np.float32)


def logmel_embed(wav: np.ndarray, sr: int, target_sr: int, n_mels: int = 128) -> np.ndarray:
    """
    纯 log-mel embedding：绝对离线、不会拉任何权重。
    """
    wav = wav.astype(np.float32)
    wav = _to_mono(wav)
    wav = _resample(wav, sr, target_sr)
    if torchaudio is None:
        # 没有 torchaudio：退化为简单分帧能量（不推荐，但能跑）
        frame = 1024
        hop = 256
        xs = []
        for i in range(0, len(wav) - frame + 1, hop):
            xs.append(np.log1p(np.mean(wav[i:i+frame] ** 2) + 1e-8))
        x = np.array(xs, dtype=np.float32)
        return np.array([x.mean()], dtype=np.float32)

    w = torch.from_numpy(wav).float()
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels,
        power=2.0,
    )(w)  # (n_mels, T)
    mel = torch.log1p(mel)
    emb = mel.mean(dim=1)  # (n_mels,)
    return emb.detach().cpu().numpy().astype(np.float32)


def _cov(x: np.ndarray) -> np.ndarray:
    # x: (N, D)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    return (xc.T @ xc) / max(x.shape[0] - 1, 1)


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    # 近似 sqrtm，用于 FD；要求 A 接近 PSD
    # 对称化
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w + 1e-12)) @ V.T


def frechet_distance(mu1, cov1, mu2, cov2) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm_psd(cov1 @ cov2)
    tr = np.trace(cov1 + cov2 - 2.0 * covmean)
    return float(diff @ diff + tr)


def collect_embs(wav_paths: List[Path], target_sr: int, n_mels: int) -> np.ndarray:
    embs = []
    for p in tqdm(wav_paths, desc="logmel"):
        wav, sr = sf.read(str(p))
        embs.append(logmel_embed(wav, sr, target_sr=target_sr, n_mels=n_mels))
    return np.stack(embs, axis=0).astype(np.float32)


def try_fad(mode_name: str, real_dir: Path, fake_dir: Path) -> Dict[str, float]:
    """
    可选：如果你本地已经把 frechet_audio_distance 相关依赖和权重弄齐了，就会算；
    否则自动给 NaN，不影响主流程（FD_logmel 仍然有）。
    """
    out = {"FAD_vggish": float("nan"), "FAD_pann": float("nan")}
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except Exception:
        return out

    # 关键：彻底禁止联网（否则会像你截图那样去 huggingface 拉 bert）
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        fad_vggish = FrechetAudioDistance(
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        out["FAD_vggish"] = float(fad_vggish.score(str(real_dir), str(fake_dir)))
    except Exception:
        pass

    try:
        fad_pann = FrechetAudioDistance(
            model_name="pann",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        out["FAD_pann"] = float(fad_pann.score(str(real_dir), str(fake_dir)))
    except Exception:
        pass

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--real_subdir", required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--max_audio_per_mode", type=int, default=-1)
    ap.add_argument("--enable_fad", action="store_true", help="本地权重齐全时再开")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    real_dir = results_root / args.real_subdir
    real_wavs = sorted(real_dir.glob("*.wav"))
    if args.max_audio_per_mode > 0:
        real_wavs = real_wavs[: args.max_audio_per_mode]
    if len(real_wavs) == 0:
        raise RuntimeError(f"real dir empty: {real_dir}")

    print(f"[INFO] real wavs: {len(real_wavs)} @ {real_dir}")
    real_embs = collect_embs(real_wavs, target_sr=args.target_sr, n_mels=args.n_mels)
    mu_r = real_embs.mean(axis=0)
    cov_r = _cov(real_embs)

    rows = []
    for mode in args.modes:
        fake_dir = results_root / mode
        fake_wavs = sorted(fake_dir.glob("*.wav"))
        if args.max_audio_per_mode > 0:
            fake_wavs = fake_wavs[: args.max_audio_per_mode]
        if len(fake_wavs) == 0:
            print(f"[WARN] {mode} empty, skip")
            continue

        print(f"[INFO] mode={mode}, wavs={len(fake_wavs)}")
        fake_embs = collect_embs(fake_wavs, target_sr=args.target_sr, n_mels=args.n_mels)
        mu_f = fake_embs.mean(axis=0)
        cov_f = _cov(fake_embs)

        fd = frechet_distance(mu_r, cov_r, mu_f, cov_f)

        extra = {"FAD_vggish": float("nan"), "FAD_pann": float("nan")}
        if args.enable_fad:
            extra = try_fad(mode, real_dir, fake_dir)

        rows.append({
            "mode": mode,
            "N_audio": len(fake_wavs),
            "FD_logmel": fd,
            "FAD_vggish": extra["FAD_vggish"],
            "FAD_pann": extra["FAD_pann"],
        })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "N_audio", "FD_logmel", "FAD_vggish", "FAD_pann"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
