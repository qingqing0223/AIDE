import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

def load_wav(path):
    """用 soundfile 读取 wav，返回一维 numpy，取第一声道。"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        return data, sr
    else:
        # (T, C) -> 取第一声道
        return data[:, 0], sr

def resample_if_needed(x, sr, target_sr=16000):
    """如果采样率不是 target_sr，可以在这里加重采样；目前假设都是 16k，直接返回。"""
    # 如果你的数据不是 16k，可以在这里加 torchaudio 的重采样
    return x, sr

def compute_snr_db(clean, adv):
    """信噪比：10*log10( P_signal / P_noise )"""
    clean = clean.astype(np.float64)
    adv = adv.astype(np.float64)
    noise = adv - clean
    ps = np.mean(clean ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    return 10.0 * np.log10(ps / pn)

def compute_lsd(clean, adv, n_fft=512, hop=256):
    """
    Log Spectral Distortion:
    对每一帧的幅度谱，计算 20*log10 差的均方根，再对所有帧取平均。
    """
    clean_t = torch.from_numpy(clean)[None, :]
    adv_t = torch.from_numpy(adv)[None, :]

    spec_clean = torch.stft(
        clean_t,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs().numpy()[0]  # (freq, time)

    spec_adv = torch.stft(
        adv_t,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs().numpy()[0]

    eps = 1e-7
    log_clean = 20.0 * np.log10(spec_clean + eps)
    log_adv = 20.0 * np.log10(spec_adv + eps)

    lsd_per_frame = np.sqrt(np.mean((log_clean - log_adv) ** 2, axis=0))
    return float(np.mean(lsd_per_frame))

def process_split(split, clean_root, delta_root, adv_root, out_root):
    clean_dir = Path(clean_root) / split
    delta_dir = Path(delta_root) / split
    adv_dir = Path(adv_root) / split

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"metrics_{split}.jsonl"

    wav_files = sorted(clean_dir.glob("*.wav"))
    if not wav_files:
        print(f"[WARN] split={split}: 在 {clean_dir} 没有找到 wav，跳过")
        return

    print(f"[INFO] split={split}: 共 {len(wav_files)} 条样本，开始计算指标...")
    num_ok = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for wav_path in wav_files:
            uid = wav_path.stem

            clean, sr = load_wav(str(wav_path))
            clean, sr = resample_if_needed(clean, sr)

            # adv 优先用已经生成的 adv.wav
            adv_path = adv_dir / f"{uid}.wav"
            if adv_path.exists():
                adv, sr2 = load_wav(str(adv_path))
                adv, sr2 = resample_if_needed(adv, sr2)
            else:
                # 如果没有 adv.wav，就用 clean + δ 重构
                delta_path = delta_dir / f"{uid}.pt"
                if not delta_path.exists():
                    print(f"[WARN] split={split} id={uid}: 找不到 adv.wav 和 delta.pt，跳过")
                    continue
                delta = torch.load(str(delta_path), map_location="cpu").numpy().astype("float32")
                # 对齐长度
                min_len = min(len(clean), len(delta))
                clean = clean[:min_len]
                delta = delta[:min_len]
                adv = clean + delta

            min_len = min(len(clean), len(adv))
            clean = clean[:min_len]
            adv = adv[:min_len]

            snr_db = compute_snr_db(clean, adv)
            lsd_val = compute_lsd(clean, adv)

            rec = {
                "id": uid,
                "snr_db": float(snr_db),
                "lsd": float(lsd_val),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num_ok += 1

    print(f"[DONE] split={split}: 共写入 {num_ok} 条指标到 {out_file}")

def main():
    CLEAN_ROOT = "data/audiocaps_wav"
    DELTA_ROOT = "out/audiocaps_eps001"
    ADV_ROOT = "out/audiocaps_eps001_wav"
    OUT_ROOT = "out/audiocaps_eps001"

    for split in ["validation", "test"]:
        process_split(split, CLEAN_ROOT, DELTA_ROOT, ADV_ROOT, OUT_ROOT)

if __name__ == "__main__":
    main()
