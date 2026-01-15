#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全离线版 AudioCaps 评测脚本（不依赖 audioldm_eval / frechet_audio_distance / HuggingFace）

目前计算的指标：
  - FD_logmel: 基于 log-mel 特征的 Frechet Distance
    （你可以在论文中把它记作 FD* 或 FAD*，在附录里说明：使用 log-mel + 简单统计）

输入目录结构（已经由 collect_results_from_out_masked.py / run_guard_full_pipeline.py 生成）：
  results/
    real_for_eval/          # 真实语音
    B0_AudioLDM/            # 基线
    B1_delta_nogate/        # +δ，无门控
    Ours_express_safe/      # 我们方法（safe prompt 下最终输出）
    Ours_express_harm/      # 我们方法（harm prompt 下最终输出）
    Ours_hidden/            # 如果之后有 latent 攻击版本，也可以一起评测

运行示例：
  cd /autodl-tmp/speech_guard
  python eval_metrics_offline.py \
    --results_root results \
    --real_subdir real_for_eval \
    --modes B0_AudioLDM B1_delta_nogate Ours_express_safe Ours_express_harm

输出：
  - results/metrics_test_baseline_offline.jsonl
  - results/metrics_test_baseline_overview_offline.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import linalg
from tqdm import tqdm

# ===== 配置 =====
SAMPLE_RATE = 16000
N_MELS = 64
DURATION = 5.0  # 固定为 5 秒和之前实验保持一致

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=320,
    n_mels=N_MELS,
    power=2.0,
).to(device)

amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)


def load_wav_mono(path: Path) -> torch.Tensor:
    """读 wav，转单声道、重采样到 16k，并裁剪/补零到固定长度"""
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = torch.from_numpy(y).float()

    if sr != SAMPLE_RATE:
        y = torchaudio.functional.resample(y.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

    target_len = int(SAMPLE_RATE * DURATION)
    if y.numel() < target_len:
        pad = target_len - y.numel()
        y = torch.nn.functional.pad(y, (0, pad))
    y = y[:target_len]
    return y.to(device)


def extract_logmel_feat(path: Path) -> np.ndarray:
    """从 wav 提取 log-mel 特征，并对时间维做平均，得到一个定长向量"""
    wav = load_wav_mono(path).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        spec = mel_spec(wav)               # [1, n_mels, T']
        logmel = amp_to_db(spec)           # [1, n_mels, T']
        feat = logmel.mean(dim=-1).squeeze(0)  # [n_mels]
    return feat.cpu().numpy().astype(np.float64)


def compute_stats(feats: np.ndarray):
    """给定 [N, D] 特征，计算均值和协方差"""
    feats = np.asarray(feats, dtype=np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """经典 FID / FD 公式"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # 数值不稳定时加一点对角噪声
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fd)


def collect_feats(wav_dir: Path):
    wav_paths = sorted(wav_dir.glob("*.wav"))
    if not wav_paths:
        raise RuntimeError(f"目录 {wav_dir} 下面没有 wav 文件！")
    feats = []
    for p in tqdm(wav_paths, desc=str(wav_dir), ncols=80):
        feats.append(extract_logmel_feat(p))
    feats = np.stack(feats, axis=0)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--real_subdir", type=str, default="real_for_eval")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=[
            "B0_AudioLDM",
            "B1_delta_nogate",
            "Ours_express_safe",
            "Ours_express_harm",
        ],
        help="要评测的子目录名，可在命令行里自行增减",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="metrics_test_baseline_offline.jsonl",
    )
    parser.add_argument(
        "--overview_csv",
        type=str,
        default="metrics_test_baseline_overview_offline.csv",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    real_dir = root / args.real_subdir
    if not real_dir.exists():
        raise SystemExit(f"[ERROR] 找不到真实语音目录: {real_dir}")

    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 真实语音目录: {real_dir}")

    # 1) 真实语音统计
    real_feats = collect_feats(real_dir)
    real_mu, real_sigma = compute_stats(real_feats)
    print(
        f"[INFO] real_for_eval: N = {real_feats.shape[0]}, "
        f"feat_dim = {real_feats.shape[1]}"
    )

    # 2) 各个模式的 FD
    rows = []
    jsonl_path = root / args.output_jsonl
    with jsonl_path.open("w", encoding="utf-8") as f_jsonl:
        for mode in args.modes:
            mode_dir = root / mode
            if not mode_dir.exists():
                print(f"[WARN] 跳过 {mode}：目录不存在 {mode_dir}")
                continue

            print("\n" + "=" * 40)
            print(f"[MODE] {mode} @ {mode_dir}")

            feats = collect_feats(mode_dir)
            mu, sigma = compute_stats(feats)
            fd_logmel = frechet_distance(real_mu, real_sigma, mu, sigma)

            row = {
                "mode": mode,
                "N_audio": int(feats.shape[0]),
                "FD_logmel": float(fd_logmel),
            }
            rows.append(row)
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(
                f"[RESULT] FD_logmel {mode}: {fd_logmel:.4f} "
                f"(N_audio={feats.shape[0]})"
            )

    # 3) 写一个 csv 方便你用 Excel 打开
    csv_path = root / args.overview_csv
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["mode", "N_audio", "FD_logmel"])
        for r in rows:
            writer.writerow([r["mode"], r["N_audio"], r["FD_logmel"]])

    print("\n[FINISH] 离线 FD 计算完成。")
    print(f"[INFO] 结果 jsonl: {jsonl_path}")
    print(f"[INFO] 结果 CSV:   {csv_path}")


if __name__ == "__main__":
    main()
