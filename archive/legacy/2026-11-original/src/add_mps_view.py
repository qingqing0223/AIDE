import os
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 直接复用原来生成 6 张图的函数
from visualize_audio_views import save_all_views


def compute_mps(y, sr, n_fft=1024, hop_length=256):
    """
    计算调制功率谱 MPS：
    1) 对语音做 STFT 得到 time-frequency 能量
    2) 在时轴上再做一次 FFT，得到 modulation 频率
    这里给出一个“工程可用”的实现，不追求最严格的信号处理细节。
    """
    # STFT  -> 频谱图 (freq x time)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    mag = np.abs(S)
    log_mag = librosa.amplitude_to_db(mag + 1e-8, ref=np.max)

    # 去掉每条频带上的直流分量，避免 DC 太亮
    log_mag = log_mag - log_mag.mean(axis=1, keepdims=True)

    # 对 2D 频谱图做 2D FFT，得到 modulation 频谱
    mps = np.fft.fftshift(np.fft.fft2(log_mag))
    mps_power = np.abs(mps)
    mps_db = 20.0 * np.log10(mps_power + 1e-8)

    return mps_db


def plot_mps(y, sr, out_path, n_fft=1024, hop_length=256):
    """把 MPS 画成一张 png 图"""
    mps_db = compute_mps(y, sr, n_fft=n_fft, hop_length=hop_length)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(4, 3))
    # 这里只是可视化，不严格标单位，主要看 pattern
    librosa.display.specshow(mps_db, cmap="magma")
    plt.title("Modulation Power Spectrum (MPS)")
    plt.colorbar(format="%.1f dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_all_views_with_mps(wav_path: str, out_dir: str, prefix: str):
    """
    先调用原来的 save_all_views 画 6 张图，
    再额外画一张 prefix_mps.png，一共 7 张图。
    """
    wav_path = str(wav_path)
    out_dir = str(out_dir)

    # 原来的 6 张图
    save_all_views(wav_path, out_dir, prefix)

    # 重新读一遍音频，画第 7 张 MPS 图
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    out_path = os.path.join(out_dir, f"{prefix}_mps.png")
    plot_mps(y, sr, out_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Save 7-view audio visualizations (6 原图 + 1 MPS) for a single wav."
    )
    parser.add_argument("--wav", type=str, required=True, help="Path to wav file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save figures")
    parser.add_argument("--prefix", type=str, required=True, help="Filename prefix")

    args = parser.parse_args()

    save_all_views_with_mps(args.wav, args.out_dir, args.prefix)
    print(f"[INFO] Saved 7 views (with MPS) to {args.out_dir} with prefix {args.prefix}")


if __name__ == "__main__":
    main()
