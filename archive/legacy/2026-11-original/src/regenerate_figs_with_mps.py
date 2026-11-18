import argparse
import shutil
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt


TARGET_SR = 16000


def load_wav(path: Path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return y, sr


def plot_wave(y, sr, out_path):
    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(6, 2))
    plt.plot(t, y, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mel(y, sr, out_path, n_fft=1024, hop_length=256, n_mels=128):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel"
    )
    plt.colorbar(format="%.1f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cqt(y, sr, out_path):
    C = librosa.cqt(y=y, sr=sr)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        C_db, sr=sr, x_axis="time", y_axis="cqt_note"
    )
    plt.colorbar(format="%.1f dB")
    plt.title("CQT")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_chroma(y, sr, out_path):
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(C, y_axis="chroma", x_axis="time")
    plt.colorbar()
    plt.title("Chromagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stft(y, sr, out_path, n_fft=1024, hop_length=256):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(4, 3))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear"
    )
    plt.colorbar(format="%.1f dB")
    plt.title("STFT Spectrogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mps(y, sr, out_path, n_fft=1024, hop_length=256):
    """
    简易版调制谱 (Modulation Power Spectrum):
    1) 先做 STFT 得到 (freq x time) 的幅度谱
    2) 对 log 幅度做 2D FFT
    3) 画功率谱 (freq_mod x freq_spec)
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    log_mag = np.log1p(mag)

    M = np.fft.fft2(log_mag)
    M = np.fft.fftshift(np.abs(M))

    M_db = 20 * np.log10(M + 1e-6)

    plt.figure(figsize=(4, 3))
    plt.imshow(
        M_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(format="%.1f dB")
    plt.title("Modulation Power Spectrum (MPS)")
    plt.xlabel("Modulation freq (time)")
    plt.ylabel("Spectral freq (freq)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_f0(y, sr, out_path):
    # 简单根频率估计：librosa.yin
    f0 = librosa.yin(
        y,
        fmin=50,
        fmax=800,
        sr=sr,
    )
    t = np.linspace(0, len(y) / sr, num=len(f0))
    plt.figure(figsize=(6, 2))
    plt.plot(t, f0, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title("Fundamental Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_all_views_for_wav(wav_path: Path, fig_dir: Path, prefix: str):
    y, sr = load_wav(wav_path)
    base = fig_dir / prefix

    plot_wave(y, sr, f"{base}_wave.png")
    plot_mel(y, sr, f"{base}_mel.png")
    plot_cqt(y, sr, f"{base}_cqt.png")
    plot_chroma(y, sr, f"{base}_chroma.png")
    plot_stft(y, sr, f"{base}_stft.png")
    plot_mps(y, sr, f"{base}_mps.png")
    plot_f0(y, sr, f"{base}_f0.png")


def process_case(case_dir: Path):
    """
    对一个 case_XXX 目录下的 5 个 wav：
    raw.wav / noise.wav / adv.wav / gen.wav / ungen.wav
    生成 7 张视图，并且在生成前删除旧的 fig_* 目录。
    """
    print(f"[INFO] Processing {case_dir}")
    mapping = [
        ("raw.wav", "fig_raw", "raw"),
        ("noise.wav", "fig_noise", "noise"),
        ("adv.wav", "fig_adv", "adv"),
        ("gen.wav", "fig_gen", "gen"),
        ("ungen.wav", "fig_ungen", "ungen"),
    ]

    for wav_name, fig_name, prefix in mapping:
        wav_path = case_dir / wav_name
        if not wav_path.exists():
            print(f"[WARN] {wav_path} not found, skip.")
            continue

        fig_dir = case_dir / fig_name
        if fig_dir.exists():
            shutil.rmtree(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        save_all_views_for_wav(wav_path, fig_dir, prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="out/adv_audiocaps_eps001_full/test/gen_ungen_all",
        help="gen_ungen_all 根目录",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=-1,
        help="调试用，>0 时只处理前 N 个 case",
    )
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root dir not found: {root}"

    case_dirs = sorted([p for p in root.glob("case_*") if p.is_dir()])
    if args.max_cases > 0:
        case_dirs = case_dirs[: args.max_cases]

    print(f"[INFO] Found {len(case_dirs)} cases under {root}")
    for i, case_dir in enumerate(case_dirs):
        print(f"[INFO] {i+1}/{len(case_dirs)} -> {case_dir.name}")
        process_case(case_dir)


if __name__ == "__main__":
    main()
