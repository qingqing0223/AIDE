import argparse
from pathlib import Path

import librosa
import librosa.display as lbd
import matplotlib.pyplot as plt
import numpy as np


# -------- 基础工具函数 --------

def _load_wav_mono(path: Path, target_sr: int = 16000):
    """读 wav，转单声道，并按需要重采样到 target_sr。"""
    y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    if target_sr is not None and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------- 各种视图绘制函数（7 张图） --------

def plot_waveform(y, sr, out_path: Path):
    t = np.arange(len(y)) / sr
    plt.figure(figsize=(6, 2))
    plt.plot(t, y, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_cqt(y, sr, out_path: Path):
    C = np.abs(librosa.cqt(y, sr=sr))
    C_db = librosa.amplitude_to_db(C, ref=np.max)

    plt.figure(figsize=(4, 3))
    lbd.specshow(C_db, sr=sr, x_axis="time", y_axis="cqt_note")
    plt.colorbar(format="%+2.0f dB")
    plt.title("CQT")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_chroma(y, sr, out_path: Path):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(4, 3))
    lbd.specshow(chroma, y_axis="chroma", x_axis="time")
    plt.colorbar()
    plt.title("Chromagram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mel(y, sr, out_path: Path):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 3))
    lbd.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_f0(y, sr, out_path: Path):
    """用 librosa.yin 估计 f0（不再依赖 Praat，避免之前那些安装问题）。"""
    f0 = librosa.yin(y, fmin=50, fmax=sr / 2, sr=sr)
    times = librosa.times_like(f0, sr=sr)

    plt.figure(figsize=(6, 2))
    plt.plot(times, f0, linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.title("Fundamental Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_stft(y, sr, out_path: Path):
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 3))
    lbd.specshow(S_db, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("STFT (Log-power)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mps(y, sr, out_path: Path):
    """
    MPS（Modulation Power Spectrum）简单实现：
    1) 做 STFT 得到功率谱
    2) 对 log-power 做 2D FFT
    3) 取幅度并 fftshift，画成 2D 图
    """
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    M = np.fft.fftshift(np.abs(np.fft.fft2(S_db)))

    plt.figure(figsize=(4, 3))
    plt.imshow(M, origin="lower", aspect="auto")
    plt.xlabel("Modulation (time)")
    plt.ylabel("Modulation (freq)")
    plt.title("Modulation Power Spectrum")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_7_views(wav_path: Path, out_dir: Path, prefix: str):
    y, sr = _load_wav_mono(wav_path, target_sr=16000)
    out_dir = _ensure_dir(out_dir)

    plot_waveform(y, sr, out_dir / f"{prefix}_wave.png")
    plot_cqt(y, sr, out_dir / f"{prefix}_cqt.png")
    plot_chroma(y, sr, out_dir / f"{prefix}_chroma.png")
    plot_mel(y, sr, out_dir / f"{prefix}_mel.png")
    plot_f0(y, sr, out_dir / f"{prefix}_f0.png")
    plot_stft(y, sr, out_dir / f"{prefix}_stft.png")
    plot_mps(y, sr, out_dir / f"{prefix}_mps.png")


# -------- 针对 gen/ungen case 的封装 --------

AUDIO_ITEMS = [
    ("raw.wav", "fig_raw", "raw"),
    ("adv.wav", "fig_adv", "adv"),
    ("gen.wav", "fig_gen", "gen"),
    ("ungen.wav", "fig_ungen", "ungen"),
]


def process_case(case_dir: Path):
    print(f"[INFO] Processing case dir: {case_dir}")
    for wav_name, fig_subdir, prefix in AUDIO_ITEMS:
        wav_path = case_dir / wav_name
        if not wav_path.exists():
            print(f"[WARN] {wav_path} not found, skip {prefix}")
            continue
        fig_dir = case_dir / fig_subdir
        save_7_views(wav_path, fig_dir, prefix)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate 7-view figures (with MPS) for gen/ungen cases."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help=(
            "根目录："
            "1) 如果下面直接有 raw.wav / adv.wav / gen.wav / ungen.wav，则认为这是单个 case；"
            "2) 如果下面是 case_*/ 子目录，则对每个 case_* 目录处理。"
        ),
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="最多处理多少个 case（调试用，可不填）。",
    )
    args = parser.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root dir not found: {root}"

    case_dirs = []

    # 情况 1：root 本身就是一个 case 目录（里面直接有 raw.wav 等）
    if (root / "raw.wav").exists():
        case_dirs = [root]
    else:
        # 情况 2：root 下面有若干个 case_*/ 子目录
        for p in sorted(root.iterdir()):
            if p.is_dir() and p.name.startswith("case_"):
                case_dirs.append(p)

    if not case_dirs:
        raise RuntimeError(
            f"No case dirs found under {root}. "
            f"要么直接把 --root 指到某个 gen_ungen_caseX，"
            f"要么指到包含 case_* 子目录的 gen_ungen_all。"
        )

    if args.max_cases is not None:
        case_dirs = case_dirs[: args.max_cases]

    print(f"[INFO] Found {len(case_dirs)} case(s) to process.")

    for i, case_dir in enumerate(case_dirs, 1):
        print(f"[INFO] === [{i}/{len(case_dirs)}] {case_dir.name} ===")
        process_case(case_dir)


if __name__ == "__main__":
    main()
