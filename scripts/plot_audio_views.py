import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ===================== 基本参数 =====================

HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 80


# ===================== 音频加载 =====================

def load_wav_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """用 torchaudio 加载 wav，转单声道并重采样到 target_sr。"""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    y = wav.squeeze(0).cpu().numpy()
    return y, sr


# ===================== 各种时频表示 =====================

def make_mel(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def make_cqt(y: np.ndarray, sr: int) -> np.ndarray:
    C = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_bins=84,
        bins_per_octave=12,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_db


def make_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    return chroma


def make_mps(mel_db: np.ndarray) -> np.ndarray:
    X = mel_db - mel_db.mean()
    F = np.fft.fft2(X)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)
    P_db = 10 * np.log10(P + 1e-8)
    return P_db


def make_cochleagram_like(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=64,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def estimate_formants_lpc(y: np.ndarray, sr: int, n_formants: int = 3) -> list[float]:
    """极简版 LPC formant 估计，用来画 F1/F2/F3 水平线。"""
    N = int(0.03 * sr)
    if len(y) < N:
        y_seg = np.pad(y, (0, N - len(y)))
    else:
        mid = len(y) // 2
        start = max(0, mid - N // 2)
        end = start + N
        y_seg = y[start:end]

    window = np.hamming(len(y_seg))
    y_win = y_seg * window

    order = int(sr / 1000) + 2
    try:
        A = librosa.lpc(y_win, order=order)
        roots = np.roots(A)
        roots = roots[np.imag(roots) >= 0.01]
        angs = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angs * (sr / (2 * np.pi))
        bws = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))

        formants = []
        for f, bw in sorted(zip(freqs, bws), key=lambda x: x[0]):
            if 90 < f < 5000 and bw < 400:
                formants.append(f)
        return formants[:n_formants]
    except Exception:
        return []


def make_f0_and_formants(y: np.ndarray, sr: int):
    """返回 F0 时间轴 / F0 曲线 / 全局 formant 频率。"""
    f0, vf, vp = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=HOP_LENGTH,
    )
    t_f0 = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    formants = estimate_formants_lpc(y, sr, n_formants=3)
    return t_f0, f0, formants


# ===================== 画图辅助 =====================

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_spec_and_save(
    data: np.ndarray,
    sr: int,
    kind: str,
    out_png: Path,
):
    _ensure_dir(out_png.parent)
    plt.figure(figsize=(6, 4))

    if kind == "mel":
        librosa.display.specshow(
            data,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
        )
        plt.title("Mel-spectrogram")
    elif kind == "cqt":
        librosa.display.specshow(
            data,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="cqt_hz",
        )
        plt.title("CQT")
    elif kind == "chroma":
        librosa.display.specshow(
            data,
            x_axis="time",
            y_axis="chroma",
        )
        plt.title("Chroma")
    elif kind == "mps":
        librosa.display.specshow(
            data,
            x_axis="time",
            y_axis="linear",
        )
        plt.title("Modulation Power Spectrum")
    elif kind == "coch":
        librosa.display.specshow(
            data,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
        )
        plt.title("Cochleagram-like")
    else:
        plt.imshow(data, aspect="auto", origin="lower")
        plt.title(kind)

    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()


def plot_f0_formant_and_save(y: np.ndarray, sr: int, out_png: Path):
    """
    底图：线性频率 spectrogram；
    叠加：F0 曲线 + F1/F2/F3 水平线。
    重点：x/y 维度完全对齐，避免 ValueError。
    """
    _ensure_dir(out_png.parent)

    # 频谱
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    times = librosa.times_like(S_db[0], sr=sr, hop_length=HOP_LENGTH)

    plt.figure(figsize=(6, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
    )
    plt.colorbar(format="%+2.0f dB")

    # F0
    t_f0, f0, formants = make_f0_and_formants(y, sr)
    if f0 is not None:
        f0 = np.array(f0)
        mask = ~np.isnan(f0)
        if mask.sum() > 0:
            # 保证 x / y 长度相同，再 plot
            t_plot = t_f0[mask]
            f_plot = f0[mask]
            n = min(len(t_plot), len(f_plot))
            plt.plot(
                t_plot[:n],
                f_plot[:n],
                color="w",
                linewidth=1.0,
                label="F0",
            )

    # Formant 水平线
    for idx, f in enumerate(formants):
        plt.hlines(
            f,
            xmin=times[0],
            xmax=times[-1],
            colors=["c", "m", "y"][idx % 3],
            linestyles="--",
            linewidth=1.0,
            label=f"F{idx+1}",
        )

    plt.ylim(0, 5000)
    plt.title("F0 & Formants (Praat-like)")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()


# ===================== 对外主函数 =====================

def save_all_views(out_dir: Path, uid: str, sr: int, wav_paths):
    """
    对一条语音的 raw / noise / adv 生成 6 张图：
    mel / CQT / MPS / Chroma / Cochleagram-like / F0+Formant
    """

    if isinstance(wav_paths, dict):
        raw_path = Path(wav_paths["raw"])
        noise_path = Path(wav_paths["noise"])
        adv_path = Path(wav_paths["adv"])
    else:
        raw_path, noise_path, adv_path = map(Path, wav_paths)

    tag2path = {
        "raw": raw_path,
        "noise": noise_path,
        "adv": adv_path,
    }

    for tag, path in tag2path.items():
        if not path.is_file():
            print(f"[WARN] {uid} 的 {tag} 文件不存在: {path}")
            continue

        y, sr_now = load_wav_mono(path, sr)

        mel_db = make_mel(y, sr_now)
        cqt_db = make_cqt(y, sr_now)
        chroma = make_chroma(y, sr_now)
        mps_db = make_mps(mel_db)
        coch_db = make_cochleagram_like(y, sr_now)

        base = out_dir / uid / tag
        _ensure_dir(base)

        plot_spec_and_save(mel_db, sr_now, "mel", base / "mel.png")
        plot_spec_and_save(cqt_db, sr_now, "cqt", base / "cqt.png")
        plot_spec_and_save(chroma, sr_now, "chroma", base / "chroma.png")
        plot_spec_and_save(mps_db, sr_now, "mps", base / "mps.png")
        plot_spec_and_save(coch_db, sr_now, "coch", base / "coch.png")
        plot_f0_formant_and_save(y, sr_now, base / "f0_formant.png")

        print(f"[OK] {uid} / {tag} 6 张图已保存到 {base}")
