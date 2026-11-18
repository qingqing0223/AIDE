from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt


def _ensure_mono(y):
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y


def plot_waveform(y, sr, out_file, title):
    y = _ensure_mono(y)
    t = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_mel(y, sr, out_file, title):
    y = _ensure_mono(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256,
                                   x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_cqt(y, sr, out_file, title):
    y = _ensure_mono(y)
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=256))

    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                                   sr=sr, hop_length=256,
                                   x_axis="time", y_axis="cqt_note",
                                   ax=ax, cmap="magma")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_chroma(y, sr, out_file, title):
    y = _ensure_mono(y)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(chroma, x_axis="time",
                                   y_axis="chroma", ax=ax, cmap="magma")
    ax.set_title(title)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_f0(y, sr, out_file, title):
    y = _ensure_mono(y)
    # 用 librosa.yin 估计基频
    f0 = librosa.yin(y, fmin=50, fmax=sr // 2, frame_length=2048, hop_length=256)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=256)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, f0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_stft(y, sr, out_file, title):
    y = _ensure_mono(y)
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256,
                                   x_axis="time", y_axis="linear", ax=ax, cmap="magma")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_mps(y, sr, out_file, title):
    """
    简单实现 MPS（调制功率谱）:
      1) 做 STFT 得到 log-power 频谱
      2) 对 log-power 做 2D FFT，再取幅度
      3) 显示为 2D 图像
    """
    y = _ensure_mono(y)
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    P = np.abs(S) ** 2
    logP = np.log(P + 1e-10)
    logP = logP - logP.mean()

    M = np.fft.fftshift(np.abs(np.fft.fft2(logP)))

    fig, ax = plt.subplots(figsize=(6, 4))
    img = ax.imshow(M, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Temporal modulation")
    ax.set_ylabel("Spectral modulation")
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def save_all_views(wav_path: Path, out_dir: Path, tag: str):
    """
    对单个 wav 生成 7 张图:
       wave / mel / cqt / chroma / f0 / stft / mps
    文件名形如:  raw_wave.png, raw_mel.png, ..., raw_mps.png
    """
    wav_path = Path(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = sf.read(str(wav_path))
    y = _ensure_mono(y)

    print(f"[INFO] save_all_views: {wav_path} -> {out_dir} (tag={tag})")

    base = tag
    plot_waveform(y, sr, out_dir / f"{base}_wave.png", f"{tag} - Waveform")
    plot_mel(y, sr, out_dir / f"{base}_mel.png",   f"{tag} - Mel spectrogram")
    plot_cqt(y, sr, out_dir / f"{base}_cqt.png",   f"{tag} - CQT")
    plot_chroma(y, sr, out_dir / f"{base}_chroma.png", f"{tag} - Chroma")
    plot_f0(y, sr, out_dir / f"{base}_f0.png",     f"{tag} - F0 track")
    plot_stft(y, sr, out_dir / f"{base}_stft.png", f"{tag} - STFT")
    plot_mps(y, sr, out_dir / f"{base}_mps.png",   f"{tag} - MPS")
