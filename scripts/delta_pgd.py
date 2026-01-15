import os
import math
import argparse
from pathlib import Path
import random
import json

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


# ========= 工程根目录 & 数据路径 =========
PROJECT_ROOT = Path("/root/autodl-tmp/speech_guard")
DATA_ROOT = PROJECT_ROOT / "data" / "audiocaps_wav"
ZTARG_ROOT = PROJECT_ROOT / "ztarg" / "musan" / "noise" / "free-sound"

SAMPLE_RATE = 16000


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav  # [1, T]


def pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """把目标噪声 z_targ 调整到和 x 一样长"""
    c, t = wav.shape
    if t == target_len:
        return wav
    if t > target_len:
        start = random.randint(0, t - target_len)
        return wav[:, start:start + target_len]
    # t < target_len, 右侧补 0
    pad = target_len - t
    return torch.nn.functional.pad(wav, (0, pad))


# ======= Mel 特征（近似“潜空间”） =======
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    f_min=0,
    f_max=SAMPLE_RATE // 2,
)


def logmel(wav: torch.Tensor) -> torch.Tensor:
    spec = _mel(wav)
    return torch.log(spec + 1e-6)


# ======= δ 质量指标 =======
def snr_db(x: torch.Tensor, adv: torch.Tensor) -> float:
    noise = adv - x
    p_sig = (x ** 2).mean()
    p_noise = (noise ** 2).mean() + 1e-9
    return 10.0 * torch.log10(p_sig / p_noise).item()


def lsd_db(x: torch.Tensor, adv: torch.Tensor) -> float:
    # 对数谱距离
    spec_x = torch.stft(x, n_fft=512, hop_length=256, win_length=512, return_complex=True)
    spec_y = torch.stft(adv, n_fft=512, hop_length=256, win_length=512, return_complex=True)
    px = spec_x.abs() ** 2 + 1e-9
    py = spec_y.abs() ** 2 + 1e-9
    lsd = (torch.log10(px) - torch.log10(py)) ** 2
    return torch.sqrt(lsd.mean()).item() * 10.0


# ======= PGD 主过程（在 log-mel 空间近似 Encoder Attack） =======
def pgd_attack(
    x: torch.Tensor,
    z_targ: torch.Tensor,
    eps: float = 0.01,
    alpha: float = 0.001,
    steps: int = 40,
    lam_targ: float = 1.0,
    lam_far: float = 0.2,
) -> (torch.Tensor, torch.Tensor):
    """
    x: [1, T], z_targ: [1, T]
    返回 δ 和 x_adv = x + δ
    """
    device = x.device
    x = x.to(device)
    z_targ = z_targ.to(device)

    delta = torch.zeros_like(x, device=device)

    for _ in range(steps):
        delta.requires_grad_(True)
        x_adv = torch.clamp(x + delta, -1.0, 1.0)

        mel_adv = logmel(x_adv)
        mel_x = logmel(x).detach()
        mel_t = logmel(z_targ).detach()

        # 让 mel_adv 接近目标噪声，同时远离原始语音
        loss_targ = torch.nn.functional.mse_loss(mel_adv, mel_t)
        loss_far = -torch.nn.functional.mse_loss(mel_adv, mel_x)
        loss = lam_targ * loss_targ + lam_far * loss_far

        loss.backward()
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta = delta + alpha * grad_sign
            delta = torch.clamp(delta, -eps, eps)
        delta = delta.detach()

    x_adv = torch.clamp(x + delta, -1.0, 1.0)
    return delta.detach(), x_adv.detach()


def build_file_list(split: str, max_samples: int | None) -> list[Path]:
    wav_dir = DATA_ROOT / split
    assert wav_dir.is_dir(), f"数据目录不存在: {wav_dir}"
    all_wavs = sorted(wav_dir.glob("*.wav"))
    if max_samples is not None and len(all_wavs) > max_samples:
        all_wavs = random.sample(all_wavs, max_samples)
    return all_wavs


def choose_target_noise(target_len: int) -> torch.Tensor:
    """
    在 ZTARG_ROOT 下收集所有 .wav/.WAV/.flac 文件，随机选一个，
    并裁剪/填充到和 x 相同长度。
    """
    print(f"[DEBUG] ZTARG_ROOT = {ZTARG_ROOT}")
    print(f"[DEBUG] exists={ZTARG_ROOT.exists()}, is_dir={ZTARG_ROOT.is_dir()}")

    if not ZTARG_ROOT.exists():
        raise RuntimeError(f"噪声根目录不存在: {ZTARG_ROOT}")

    # 递归把所有子目录里的 wav / flac 都找出来
    noise_files = [
        p for p in ZTARG_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in (".wav", ".flac")
    ]

    print(f"[DEBUG] 在 ZTARG_ROOT 及子目录下共找到 {len(noise_files)} 个候选噪声文件")

    if len(noise_files) == 0:
        raise RuntimeError(
            f"在 {ZTARG_ROOT} 下没有找到任何 .wav/.WAV/.flac 噪声文件，"
            f"请检查扩展名和路径。"
        )

    path = random.choice(noise_files)
    print(f"[DEBUG] 本次随机选中的噪声: {path}")

    z, sr = sf.read(str(path), always_2d=True)
    z = torch.from_numpy(z.T).float()
    if sr != SAMPLE_RATE:
        z = torchaudio.functional.resample(z, sr, SAMPLE_RATE)
    if z.size(0) > 1:
        z = z.mean(dim=0, keepdim=True)
    z = pad_or_trim(z, target_len)
    return z
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--max_samples", type=int, default=None, help="低资源试验时限制样本数")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--save_root", type=str, default="out/adv_audiocaps_eps001")
    args = parser.parse_args()

    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATA_ROOT    = {DATA_ROOT}")
    print(f"[INFO] ZTARG_ROOT   = {ZTARG_ROOT}")

    random.seed(2025)
    torch.manual_seed(2025)

    save_root = PROJECT_ROOT / args.save_root / args.split
    (save_root / "wav").mkdir(parents=True, exist_ok=True)
    metrics_path = save_root / "metrics.jsonl"

    file_list = build_file_list(args.split, args.max_samples)
    print(f"[INFO] split={args.split}, 样本数={len(file_list)}, 保存到 {save_root}")

    all_metrics = []

    for wav_path in tqdm(file_list):
        x = load_audio(wav_path)  # [1, T]
        z_targ = choose_target_noise(x.shape[1])

        delta, x_adv = pgd_attack(
            x,
            z_targ,
            eps=args.eps,
            alpha=args.alpha,
            steps=args.steps,
        )

        # 计算指标
        cur_snr = snr_db(x, x_adv)
        cur_lsd = lsd_db(x, x_adv)

        name = wav_path.stem
        raw_out = save_root / "wav" / f"{name}_raw.wav"
        noise_out = save_root / "wav" / f"{name}_noise.wav"
        adv_out = save_root / "wav" / f"{name}_adv.wav"

        torchaudio.save(str(raw_out), x.cpu(), SAMPLE_RATE)
        torchaudio.save(str(noise_out), delta.cpu(), SAMPLE_RATE)
        torchaudio.save(str(adv_out), x_adv.cpu(), SAMPLE_RATE)

        all_metrics.append(
            {
                "id": name,
                "snr_db": cur_snr,
                "lsd_db": cur_lsd,
            }
        )

    # 把所有样本的指标写成一行一个 json，方便后面聚合画曲线
    with open(metrics_path, "w", encoding="utf-8") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")

    snrs = [m["snr_db"] for m in all_metrics]
    lsds = [m["lsd_db"] for m in all_metrics]
    print(f"[DONE] 共 {len(all_metrics)} 条样本, 平均 SNR={sum(snrs)/len(snrs):.2f} dB, 平均 LSD={sum(lsds)/len(lsds):.2f} dB")
    print(f"[DONE] 详细指标保存在 {metrics_path}")


if __name__ == "__main__":
    main()
