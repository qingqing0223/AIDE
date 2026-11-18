import os
import math
import random
from typing import Tuple, List

import torch
import torchaudio
from torch import nn


# ================== 配置 ==================

DATA_ROOT = "data/audiocaps_wav"                 # 已经划分好的 audiocaps
ZTARG_ROOT = "ztarg/musan/noise/free-sound"      # 目标噪声语音根目录（相对 /autodl-tmp/speech_guard）
SPLITS = ["validation", "test"]                  # 只在 val/test 上生成 PGD

EPS = 0.01        # l_inf 范数约束
STEP_SIZE = 0.001 # PGD 步长
N_ITERS = 40      # PGD 迭代步数

# 为低资源实验预留：0=全量，>0 表示每个 split 只取前 N 条
MAX_PER_SPLIT = 0

# 保存 δ tensor 和对抗语音的目录
DELTA_ROOT = "out/audiocaps_eps001"       # out/audiocaps_eps001/{split}/*.pt
ADV_WAV_ROOT = "out/audiocaps_eps001_wav" # out/audiocaps_eps001_wav/{split}/*.wav

# 全局缓存噪声文件列表
NOISE_FILES: List[str] = []


# ================== 特征提取（近似 encoder） ==================


class MelFeature(nn.Module):
    """
    用 log-mel + 频带加权作为“特征空间”，
    后面如果要换成 AudioLDM encoder 只需要改这里的 forward。
    """
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=64,
            center=True,
            power=2.0,
        )
        self.eps = 1e-6

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: [1, T] 或 [B, 1, T]
        返回: [B, D]
        """
        if wav.dim() == 2:  # [C, T]
            wav = wav.unsqueeze(0)  # [1, C, T]
        if wav.size(1) > 1:
            wav = wav.mean(dim=1, keepdim=True)

        spec = self.melspec(wav)  # [B, n_mels, frames]
        log_spec = torch.log(spec + self.eps)

        # 简单的“频谱整形”——高频稍微放大一点
        freqs = torch.linspace(0, 1, log_spec.size(1), device=log_spec.device)
        weight = 0.5 + 0.5 * freqs  # 低频 0.5，高频 1.0
        log_spec = log_spec * weight.view(1, -1, 1)

        feat = log_spec.mean(dim=2)  # [B, n_mels]
        feat = nn.functional.normalize(feat, dim=-1)
        return feat


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    p_signal = clean.pow(2).mean().item() + 1e-8
    p_noise = noise.pow(2).mean().item() + 1e-8
    return 10.0 * math.log10(p_signal / p_noise)


# ================== 读取 ztarg 噪声 ==================


def init_noise_files():
    """
    只在第一次调用时扫描 ZTARG_ROOT，之后全程复用。
    支持 .wav/.WAV/.flac/.mp3/.ogg 等常见格式。
    """
    global NOISE_FILES
    if NOISE_FILES:
        return

    root = ZTARG_ROOT
    abs_root = os.path.abspath(root)

    if not os.path.isdir(root):
        raise RuntimeError(f"ZTARG_ROOT 目录不存在: {abs_root}")

    exts = (".wav", ".flac", ".mp3", ".ogg")
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(os.path.join(dirpath, name))

    if not files:
        raise RuntimeError(f"在 {abs_root} 下没有找到任何音频文件(*.wav/*.flac/*.mp3/*.ogg) 作为 ztarg。")

    NOISE_FILES = files
    print(f"[PGD] 在 {abs_root} 下找到 {len(NOISE_FILES)} 个噪声文件可作为 ztarg。")


def pick_random_ztarg(sr: int, length: int, device: torch.device) -> torch.Tensor:
    """
    从全局 NOISE_FILES 里随机取一条噪声，截断/循环到与 x 同长度。
    """
    init_noise_files()

    path = random.choice(NOISE_FILES)
    wav, sr0 = torchaudio.load(path)
    if sr0 != sr:
        wav = torchaudio.functional.resample(wav, sr0, sr)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    T = wav.size(-1)
    if T >= length:
        wav = wav[..., :length]
    else:
        # 不够长就循环填充
        n_repeat = (length + T - 1) // T
        wav = wav.repeat(1, n_repeat)[..., :length]

    return wav.to(device)


# ================== PGD 主体 ==================


def pgd_attack(
    x: torch.Tensor,
    encoder: nn.Module,
    z_targ: torch.Tensor,
    eps: float,
    step_size: float,
    n_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float]:
    """
    在 waveform 空间做 PGD：
    让 encoder(x+δ) 接近 z_targ，同时远离 encoder(x)。
    """
    device = x.device
    encoder.eval()

    with torch.no_grad():
        z_x = encoder(x.unsqueeze(0))  # [1, D]

    delta = torch.zeros_like(x, device=device)

    for _ in range(n_iters):
        adv = (x + delta).clamp(-1.0, 1.0)
        adv = adv.detach().requires_grad_(True)

        z_adv = encoder(adv.unsqueeze(0))

        loss_close = (z_adv - z_targ).pow(2).mean()
        loss_far = (z_adv - z_x).pow(2).mean()
        loss = loss_close - 0.5 * loss_far

        loss.backward()
        grad = adv.grad

        delta = delta + step_size * grad.sign()
        delta = torch.clamp(delta, -eps, eps)

    adv = (x + delta).clamp(-1.0, 1.0).detach()

    noise = adv - x
    with torch.no_grad():
        l2 = noise.view(-1).norm(p=2).item()
        linf = noise.view(-1).abs().max().item()
        s = snr_db(x, adv)

    return delta.detach(), adv, s, l2, linf


# ================== split 级处理 ==================


def process_split(split: str, device: torch.device):
    raw_dir = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(raw_dir):
        print(f"[{split}] 找不到原始语音目录 {raw_dir}，跳过。")
        return

    delta_dir = os.path.join(DELTA_ROOT, split)
    adv_dir = os.path.join(ADV_WAV_ROOT, split)
    os.makedirs(delta_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".wav")])
    if MAX_PER_SPLIT > 0:
        wav_files = wav_files[:MAX_PER_SPLIT]

    print(f"[{split}] 需要处理 {len(wav_files)} 条样本。")
    if not wav_files:
        return

    # 用第一条语音的采样率初始化特征提取器（之前的 bug 就出在这里）
    first_path = os.path.join(raw_dir, wav_files[0])
    wav0, sr0 = torchaudio.load(first_path)
    sample_rate = int(sr0)
    feat_encoder = MelFeature(sample_rate=sample_rate).to(device)

    total_snr = 0.0
    total_l2 = 0.0
    total_linf = 0.0
    n_used = 0

    for i, fname in enumerate(wav_files):
        utt_id = os.path.splitext(fname)[0]
        raw_path = os.path.join(raw_dir, fname)

        delta_path = os.path.join(delta_dir, f"{utt_id}.pt")
        adv_path = os.path.join(adv_dir, f"{utt_id}.wav")
        if os.path.exists(delta_path) and os.path.exists(adv_path):
            continue  # 方便断点续跑

        wav, sr = torchaudio.load(raw_path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.to(device)

        # 选一条目标噪声
        ztarg_wav = pick_random_ztarg(sample_rate, wav.size(-1), device)
        with torch.no_grad():
            z_targ = feat_encoder(ztarg_wav.unsqueeze(0))

        delta, adv, s, l2, linf = pgd_attack(
            wav, feat_encoder, z_targ, EPS, STEP_SIZE, N_ITERS
        )

        torch.save(delta.cpu(), delta_path)
        torchaudio.save(adv_path, adv.cpu(), sample_rate)

        total_snr += s
        total_l2 += l2
        total_linf += linf
        n_used += 1

        if (i + 1) % 50 == 0:
            print(
                f"[{split}] 已处理 {i+1}/{len(wav_files)}，"
                f"SNR={s:.2f}dB, ||δ||2={l2:.4f}, ||δ||∞={linf:.4f}"
            )

    if n_used == 0:
        print(f"[{split}] 没有成功处理任何样本。")
        return

    print(
        f"[{split}] PGD 完成，共 {n_used} 条样本，"
        f"平均 SNR={total_snr/n_used:.2f} dB, "
        f"平均 ||δ||2={total_l2/n_used:.4f}, "
        f"平均 ||δ||∞={total_linf/n_used:.4f}"
    )
    print(f"[{split}] δ 输出目录: {delta_dir}")
    print(f"[{split}] adv 输出目录: {adv_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    print("DATA_ROOT =", DATA_ROOT)
    print("ZTARG_ROOT =", ZTARG_ROOT, "=>", os.path.abspath(ZTARG_ROOT))
    print("DELTA_ROOT =", DELTA_ROOT)
    print("ADV_WAV_ROOT =", ADV_WAV_ROOT)
    print("SPLITS =", SPLITS)
    print("EPS =", EPS, "STEP_SIZE =", STEP_SIZE, "N_ITERS =", N_ITERS)
    print("MAX_PER_SPLIT =", MAX_PER_SPLIT)

    for split in SPLITS:
        print("\n========== 处理 split:", split, "==========")
        process_split(split, device)


if __name__ == "__main__":
    main()
