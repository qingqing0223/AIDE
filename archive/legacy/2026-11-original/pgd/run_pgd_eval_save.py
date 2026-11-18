import os
import argparse

import torch
import torchaudio
from torch import nn


# ========== 需要你按原来 PGD 训练代码补全的几个函数 ==========

def load_encoder_and_target(device):
    """
    TODO: 把你原来 PGD 攻击时用到的 encoder / target 载入逻辑粘过来：

    - encoder(x) -> z_x  （比如 AudioLDM 里 CLAP 或 text/audio encoder）
    - z_targ: 目标 latent，可以从磁盘加载，或者按照你原来的方式选取

    返回:
        encoder: 一个 nn.Module，把 waveform -> latent
        z_targ_dict: {utt_id: z_targ_tensor} 或者一个函数 get_z_targ(utt_id)
    """
    raise NotImplementedError


def compute_pgd_loss(z_x, z_targ):
    """
    TODO: 这里写你原来 PGD 的 loss（比如在 latent 空间里 ||z_x - z_targ||2）
    """
    raise NotImplementedError


# ========== 以下是通用 PGD 框架 + 保存逻辑，不需要改 ==========

def pgd_attack(x, encoder, z_targ, eps=0.01, step_size=0.001, iters=20):
    """
    在 waveform 空间上做 PGD，返回 delta 和 adv = x + delta
    """
    device = x.device
    delta = torch.zeros_like(x, requires_grad=True).to(device)

    for _ in range(iters):
        adv = x + delta
        z_adv = encoder(adv)
        loss = compute_pgd_loss(z_adv, z_targ)

        loss.backward()
        with torch.no_grad():
            grad = delta.grad.sign()
            delta += step_size * grad
            delta = torch.clamp(delta, -eps, eps)
            delta.grad.zero_()

    adv = x + delta
    return delta.detach(), adv.detach()


def process_split(split, args, encoder, z_targ_dict, device):
    data_root = args.data_root              # data/audiocaps_wav
    raw_dir = os.path.join(data_root, split)
    out_root = args.out_root                # out/pgd_audiocaps_eps001
    delta_dir = os.path.join(out_root, split, "delta")
    adv_dir = os.path.join(out_root, split, "adv")

    os.makedirs(delta_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    wav_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".wav")])
    print(f"[{split}] 找到 {len(wav_files)} 条 wav")

    for i, fname in enumerate(wav_files):
        utt_id = os.path.splitext(fname)[0]
        raw_path = os.path.join(raw_dir, fname)

        # 如果已经算过就跳过，方便断点续跑
        delta_path = os.path.join(delta_dir, f"{utt_id}.pt")
        adv_path = os.path.join(adv_dir, f"{utt_id}.wav")
        if os.path.exists(delta_path) and os.path.exists(adv_path):
            continue

        wav, sr = torchaudio.load(raw_path)
        wav = wav.to(device)

        # 取对应的 z_targ（根据你实现 load_encoder_and_target 的方式来）
        if isinstance(z_targ_dict, dict):
            z_targ = z_targ_dict[utt_id].to(device)
        else:
            # 如果是函数
            z_targ = z_targ_dict(utt_id).to(device)

        delta, adv = pgd_attack(
            wav,
            encoder,
            z_targ,
            eps=args.eps,
            step_size=args.step_size,
            iters=args.iters,
        )

        # 保存 delta
        torch.save(delta.cpu(), delta_path)
        # 保存 adv.wav（方便以后直接评估 / 画图）
        torchaudio.save(adv_path, adv.cpu(), sr)

        if (i + 1) % 50 == 0:
            print(f"[{split}] 已处理 {i+1}/{len(wav_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/audiocaps_wav")
    parser.add_argument("--out_root", type=str, default="out/pgd_audiocaps_eps001")
    parser.add_argument("--splits", type=str, default="validation,test")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--step_size", type=float, default=0.001)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, z_targ_dict = load_encoder_and_target(device)
    encoder.to(device)
    encoder.eval()

    for split in args.splits.split(","):
        split = split.strip()
        print(f"====== 处理 split: {split} ======")
        process_split(split, args, encoder, z_targ_dict, device)


if __name__ == "__main__":
    main()
