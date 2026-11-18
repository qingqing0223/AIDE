import os
import glob
import math

import torch
import torchaudio


# ===================== 按你的实际情况改这里 =====================

DATA_ROOT = "data/audiocaps_wav"

# 就用 validation + test 两个 split
SPLITS = ["validation", "test"]

# ★★★ 这里一定要改成 locate_pgd_delta.py 输出的 root: 后面的那一串 ★★★
# 比如 locate 脚本打印的是:
#   root: out/adv_audiocaps_eps001
# 就写:
#   PGD_DELTA_ROOT = "out/adv_audiocaps_eps001"
PGD_DELTA_ROOT = "out/adv_audiocaps_eps001"  # <<< 把这个改成你终端里看到的 root

OUT_ROOT = "out/audiocaps_eps001_wav"

SCALE = 1.0

# ===================== 下面不用改 =====================


def load_delta(delta_path: str) -> torch.Tensor:
    if delta_path.endswith(".npy"):
        import numpy as np

        arr = np.load(delta_path)
        return torch.from_numpy(arr)

    obj = torch.load(delta_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ["delta", "noise", "perturb", "adv_noise"]:
            if k in obj:
                return obj[k]
        if "x_adv" in obj and "x" in obj:
            return obj["x_adv"] - obj["x"]
    raise ValueError(f"无法从 {delta_path} 中解析 delta，内容类型: {type(obj)}")


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    p_signal = clean.pow(2).mean().item() + 1e-8
    p_noise = noise.pow(2).mean().item() + 1e-8
    return 10.0 * math.log10(p_signal / p_noise)


def process_split(split: str):
    raw_dir = os.path.join(DATA_ROOT, split)
    delta_dir = os.path.join(PGD_DELTA_ROOT, split)
    out_dir = os.path.join(OUT_ROOT, split)

    if not os.path.isdir(raw_dir):
        print(f"[{split}] 跳过：找不到原始语音目录 {raw_dir}")
        return
    if not os.path.isdir(delta_dir):
        print(f"[{split}] 跳过：找不到 PGD δ 目录 {delta_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    delta_map = {}
    for ext in ("*.pt", "*.pth", "*.npy"):
        for p in glob.glob(os.path.join(delta_dir, ext)):
            base = os.path.splitext(os.path.basename(p))[0]
            utt_id = base.split("_")[0]
            if utt_id not in delta_map:
                delta_map[utt_id] = p

    print(f"[{split}] 在 {delta_dir} 下找到 {len(delta_map)} 个 δ 文件")

    wav_paths = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    print(f"[{split}] 在 {raw_dir} 下找到 {len(wav_paths)} 个原始 wav")

    if not wav_paths:
        print(f"[{split}] 没有原始 wav，直接返回。")
        return

    total_snr = 0.0
    total_l2 = 0.0
    total_linf = 0.0
    n_used = 0

    for wav_path in wav_paths:
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        if utt_id not in delta_map:
            continue

        delta_path = delta_map[utt_id]

        wav_raw, sr = torchaudio.load(wav_path)
        delta = load_delta(delta_path)

        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        if delta.shape[0] != wav_raw.shape[0]:
            delta = delta[:1, :]

        T = min(wav_raw.shape[-1], delta.shape[-1])
        wav_raw = wav_raw[..., :T]
        delta = delta[..., :T]

        noise = SCALE * delta
        adv = wav_raw + noise

        max_val = max(
            wav_raw.abs().max().item(),
            noise.abs().max().item(),
            adv.abs().max().item(),
        )
        if max_val > 1.0:
            wav_raw = wav_raw / max_val
            adv = adv / max_val
            noise = adv - wav_raw

        with torch.no_grad():
            l2 = noise.view(-1).norm(p=2).item()
            linf = noise.view(-1).abs().max().item()
            s = snr_db(wav_raw, adv)

        total_snr += s
        total_l2 += l2
        total_linf += linf
        n_used += 1

        noise_path = os.path.join(out_dir, f"{utt_id}_noise.wav")
        adv_path = os.path.join(out_dir, f"{utt_id}_adv.wav")

        torchaudio.save(noise_path, noise, sr)
        torchaudio.save(adv_path, adv, sr)

    if n_used == 0:
        print(f"[{split}] 没能匹配到任何 (raw, δ) 对，请检查文件命名。")
        return

    print(
        f"[{split}] 导出完成，共 {n_used} 条样本，"
        f"平均 SNR={total_snr/n_used:.2f} dB, "
        f"平均 ||δ||2={total_l2/n_used:.4f}, "
        f"平均 ||δ||∞={total_linf/n_used:.4f}"
    )
    print(f"[{split}] 导出目录：{out_dir}")


def main():
    print("DATA_ROOT =", DATA_ROOT)
    print("PGD_DELTA_ROOT =", PGD_DELTA_ROOT)
    print("OUT_ROOT =", OUT_ROOT)

    for split in SPLITS:
        print("\n========== 处理 split:", split, "==========")
        process_split(split)


if __name__ == "__main__":
    main()
