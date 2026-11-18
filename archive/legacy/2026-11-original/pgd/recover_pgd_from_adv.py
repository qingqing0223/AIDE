import os
import glob
import math

import torch
import torchaudio


# ========= 配置：保持默认即可 =========

# 原始 audiocaps 语音
DATA_ROOT = "data/audiocaps_wav"

# 只在 validation / test 上恢复
SPLITS = ["validation", "test"]

# 新导出目录：里边会有 *_noise.wav 和 *_adv.wav
OUT_ROOT = "out/audiocaps_eps001_wav"


# ========= 工具函数：不用改 =========

def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    p_signal = clean.pow(2).mean().item() + 1e-8
    p_noise = noise.pow(2).mean().item() + 1e-8
    return 10.0 * math.log10(p_signal / p_noise)


def guess_adv_dir_for_split(root: str, split: str):
    """
    在 out/ 下面自动“猜”一下哪个目录是这个 split 的 adv 结果：
    条件：
      - 路径名里同时包含 "adv" 和 "audiocaps"
      - 路径名里包含 split 名（validation / test）
      - 目录下有不少 .wav 文件（>= 10）
    """
    search_root = os.path.join(root, "out")
    candidates = []

    for dirpath, dirnames, filenames in os.walk(search_root):
        rel_dir = os.path.relpath(dirpath, root)
        # 只看路径里带 split / adv / audiocaps 的
        if split not in rel_dir:
            continue
        if "adv" not in rel_dir:
            continue
        if "audiocaps" not in rel_dir:
            continue

        wavs = [f for f in filenames if f.endswith(".wav")]
        if len(wavs) >= 10:
            candidates.append((dirpath, len(wavs)))

    if not candidates:
        return None

    # 选 .wav 数量最多的那个目录
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_dir, n_wav = candidates[0]
    print(f"[{split}] 猜测 adv 目录: {os.path.relpath(best_dir, root)} (含 {n_wav} 个 wav)")
    return best_dir


def process_split(root: str, split: str):
    raw_dir = os.path.join(root, DATA_ROOT, split)
    if not os.path.isdir(raw_dir):
        print(f"[{split}] 找不到原始语音目录 {raw_dir}，跳过。")
        return

    adv_dir = guess_adv_dir_for_split(root, split)
    if adv_dir is None:
        print(f"[{split}] 在 out/ 下没有找到符合条件的 adv 目录，请检查 out 结构。")
        return

    out_dir = os.path.join(root, OUT_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)

    # 所有原始 utt_id
    raw_wavs = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    raw_ids = {os.path.splitext(os.path.basename(p))[0] for p in raw_wavs}

    # 所有 adv utt_id
    adv_wavs = sorted(glob.glob(os.path.join(adv_dir, "*.wav")))
    adv_ids = {os.path.splitext(os.path.basename(p))[0] for p in adv_wavs}

    common_ids = sorted(raw_ids & adv_ids)
    print(f"[{split}] 原始 {len(raw_ids)} 条, adv {len(adv_ids)} 条, 交集 {len(common_ids)} 条。")

    if not common_ids:
        print(f"[{split}] 没有任何 (raw, adv) 匹配，无法恢复 δ。")
        return

    total_snr = 0.0
    total_l2 = 0.0
    total_linf = 0.0

    for i, utt_id in enumerate(common_ids):
        raw_path = os.path.join(raw_dir, f"{utt_id}.wav")
        adv_path = os.path.join(adv_dir, f"{utt_id}.wav")

        wav_raw, sr1 = torchaudio.load(raw_path)
        wav_adv, sr2 = torchaudio.load(adv_path)

        if sr1 != sr2:
            print(f"[{split}] {utt_id} 采样率不一致，raw={sr1}, adv={sr2}，跳过。")
            continue

        T = min(wav_raw.shape[-1], wav_adv.shape[-1])
        wav_raw = wav_raw[..., :T]
        wav_adv = wav_adv[..., :T]

        noise = wav_adv - wav_raw

        # 统计指标
        with torch.no_grad():
            l2 = noise.view(-1).norm(p=2).item()
            linf = noise.view(-1).abs().max().item()
            s = snr_db(wav_raw, wav_adv)

        total_snr += s
        total_l2 += l2
        total_linf += linf

        noise_path = os.path.join(out_dir, f"{utt_id}_noise.wav")
        adv_out_path = os.path.join(out_dir, f"{utt_id}_adv.wav")

        torchaudio.save(noise_path, noise, sr1)
        # 再保存一份 adv 到统一目录，便于后续画图 / 对齐
        torchaudio.save(adv_out_path, wav_adv, sr1)

        if (i + 1) % 100 == 0:
            print(f"[{split}] 已处理 {i+1}/{len(common_ids)}")

    n = len(common_ids)
    if n == 0:
        print(f"[{split}] 最终没有成功恢复任何样本。")
        return

    print(
        f"[{split}] 恢复完成，共 {n} 条样本，"
        f"平均 SNR={total_snr/n:.2f} dB, "
        f"平均 ||δ||2={total_l2/n:.4f}, "
        f"平均 ||δ||∞={total_linf/n:.4f}"
    )
    print(f"[{split}] 导出目录：{out_dir}")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("ROOT =", root)
    print("DATA_ROOT =", DATA_ROOT)
    print("OUT_ROOT =", OUT_ROOT)

    for split in SPLITS:
        print(f"\n========== 处理 split: {split} ==========")
        process_split(root, split)


if __name__ == "__main__":
    main()
