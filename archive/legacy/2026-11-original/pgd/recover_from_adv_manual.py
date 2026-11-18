import os
import glob
import math
import argparse

import torch
import torchaudio


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise = noisy - clean
    p_signal = clean.pow(2).mean().item() + 1e-8
    p_noise = noise.pow(2).mean().item() + 1e-8
    return 10.0 * math.log10(p_signal / p_noise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="原始 wav 所在目录，例如 data/audiocaps_wav/validation")
    parser.add_argument("--adv_dir", type=str, required=True,
                        help="PGD 生成的对抗 wav 所在目录，例如 out/adv_audiocaps_eps001_validation")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="导出 noise/adv 的目录，例如 out/audiocaps_eps001_wav/validation")
    args = parser.parse_args()

    raw_dir = args.raw_dir
    adv_dir = args.adv_dir
    out_dir = args.out_dir

    print("raw_dir:", raw_dir)
    print("adv_dir:", adv_dir)
    print("out_dir:", out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # 所有原始 wav
    raw_paths = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    raw_ids = {os.path.splitext(os.path.basename(p))[0] for p in raw_paths}

    # 所有 adv wav
    adv_paths = sorted(glob.glob(os.path.join(adv_dir, "*.wav")))
    adv_ids = {os.path.splitext(os.path.basename(p))[0] for p in adv_paths}

    common_ids = sorted(raw_ids & adv_ids)
    print(f"原始 {len(raw_ids)} 条, adv {len(adv_ids)} 条, 交集 {len(common_ids)} 条。")

    if not common_ids:
        print("没有任何 (raw, adv) 匹配，请检查 raw_dir 和 adv_dir 是否对应同一个 split。")
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
            print(f"{utt_id}: 采样率不一致 raw={sr1}, adv={sr2}，跳过。")
            continue

        T = min(wav_raw.shape[-1], wav_adv.shape[-1])
        wav_raw = wav_raw[..., :T]
        wav_adv = wav_adv[..., :T]

        noise = wav_adv - wav_raw

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
        torchaudio.save(adv_out_path, wav_adv, sr1)

        if (i + 1) % 100 == 0:
            print(f"已处理 {i+1}/{len(common_ids)}")

    n = len(common_ids)
    if n == 0:
        print("最终没有成功恢复任何样本。")
        return

    print(
        f"恢复完成，共 {n} 条样本，"
        f"平均 SNR={total_snr/n:.2f} dB, "
        f"平均 ||δ||2={total_l2/n:.4f}, "
        f"平均 ||δ||∞={total_linf/n:.4f}"
    )
    print("导出目录:", out_dir)


if __name__ == "__main__":
    main()
