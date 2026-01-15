#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

import torch
from diffusers import AudioLDM2Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    if not items:
        raise RuntimeError(f"No records found in {path}")
    return items


def load_wav(path, target_sr=None):
    """用 soundfile 读 wav，必要时重采样到 target_sr。"""
    path = Path(path)
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if target_sr is None or sr == target_sr:
        return audio, sr

    # 简单重采样（线性插值，够用做可视化和叠加）
    duration = len(audio) / sr
    t_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    t_new = np.linspace(0, duration, num=int(duration * target_sr), endpoint=False)
    audio_new = np.interp(t_new, t_old, audio).astype(np.float32)
    return audio_new, target_sr


def save_wav(path, audio, sr):
    path = Path(path)
    sf.write(path, audio, sr)


def apply_gate(gen_audio, raw_audio, adv_audio, gate):
    """
    gate=0: 完全不表达扰动 (只保留 gen_audio)
    gate=1: 完全表达扰动 (gen_audio + (adv - raw))
    """
    # 对齐长度
    L = min(len(raw_audio), len(adv_audio))
    raw = raw_audio[:L]
    adv = adv_audio[:L]
    delta = adv - raw

    if len(gen_audio) < L:
        gen = np.pad(gen_audio, (0, L - len(gen_audio)))
    else:
        gen = gen_audio[:L]

    out = gen + gate * delta

    max_abs = np.max(np.abs(out))
    if max_abs > 1e-8 and max_abs > 1.0:
        out = out / max_abs
    return out.astype(np.float32)


def plot_waveforms(case_dir, sr):
    """画 raw/adv/gen/ungen 四行波形图，方便快速肉眼对比。"""
    case_dir = Path(case_dir)
    tags = [("raw", "raw.wav"),
            ("adv", "adv.wav"),
            ("gen", "gen.wav"),
            ("ungen", "ungen.wav")]

    fig, axes = plt.subplots(len(tags), 1, figsize=(8, 2.5 * len(tags)), sharex=True)
    for ax, (name, fname) in zip(axes, tags):
        wav_path = case_dir / fname
        if not wav_path.exists():
            ax.set_title(f"{name} - MISSING")
            ax.axis("off")
            continue
        audio, _ = load_wav(wav_path, target_sr=sr)
        t = np.arange(len(audio)) / sr
        ax.plot(t, audio)
        ax.set_ylabel("Amp.")
        ax.set_title(f"{name} - Waveform")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(case_dir / "waveforms_4x1.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--harm_prompt", type=str, required=True)
    parser.add_argument("--safe_prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="cvssp/audioldm2-large",
        help="HuggingFace 模型名，例如 cvssp/audioldm2-large",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    metrics_path = root / args.metrics_jsonl
    records = load_jsonl(metrics_path)
    if not (0 <= args.index < len(records)):
        raise IndexError(f"index {args.index} out of range 0..{len(records)-1}")

    rec = records[args.index]
    sample_id = str(rec["id"])

    print(f"[INFO] Selected index={args.index}, id={sample_id}")

    # ====== 找 raw / adv / noise 路径 ======
    data_root = root / "data" / "audiocaps_wav" / "test"
    raw_path = data_root / f"{sample_id}.wav"

    adv_root = root / "out" / "adv_audiocaps_eps001_full" / "test" / "wav"
    adv_path = adv_root / f"{sample_id}_adv.wav"
    noise_path = adv_root / f"{sample_id}_noise.wav"

    if not raw_path.exists():
        raise FileNotFoundError(f"raw wav not found: {raw_path}")
    if not adv_path.exists():
        raise FileNotFoundError(f"adv wav not found: {adv_path}")

    # ====== 输出目录准备 ======
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 把 raw / adv / noise 复制一份到 case 目录，便于自包含
    shutil.copy2(raw_path, out_dir / "raw.wav")
    shutil.copy2(adv_path, out_dir / "adv.wav")
    if noise_path.exists():
        shutil.copy2(noise_path, out_dir / "noise.wav")

    # ====== 读取 raw / adv，供 gate 使用 ======
    target_sr = 16000
    raw_audio, sr = load_wav(raw_path, target_sr=target_sr)
    adv_audio, _ = load_wav(adv_path, target_sr=target_sr)

    # ====== 加载 AudioLDM 模型 ======
    print(f"[INFO] Loading AudioLDM model: {args.model_name}")
    pipe = AudioLDM2Pipeline.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to(args.device)

    # ====== 生成 gen（安全 prompt） ======
    gen_seed = args.seed + args.index
    print(f"[INFO] Generating GEN with safe_prompt, seed={gen_seed}")
    g = torch.Generator(device=args.device).manual_seed(gen_seed)
    gen_res = pipe(
        args.safe_prompt,
        audio_length_in_s=args.duration,
        num_inference_steps=50,
        guidance_scale=3.5,
        generator=g,
    )
    gen_audio = np.array(gen_res.audios[0], dtype=np.float32)

    # ====== 生成 harmful 基底（还没加 gate） ======
    ungen_seed = args.seed + 10000 + args.index
    print(f"[INFO] Generating UNGEN base with harm_prompt, seed={ungen_seed}")
    g2 = torch.Generator(device=args.device).manual_seed(ungen_seed)
    ungen_base = pipe(
        args.harm_prompt,
        audio_length_in_s=args.duration,
        num_inference_steps=50,
        guidance_scale=3.5,
        generator=g2,
    )
    ungen_audio_base = np.array(ungen_base.audios[0], dtype=np.float32)

    # ====== 门控 g：safe=0, harm=1 ======
    gate_safe = 0.0
    gate_harm = 1.0

    gen_out = apply_gate(gen_audio, raw_audio, adv_audio, gate_safe)
    ungen_out = apply_gate(ungen_audio_base, raw_audio, adv_audio, gate_harm)

    # 保存生成结果
    save_wav(out_dir / "gen.wav", gen_out, target_sr)
    save_wav(out_dir / "ungen.wav", ungen_out, target_sr)

    # 画 4 行波形图
    plot_waveforms(out_dir, target_sr)

    # 写 meta 信息（含门控 g）
    meta = {
        "id": sample_id,
        "harm_prompt": args.harm_prompt,
        "safe_prompt": args.safe_prompt,
        "raw_path": "raw.wav",
        "adv_path": "adv.wav",
        "noise_path": "noise.wav" if (out_dir / "noise.wav").exists() else None,
        "gen_path": "gen.wav",
        "ungen_path": "ungen.wav",
        "gate_safe": float(gate_safe),
        "gate_harm": float(gate_harm),
        "model_name": args.model_name,
        "duration": float(args.duration),
        "metrics": rec,
    }
    with open(out_dir / "meta_all.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved all wavs, figure and meta to {out_dir}")


if __name__ == "__main__":
    main()
