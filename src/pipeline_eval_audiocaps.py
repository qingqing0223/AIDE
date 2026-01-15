import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# 可视化：调用你之前写好的 6/7 视图脚本
from visualize_audio_views import save_all_views

try:
    # 用真正的 AudioLDM
    from audioldm import text_to_audio
except ImportError as e:
    raise ImportError(
        "没有找到 audioldm 包，请先在当前环境中安装：pip install audioldm==0.0.6"
    ) from e


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("1", "true", "yes", "y", "t"):
        return True
    if v in ("0", "false", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run AudioLDM-based gen/ungen pipeline on one AudioCaps sample."
    )
    p.add_argument("--metrics_jsonl", type=str, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--harm_prompt", type=str, required=True)
    p.add_argument("--safe_prompt", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--model_name", type=str, default="audioldm-s-full-v2")

    # 新增的两个参数：用 True / False 形式传进来也没问题
    p.add_argument("--use_audioldm", type=str2bool, default=False)
    p.add_argument("--write_gate", type=str2bool, default=False)

    return p.parse_args()


def load_metrics(jsonl_path: Path):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    if not items:
        raise RuntimeError(f"No records found in {jsonl_path}")
    return items


def is_harmful_prompt(text: str) -> bool:
    """非常简单的 keyword 门控：只要包含这些词就视为有害。"""
    t = text.lower()
    keywords = ["bomb", "attack", "weapon", "kill", "cheat", "drug", "poison", "terror"]
    return any(k in t for k in keywords)


def gen_views(wav_path: Path, out_dir: Path, tag: str):
    """
    为单个 wav 画 6/7 张图：wave / cqt / chroma / mel / stft / f0 / mps(如果你在可视化里有).
    输出目录类似 case_0/fig_gen / fig_ungen / fig_raw / fig_adv
    """
    fig_dir = out_dir / f"fig_{tag}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 你的 visualize_audio_views.save_all_views(wav, out_dir) 之前已经验证过
    save_all_views(str(wav_path), str(fig_dir))


def main():
    args = parse_args()

    metrics_jsonl = Path(args.metrics_jsonl).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = load_metrics(metrics_jsonl)
    if not (0 <= args.index < len(metrics_list)):
        raise IndexError(
            f"index={args.index} 越界，metrics.jsonl 里只有 {len(metrics_list)} 条记录"
        )
    rec = metrics_list[args.index]
    sample_id = str(rec.get("id"))
    print(f"[INFO] Selected index={args.index}, id={sample_id}")

    # === 1. 找到 raw / adv / noise 三个 wav 路径 ===
    test_root = metrics_jsonl.parent                 # .../out/adv_audiocaps_eps001_full/test
    wav_root = test_root / "wav"                     # .../test/wav
    raw_src = Path("data/audiocaps_wav/test") / f"{sample_id}.wav"
    adv_src = wav_root / f"{sample_id}_adv.wav"
    noise_src = wav_root / f"{sample_id}_noise.wav"

    for p in [raw_src, adv_src, noise_src]:
        if not p.exists():
            raise FileNotFoundError(f"缺少 wav 文件: {p}")

    # 读取三种 wav
    raw, sr_raw = sf.read(raw_src)
    adv, sr_adv = sf.read(adv_src)
    noise, sr_noise = sf.read(noise_src)

    # === 2. 把 raw / adv / noise 复制到 case 目录 ===
    raw_out = out_dir / "raw.wav"
    adv_out = out_dir / "adv.wav"
    noise_out = out_dir / "noise.wav"

    sf.write(raw_out, raw, sr_raw)
    sf.write(adv_out, adv, sr_adv)
    sf.write(noise_out, noise, sr_noise)

    # === 3. 用 AudioLDM 生成 gen / ungen ===
    gen_out = out_dir / "gen.wav"
    ungen_out = out_dir / "ungen.wav"

    if args.use_audioldm:
        print("[INFO] Using AudioLDM to generate gen.wav / ungen.wav ...")

        # gen：安全 prompt -> 正常语音
        gen_wav = text_to_audio(
            args.safe_prompt,
            model_name=args.model_name,
            duration=args.duration,
        )
        # AudioLDM 默认采样率 16000
        sf.write(gen_out, gen_wav, 16000)

        # ungen：针对有害 prompt，不直接照着 prompt 生成，
        # 而是通过“防御 prompt”强迫模型生成杂音 / 非预期内容
        defense_prompt = (
            "white noise, heavily distorted, random static, unintelligible sound, "
            "no clear speech, not following the harmful instruction"
        )
        ungen_wav = text_to_audio(
            defense_prompt,
            model_name=args.model_name,
            duration=args.duration,
        )
        sf.write(ungen_out, ungen_wav, 16000)
    else:
        print("[WARN] use_audioldm=False，只做占位复制（不推荐）。")
        sf.write(gen_out, adv, sr_adv)
        sf.write(ungen_out, raw, sr_raw)

    # === 4. 画图：raw / adv / gen / ungen 各 6~7 张 ===
    gen_views(raw_out, out_dir, "raw")
    gen_views(adv_out, out_dir, "adv")
    gen_views(gen_out, out_dir, "gen")
    gen_views(ungen_out, out_dir, "ungen")

    # === 5. 写 meta_all.json，包含门控 g 等 ===
    g = 1.0 if (args.write_gate and is_harmful_prompt(args.harm_prompt)) else 0.0

    meta = {
        "id": sample_id,
        "snr_db": rec.get("snr_db", None),
        "lsd_db": rec.get("lsd_db", None),
        "harm_prompt": args.harm_prompt,
        "safe_prompt": args.safe_prompt,
        "gate_g": g,                      # PSM 门控：目前用 keyword 规则近似
        "use_audioldm": bool(args.use_audioldm),
        "model_name": args.model_name,
        "duration": float(args.duration),
        "raw_wav": "raw.wav",
        "adv_wav": "adv.wav",
        "noise_wav": "noise.wav",
        "gen_wav": "gen.wav",
        "ungen_wav": "ungen.wav",
    }

    with open(out_dir / "meta_all.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved all wavs, figures and meta to {out_dir}")


if __name__ == "__main__":
    main()
