from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import argparse
import json
import ast
import sys
import numpy as np
import soundfile as sf

# 工程根目录（在数据盘）
ROOT = Path("/autodl-tmp/speech_guard").resolve()

# 我们自己写的 AudioLDM 封装在 src/audioldm_api.py
sys.path.insert(0, str(ROOT / "src"))
from audioldm_api import call_audioldm  # noqa: E402


def load_metrics(jsonl_path: Path) -> List[Dict[str, Any]]:
    """读取 adv_audiocaps_eps001_full/test/metrics.jsonl"""
    items: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # metrics.jsonl 有可能是 python dict 字符串
                obj = ast.literal_eval(line)
            items.append(obj)
    if not items:
        raise RuntimeError(f"No records found in {jsonl_path}")
    return items


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """只用 numpy 做一个线性插值重采样，避免再引入 librosa 报错。"""
    if sr_in == sr_out:
        return x

    # 转成单声道
    if x.ndim > 1:
        x = x.mean(axis=1)

    n_in = x.shape[0]
    duration = n_in / float(sr_in)
    n_out = int(round(duration * sr_out))

    t_in = np.linspace(0.0, duration, n_in, endpoint=False, dtype=np.float64)
    t_out = np.linspace(0.0, duration, n_out, endpoint=False, dtype=np.float64)

    y = np.interp(t_out, t_in, x.astype(np.float64))
    return y.astype(np.float32)


def build_paths(sample_id: str) -> Dict[str, Path]:
    """根据 sample_id 拼出 raw / adv / noise 的路径"""
    raw = ROOT / "data" / "audiocaps_wav" / "test" / f"{sample_id}.wav"

    adv_root = ROOT / "out" / "adv_audiocaps_eps001_full" / "test" / "wav"
    adv = adv_root / f"{sample_id}_adv.wav"
    noise = adv_root / f"{sample_id}_noise.wav"

    return {"raw": raw, "adv": adv, "noise": noise}


def prepare_case_wavs(paths: Dict[str, Path], case_dir: Path) -> int:
    """
    把 raw / adv / noise 统一采样率后写到 case 目录下：
      raw.wav, adv.wav, noise.wav
    返回统一后的采样率 sr_target
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    raw, sr_raw = sf.read(paths["raw"])
    adv, sr_adv = sf.read(paths["adv"])

    # 转单声道
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if adv.ndim > 1:
        adv = adv.mean(axis=1)

    sr_target = int(sr_adv)

    # 重采样 raw -> sr_target
    raw_rs = resample_to(raw, int(sr_raw), sr_target)

    # 处理 noise：如果有文件，用文件；否则直接用 adv - raw_rs
    if paths["noise"].exists():
        noise, sr_noise = sf.read(paths["noise"])
        if noise.ndim > 1:
            noise = noise.mean(axis=1)
        noise_rs = resample_to(noise, int(sr_noise), sr_target)
    else:
        noise_rs = (adv - raw_rs).astype(np.float32)

    # 写到 case 目录
    sf.write(case_dir / "raw.wav", raw_rs, sr_target)
    sf.write(case_dir / "adv.wav", adv, sr_target)
    sf.write(case_dir / "noise.wav", noise_rs, sr_target)

    return sr_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one AudioLDM case on adv_audiocaps_eps001_full/test"
    )
    parser.add_argument("--metrics_jsonl", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--harm_prompt", type=str, required=True)
    parser.add_argument("--safe_prompt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--model_name", type=str, default="audioldm-s-full-v2")
    parser.add_argument("--write_gate", action="store_true")
    args = parser.parse_args()

    jsonl_path = Path(args.metrics_jsonl)
    items = load_metrics(jsonl_path)

    if args.index < 0 or args.index >= len(items):
        raise IndexError(f"index {args.index} out of range, got {len(items)} samples")

    record = items[args.index]
    sample_id = str(record.get("id"))
    print(f"[INFO] Selected index={args.index}, id={sample_id}")

    paths = build_paths(sample_id)
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"{name} wav not found: {p}")

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    case_dir = out_root / f"case_{args.index}"

    # 统一采样率 & 写 raw/adv/noise
    sr = prepare_case_wavs(paths, case_dir)
    print(f"[INFO] Unified sample rate: {sr} Hz")

    # ===================== AudioLDM 生成 UNGEN / GEN =====================
    # harmful prompt -> UNGEN（有害）
    print(f"[INFO] Generating UNGEN with AudioLDM model={args.model_name}")
    ungen_audio, sr_model = call_audioldm(
        prompt=args.harm_prompt,
        duration=args.duration,
        model_name=args.model_name,
    )
    if sr_model != sr:
        ungen_audio = resample_to(ungen_audio, sr_model, sr)
    sf.write(case_dir / "ungen.wav", ungen_audio, sr)

    # safe prompt -> GEN（安全）
    print(f"[INFO] Generating GEN with AudioLDM model={args.model_name}")
    gen_audio, sr_model2 = call_audioldm(
        prompt=args.safe_prompt,
        duration=args.duration,
        model_name=args.model_name,
    )
    if sr_model2 != sr:
        gen_audio = resample_to(gen_audio, sr_model2, sr)
    sf.write(case_dir / "gen.wav", gen_audio, sr)

    # ===================== 写 gate / meta =====================
    gate = 1.0 if args.write_gate else 0.0
    meta = {
        "id": sample_id,
        "index": args.index,
        "sr": sr,
        "duration": args.duration,
        "harm_prompt": args.harm_prompt,
        "safe_prompt": args.safe_prompt,
        "gate": gate,  # 有害任务门控，后面训练 / 论文里可以直接用
    }
    meta_path = out_root / "meta_all.jsonl"
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"[INFO] Done. Case dir = {case_dir}")
    print(f"[INFO] Meta appended to {meta_path}")


if __name__ == "__main__":
    main()
