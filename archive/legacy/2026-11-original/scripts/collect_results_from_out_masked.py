#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 out_masked/ 里把各模式的 wav 收集到 results/* 目录：
- real_for_eval/          : 原始 raw（作真分布）
- B0_AudioLDM/            : safe_gen（baseline）
- B1_delta_nogate/        : adv（只加 δ，未门控）
- Ours_express_safe/      : *_safe_gen（我们方法在 safe prompt 场景下的输出）
- Ours_express_harm/      : *_harm_ungen（我们方法在 harm prompt 场景下的输出）
Ours_hidden 后面门控脚本里再生成，这里先留空目录。
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_MASKED = ROOT / "out_masked"

RESULTS_ROOT = ROOT / "results"
REAL_DIR     = RESULTS_ROOT / "real_for_eval"
B0_DIR       = RESULTS_ROOT / "B0_AudioLDM"
B1_DIR       = RESULTS_ROOT / "B1_delta_nogate"
OURS_SAFE    = RESULTS_ROOT / "Ours_express_safe"
OURS_HARM    = RESULTS_ROOT / "Ours_express_harm"
OURS_HIDDEN  = RESULTS_ROOT / "Ours_hidden"   # 先建空

for d in [REAL_DIR, B0_DIR, B1_DIR, OURS_SAFE, OURS_HARM, OURS_HIDDEN]:
    d.mkdir(parents=True, exist_ok=True)

def copy_if_exists(src: Path, dst_dir: Path):
    if not src.exists():
        print(f"[WARN] {src} not exists, skip.")
        return
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"[COPY] {src} -> {dst}")

def main():
    sample_dirs = sorted(OUT_MASKED.glob("*"))
    sample_dirs = [d for d in sample_dirs if d.is_dir()]

    print(f"[INFO] Found {len(sample_dirs)} sample dirs under {OUT_MASKED}")

    for sd in sample_dirs:
        sid = sd.name

        raw  = sd / f"{sid}_raw.wav"
        adv  = sd / f"{sid}_adv.wav"
        safe_gens = list(sd.glob(f"{sid}_*_safe_gen.wav"))
        harm_ungs = list(sd.glob(f"{sid}_*_harm_ungen.wav"))

        # real 分布：原始 raw
        copy_if_exists(raw, REAL_DIR)

        # B0_AudioLDM：这里用 safe_gen，当 baseline
        if safe_gens:
            copy_if_exists(safe_gens[0], B0_DIR)
            copy_if_exists(safe_gens[0], OURS_SAFE)
        else:
            print(f"[WARN] no safe_gen for {sid}")

        # B1_delta_nogate：adv（加 δ，无门控）
        copy_if_exists(adv, B1_DIR)

        # Ours_express_harm：harm_ungen
        if harm_ungs:
            copy_if_exists(harm_ungs[0], OURS_HARM)
        else:
            print(f"[WARN] no harm_ungen for {sid}")

    print("[INFO] done collecting wavs into results/*")

if __name__ == "__main__":
    main()
