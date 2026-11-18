#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集中计算 AudioCaps(test) 上的 FD / IS / KL / FAD 等指标（完全不用 HuggingFace）

输入目录（collect_results_from_out_masked.py 已经生成）：
- results/real_for_eval/      : 真实语音（参考）
- results/B0_AudioLDM/        : B0 基线
- results/B1_delta_nogate/    : B1 (+δ, no gate)
- results/Ours_express_safe/  : 我们方法 safe 场景（先用 *_safe_gen）
- results/Ours_express_harm/  : 我们方法 harm 场景（先用 *_harm_ungen）

输出：
- results/metrics_*.json      : 每个模式一份 json
- results/metrics_overview.csv: 汇总表（方便直接做论文里的表）
"""

import os
import sys
import json
import csv
from pathlib import Path

import torch

# ===== 0. 把本地 audioldm_eval 加进 sys.path（数据盘） =====
ROOT = Path(__file__).resolve().parent
PKG_ROOT = ROOT / "tools" / "audioldm_eval"
PKG_SRC = PKG_ROOT / "src"

for p in (PKG_ROOT, PKG_SRC):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audioldm_eval import EvaluationHelper  # 现在可以安全导入

SAMPLE_RATE = 16000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULTS_ROOT = ROOT / "results"
REF_DIR = RESULTS_ROOT / "real_for_eval"

# 这里的目录名要和你前面 collect_results_from_out_masked.py 生成的一致
MODES = {
    "B0_AudioLDM":       RESULTS_ROOT / "B0_AudioLDM",
    "B1_delta_nogate":   RESULTS_ROOT / "B1_delta_nogate",
    "Ours_hidden":       RESULTS_ROOT / "Ours_hidden",   # <- 加这一行
    "Ours_express_safe": RESULTS_ROOT / "Ours_express_safe",
    "Ours_express_harm": RESULTS_ROOT / "Ours_express_harm",
}


def build_evaluator():
    """兼容不同版本 audioldm_eval，强制使用 cnn14（不用 MERT/HF）"""
    print("使用设备:", DEVICE)
    # 有的版本 backbone 写在 __init__，有的写在 main() 里
    # 这里两种都试一下，保证至少有一个生效
    try:
        evaluator = EvaluationHelper(SAMPLE_RATE, DEVICE, backbone="cnn14")
        print("[INFO] EvaluationHelper(..., backbone='cnn14') 创建成功")
        return evaluator, True  # True 表示 __init__ 已经吃掉了 backbone
    except TypeError:
        # 老版本可能没有 backbone 这个参数
        print("[WARN] EvaluationHelper.__init__ 不接受 backbone 参数，"
              "将改为在 main() 中传入 backbone")
        evaluator = EvaluationHelper(SAMPLE_RATE, DEVICE)
        return evaluator, False  # False 表示需要在 main() 里传 backbone


def eval_one_mode(name: str,
                  gen_dir: Path,
                  ref_dir: Path,
                  evaluator,
                  backbone_in_init: bool):
    """对某一个模式（B0 / B1 / Ours_xxx）计算指标并保存 json"""
    gen_dir = Path(gen_dir)
    ref_dir = Path(ref_dir)

    assert gen_dir.exists(), f"[ERROR] 生成目录不存在: {gen_dir}"
    assert ref_dir.exists(), f"[ERROR] 参考目录不存在: {ref_dir}"

    print(f"\n===== 模式: {name} =====")
    print(f"  生成音频目录: {gen_dir}")
    print(f"  参考音频目录: {ref_dir}")

    # 兼容不同版本的 main()
    try:
        if backbone_in_init:
            # backbone 已在 __init__ 里指定，这里就不要再传了
            metrics = evaluator.main(
                str(gen_dir),
                str(ref_dir),
                limit_num=None,
            )
        else:
            # __init__ 不吃 backbone，在 main 里传
            metrics = evaluator.main(
                str(gen_dir),
                str(ref_dir),
                backbone="cnn14",
                limit_num=None,
            )
    except TypeError:
        # 万一 main() 也不认 backbone，就退回最简单的写法
        print("[WARN] evaluator.main 不接受 backbone 参数，使用默认设置 "
              "(通常默认也是 cnn14)")
        metrics = evaluator.main(
            str(gen_dir),
            str(ref_dir),
            limit_num=None,
        )

    # 将 json 存到 results/metrics_*.json
    out_json = RESULTS_ROOT / f"metrics_{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已保存 JSON 指标到: {out_json}")

    return metrics


def main():
    assert REF_DIR.exists(), f"[ERROR] 参考目录不存在: {REF_DIR}"
    for mode, gen_dir in MODES.items():
        if not Path(gen_dir).exists():
            raise FileNotFoundError(f"{mode} 的生成目录不存在: {gen_dir}")

    evaluator, backbone_in_init = build_evaluator()

    overview_path = RESULTS_ROOT / "metrics_overview.csv"
    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode",
            "FD",          # frechet_distance
            "IS",          # is_mean
            "KL",          # kl_softmax
            "FAD",         # frechet_audio_distance
            "LSD",
            "PSNR",
            "SSIM",
            "SSIM_STFT",
            "KID_mean",
            "KID_std",
        ])

        for mode, gen_dir in MODES.items():
            metrics = eval_one_mode(
                mode,
                gen_dir,
                REF_DIR,
                evaluator,
                backbone_in_init,
            )

            row = [
                mode,
                metrics.get("frechet_distance", -1),
                metrics.get("is_mean", -1),
                metrics.get("kl_softmax", -1),
                metrics.get("frechet_audio_distance", -1),
                metrics.get("lsd", -1),
                metrics.get("psnr", -1),
                metrics.get("ssim", -1),
                metrics.get("ssim_stft", -1),
                metrics.get("kid_mean", -1),
                metrics.get("kid_std", -1),
            ]
            writer.writerow(row)
            print("[INFO] 当前模式简表:", row)

    print(f"\n[DONE] 全部模式评测完成，汇总表已写入: {overview_path}")


if __name__ == "__main__":
    main()
