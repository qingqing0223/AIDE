import os
import argparse
from pathlib import Path
from offline_torchhub_patch import patch_torchhub
patch_torchhub()

def _ensure_offline(torch_home: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", torch_home)

def _check_weights_exist():
    import torch
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    need_any = [
        "Cnn14_16k_mAP=0.438.pth",
    ]
    missing = [f for f in need_any if not (ckpt_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"[FATAL] torch.hub checkpoints 缺少: {missing}\n"
            f"torch.hub.get_dir()={torch.hub.get_dir()}\n"
            f"请把这些文件放到: {ckpt_dir}\n"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--real_subdir", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--dtype", type=str, default="float32")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parent
    torch_home = os.environ.get("TORCH_HOME", str(proj / "torch_cache"))
    _ensure_offline(torch_home)

    # 依赖包
    import pandas as pd
    from frechet_audio_distance import FrechetAudioDistance

    results_root = Path(args.results_root)
    real_dir = results_root / args.real_subdir
    if not real_dir.exists():
        raise FileNotFoundError(f"[FATAL] real_dir 不存在: {real_dir}")

    df = pd.read_csv(args.metrics_csv)

    # 确保列存在
    for col in ["FAD_vggish", "FAD_pann"]:
        if col not in df.columns:
            df[col] = ""

    # 先检查 pann 权重是否能被 torch.hub 找到（否则一定会联网）
    _check_weights_exist()

    # 初始化两个 FAD（如果本地权重齐全，不会联网）
    fad_vggish = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=args.sample_rate,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )
    fad_pann = FrechetAudioDistance(
        model_name="pann",
        sample_rate=args.sample_rate,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    # 逐行算（mode 列必须是结果子目录名）
    for i, row in df.iterrows():
        mode = str(row["mode"])
        fake_dir = results_root / mode
        if not fake_dir.exists():
            print(f"[WARN] fake_dir 不存在，跳过: {fake_dir}")
            continue

        print(f"\n=== [{i+1}/{len(df)}] mode={mode} ===")
        print(f"real: {real_dir}")
        print(f"fake: {fake_dir}")

        # 计算
        v = fad_vggish.score(str(real_dir), str(fake_dir), dtype=args.dtype)
        p = fad_pann.score(str(real_dir), str(fake_dir), dtype=args.dtype)

        df.at[i, "FAD_vggish"] = float(v)
        df.at[i, "FAD_pann"] = float(p)

        print(f"[OK] FAD_vggish={v:.6f} | FAD_pann={p:.6f}")

    out = Path(args.output_csv)
    df.to_csv(out, index=False)
    print(f"\n[DONE] 写出: {out.resolve()}")

if __name__ == "__main__":
    main()
