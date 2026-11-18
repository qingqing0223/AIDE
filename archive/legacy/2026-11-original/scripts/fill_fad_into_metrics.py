import os
import argparse
from pathlib import Path

import pandas as pd
import torch
from frechet_audio_distance import FrechetAudioDistance

def ensure_offline():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def set_torch_hub_dir(torch_home: Path):
    # 关键：强制 torch.hub.get_dir() 指到你指定的位置（不依赖 shell 是否 export 成功）
    hub_dir = torch_home / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub_dir))
    return hub_dir

def must_exist(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"[MISSING] {path}\n{hint}")

def count_wavs(d: Path) -> int:
    if not d.exists():
        return 0
    n = 0
    for _ in d.rglob("*.wav"):
        n += 1
    return n

def main():
    ensure_offline()

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--real_subdir", type=str, required=True)
    ap.add_argument("--metrics_csv", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--dtype", type=str, default="float32")
    args = ap.parse_args()

    proj = Path(__file__).resolve().parent
    torch_home = Path(os.environ.get("TORCH_HOME", str(proj / "torch_cache"))).resolve()
    hub_dir = set_torch_hub_dir(torch_home)
    ckpt = hub_dir / "checkpoints"

    # ---- 必备权重检查：缺任何一个都会触发下载（你这里没网就会报错） ----
    pann_w = ckpt / "Cnn14_16k_mAP=0.438.pth"
    must_exist(
        pann_w,
        hint=(
            "把你上传的权重复制到：\n"
            f"  {pann_w}\n"
            "命令：\n"
            "  cp -f /autodl-tmp/speech_guard/Cnn14_16k_mAP=0.438.pth $TORCH_HOME/hub/checkpoints/\n"
        ),
    )

    # vggish 至少需要 vggish.pth；pca_params 有的话更好（某些实现会用）
    vggish_w = ckpt / "vggish.pth"
    must_exist(
        vggish_w,
        hint=(
            "你机器上一般已有 vggish 权重在 /root/.cache/torch/hub/checkpoints/。\n"
            "把它链接到 TORCH_HOME：\n"
            "  ln -sf /root/.cache/torch/hub/checkpoints/vggish.pth $TORCH_HOME/hub/checkpoints/vggish.pth\n"
        ),
    )

    results_root = Path(args.results_root).resolve() if Path(args.results_root).is_absolute() else (proj / args.results_root).resolve()
    real_dir = results_root / args.real_subdir
    must_exist(real_dir, hint=f"real_dir 不存在：{real_dir}（检查 --results_root/--real_subdir）")

    print(f"[INFO] project   = {proj}")
    print(f"[INFO] TORCH_HOME= {torch_home}")
    print(f"[INFO] hub_dir   = {hub_dir}")
    print(f"[INFO] ckpt_dir  = {ckpt}")
    print(f"[INFO] real_dir  = {real_dir} (wav_cnt={count_wavs(real_dir)})", flush=True)

    df = pd.read_csv(args.metrics_csv)
    if "mode" not in df.columns:
        raise ValueError("metrics_csv 里必须有一列叫 mode")

    # ---- 初始化两个 FAD 计算器（这一步如果权重没放对，就会尝试下载）----
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

    out_fad_vggish = []
    out_fad_pann = []

    for i, row in df.iterrows():
        mode = str(row["mode"])
        fake_dir = results_root / mode
        if not fake_dir.exists():
            print(f"[WARN] skip mode={mode}: fake_dir not found: {fake_dir}", flush=True)
            out_fad_vggish.append(float("nan"))
            out_fad_pann.append(float("nan"))
            continue

        n_fake = count_wavs(fake_dir)
        print(f"\n[MODE {i+1}/{len(df)}] {mode}")
        print(f"  fake_dir={fake_dir} (wav_cnt={n_fake})", flush=True)

        # 注意：FAD 是“整体分布距离”，wav 数不等也能算，但数太少会不稳定
        try:
            v = fad_vggish.score(str(real_dir), str(fake_dir), dtype=args.dtype)
        except Exception as e:
            print(f"  [ERR] FAD_vggish failed: {e}", flush=True)
            v = float("nan")

        try:
            p = fad_pann.score(str(real_dir), str(fake_dir), dtype=args.dtype)
        except Exception as e:
            print(f"  [ERR] FAD_pann failed: {e}", flush=True)
            p = float("nan")

        print(f"  FAD_vggish={v}")
        print(f"  FAD_pann  ={p}", flush=True)

        out_fad_vggish.append(v)
        out_fad_pann.append(p)

    df["FAD_vggish"] = out_fad_vggish
    df["FAD_pann"] = out_fad_pann

    out_path = Path(args.output_csv)
    df.to_csv(out_path, index=False)
    print(f"\n[DONE] wrote: {out_path.resolve()}", flush=True)

if __name__ == "__main__":
    main()
