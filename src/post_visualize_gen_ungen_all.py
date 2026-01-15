import argparse
from pathlib import Path

# 直接复用我们已经调好的 7 图接口
from visualize_audio_views import save_all_views


def process_root(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root dir not found: {root}")

    case_dirs = sorted([p for p in root.glob("case_*") if p.is_dir()])
    print(f"[INFO] Found {len(case_dirs)} cases under {root}")

    for idx, case_dir in enumerate(case_dirs):
        print(f"[INFO] ({idx+1}/{len(case_dirs)}) case = {case_dir.name}")

        # 我们关注的 5 种 wav：x, δ, x+δ, gen, ungen
        wav_and_figs = [
            ("raw.wav", "fig_raw"),
            ("noise.wav", "fig_noise"),
            ("adv.wav", "fig_adv"),
            ("gen.wav", "fig_gen"),
            ("ungen.wav", "fig_ungen"),
        ]

        for wav_name, fig_subdir in wav_and_figs:
            wav_path = case_dir / wav_name
            if not wav_path.exists():
                print(f"  [WARN] {wav_path} not found, skip.")
                continue

            out_dir = case_dir / fig_subdir
            out_dir.mkdir(parents=True, exist_ok=True)

            prefix = wav_name.split(".")[0]  # raw / noise / adv / gen / ungen
            print(f"  [INFO] plotting 7 views for {wav_path} -> {out_dir}")

            # 这里会生成 7 张图：wave/mel/cqt/chroma/stft/mps/f0
            save_all_views(str(wav_path), str(out_dir), prefix)

    print("[INFO] All cases finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="out/adv_audiocaps_eps001_full/test/gen_ungen_all",
        help="Root dir of gen/ungen cases.",
    )
    args = parser.parse_args()
    process_root(args.root)


if __name__ == "__main__":
    main()
