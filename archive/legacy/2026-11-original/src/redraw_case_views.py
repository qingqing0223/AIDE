from pathlib import Path
import argparse

from visualize_audio_views import save_all_views


def process_case(case_dir: Path):
    print(f"[INFO] Process case: {case_dir}")
    wavs = {
        "raw":   case_dir / "raw.wav",
        "adv":   case_dir / "adv.wav",
        "noise": case_dir / "noise.wav",
        "gen":   case_dir / "gen.wav",
        "ungen": case_dir / "ungen.wav",
    }

    for key, path in wavs.items():
        if not path.exists():
            print(f"[WARN] {path} 不存在，跳过 {key}")
            continue
        out_dir = case_dir / f"fig_{key}"
        save_all_views(path, out_dir, key)


def main():
    parser = argparse.ArgumentParser(
        description="从 case 目录中的 raw/adv/noise/gen/ungen.wav 生成 7 张可视化图"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="可以是单个 case 目录（包含 raw.wav 等），也可以是包含多个 case_* 子目录的根目录",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if (root / "raw.wav").exists():
        # root 本身就是一个 case 目录
        process_case(root)
    else:
        # root 底下有很多 case_* 子目录
        case_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("case_"))
        if not case_dirs:
            print(f"[ERROR] 在 {root} 下没有找到 case_* 目录")
            return
        for d in case_dirs:
            process_case(d)


if __name__ == "__main__":
    main()
