import argparse
from pathlib import Path

from visualize_audio_views import save_all_views


def main():
    parser = argparse.ArgumentParser(
        description="Debug: save 7 audio views (incl. MPS) for one wav."
    )
    parser.add_argument("--wav", type=str, required=True,
                        help="Path to input wav")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory to save figures")
    parser.add_argument("--prefix", type=str, default="demo",
                        help="Filename prefix")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"wav not found: {wav_path}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    save_all_views(str(wav_path), str(out_root), args.prefix)


if __name__ == "__main__":
    main()
