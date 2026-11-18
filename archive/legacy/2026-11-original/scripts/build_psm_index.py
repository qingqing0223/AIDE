import argparse
import json
import sys
from pathlib import Path


def build_index(subset_root: Path, out_jsonl: Path):
    subset_root = subset_root.expanduser().resolve()
    if not subset_root.exists():
        print(f"[ERROR] 子集根目录不存在: {subset_root}", file=sys.stderr)
        return

    out_jsonl = out_jsonl.expanduser().resolve()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 子集根目录: {subset_root}")
    print(f"[INFO] 输出索引文件: {out_jsonl}")

    total = 0
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        # 按固定顺序尝试 train / validation / test
        for split in ["train", "validation", "test"]:
            split_dir = subset_root / split
            if not split_dir.exists():
                print(f"[WARN] 子目录 {split_dir} 不存在，跳过该 split")
                continue

            uid_dirs = sorted(
                p for p in split_dir.iterdir() if p.is_dir()
            )
            print(f"[INFO] 处理 {split}，共 {len(uid_dirs)} 个 uid 目录")

            for uid_dir in uid_dirs:
                uid = uid_dir.name

                raw = next(uid_dir.glob("*_raw.wav"), None)
                adv = next(uid_dir.glob("*_adv.wav"), None)

                if raw is None or adv is None:
                    print(
                        f"[WARN] {uid_dir} 中找不到 *_raw.wav 或 *_adv.wav，跳过该 uid",
                        file=sys.stderr,
                    )
                    continue

                # raw 样本（label = 0）
                rec_raw = {
                    "split": split,
                    "uid": uid,
                    "kind": "raw",
                    "label": 0,
                    "wav": str(raw.resolve()),
                }
                f_out.write(json.dumps(rec_raw, ensure_ascii=False) + "\n")
                total += 1

                # adv 样本（label = 1）
                rec_adv = {
                    "split": split,
                    "uid": uid,
                    "kind": "adv",
                    "label": 1,
                    "wav": str(adv.resolve()),
                }
                f_out.write(json.dumps(rec_adv, ensure_ascii=False) + "\n")
                total += 1

    print(f"[INFO] 共写入 {total} 条样本到 {out_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="为 PSM 生成 raw/adv 索引(jsonl)"
    )
    parser.add_argument(
        "--subset_root",
        type=str,
        required=True,
        help="某个子集根目录，例如 /root/autodl-tmp/speech_guard/out/audiocaps_eps001/n100",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        required=True,
        help="输出的 jsonl 路径，例如 out/psm_data/n100_index.jsonl",
    )
    args = parser.parse_args()

    build_index(Path(args.subset_root), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
