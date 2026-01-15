import os

# 工程根目录：locate_pgd_delta.py 放在 pgd 下面，所以往上一层
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print("ROOT =", ROOT)

# 记录：root_dir -> set(splits)
root_to_splits = {}

# 只扫最多 8 层目录，避免太慢
for dirpath, dirnames, filenames in os.walk(ROOT):
    rel_dir = os.path.relpath(dirpath, ROOT)

    # 忽略太深的路径
    if rel_dir.count(os.sep) > 7:
        continue

    for name in filenames:
        # 只看可能是 PGD 结果的文件：名字里带 delta/eps，后缀是 pt/pth/npy
        if not (name.endswith((".pt", ".pth", ".npy")) and ("delta" in name or "eps" in name)):
            continue

        parts = rel_dir.split(os.sep)
        if len(parts) < 2:
            # 说明下面没有 split 这一层，先不处理
            continue

        split = parts[-1]          # 最后一级目录名，比如 "train"/"validation"/"test"
        root_dir = os.path.join(*parts[:-1])  # 倒数第二级之前，比如 "out/audiocaps_eps001"

        if split not in {"train", "dev", "validation", "val", "test"}:
            # 不是我们关心的 split 名，就跳过
            continue

        root_to_splits.setdefault(root_dir, set()).add(split)
        break  # 这个目录已经确认有一个 pgd 文件了，没必要继续看更多文件

if not root_to_splits:
    print("没有在工程目录下找到任何 *delta*.pt / *delta*.pth / *delta*.npy 文件，")
    print("请确认你的 PGD 训练代码是否真的把 δ 保存到了磁盘。")
else:
    print("\n可能的 PGD 结果目录如下：")
    for root_dir, splits in sorted(root_to_splits.items()):
        print(f"  root: {root_dir}")
        print(f"    splits: {', '.join(sorted(splits))}")
