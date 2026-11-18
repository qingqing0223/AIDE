import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT = os.path.join(ROOT, "out")

print("ROOT =", ROOT)
print("OUT_ROOT =", OUT_ROOT)

if not os.path.isdir(OUT_ROOT):
    print("当前工程下没有 out 目录")
    raise SystemExit

for dirpath, dirnames, filenames in os.walk(OUT_ROOT):
    rel_dir = os.path.relpath(dirpath, ROOT)
    pts = [f for f in filenames if f.endswith((".pt", ".pth", ".npy"))]
    if not pts:
        continue

    print(f"\n目录: {rel_dir}")
    print(f"  文件数: {len(pts)}")
    # 只展示前 5 个文件名，避免刷屏
    for name in sorted(pts)[:5]:
        print("   -", name)
