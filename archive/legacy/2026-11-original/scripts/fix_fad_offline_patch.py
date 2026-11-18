import importlib.util
from pathlib import Path
import re
import shutil

def backup(p: Path):
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
        print("[BACKUP]", bak)

def insert_offline_block(txt: str) -> str:
    marker = "OFFLINE_PATCH_NO_LAION_CLAP"
    if marker in txt:
        return txt
    block = (
        "# OFFLINE_PATCH_NO_LAION_CLAP: avoid importing laion_clap (which triggers HF downloads)\n"
        "try:\n"
        "    import laion_clap  # type: ignore\n"
        "except Exception:\n"
        "    laion_clap = None  # offline-safe\n\n"
    )
    lines = txt.splitlines(True)
    # 插入点：文件开头 import 区域后
    insert_i = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s == "" or s.startswith("#") or s.startswith("import ") or s.startswith("from "):
            insert_i = i + 1
            continue
        break
    lines.insert(insert_i, block)
    return "".join(lines)

def patch_file(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # 1) 修复最常见的“try 后没缩进”
    txt = re.sub(r"(?m)^(\s*)try:\s*\n\1(import\s+laion_clap\s*)$",
                 r"\1try:\n\1    \2", txt)

    # 2) 删除裸 import（避免 import 时触发 HF）
    txt = re.sub(r"(?m)^\s*import\s+laion_clap\s*\n", "", txt)
    txt = re.sub(r"(?m)^\s*from\s+laion_clap[^\n]*\n", "", txt)

    # 3) 插入离线安全块（定义 laion_clap=None，不会 NameError）
    txt = insert_offline_block(txt)

    # 4) 最终语法校验，防止 IndentationError
    compile(txt, str(path), "exec")

    path.write_text(txt, encoding="utf-8")
    print("[PATCHED]", path)

def patch_init(init_py: Path):
    txt = init_py.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    out = []
    for line in txt:
        # 避免 __init__ 导入 clap_score 导致连带 import laion_clap
        if "clap_score" in line or "CLAPScore" in line:
            continue
        out.append(line)
    # 确保 FrechetAudioDistance 可用
    if not any("FrechetAudioDistance" in l for l in out):
        out.insert(0, "from .fad import FrechetAudioDistance\n")
    init_py.write_text("".join(out), encoding="utf-8")
    print("[PATCHED]", init_py)

def main():
    spec = importlib.util.find_spec("frechet_audio_distance")
    if spec is None or spec.origin is None:
        raise SystemExit("[ERROR] frechet_audio_distance not found in env")

    pkg_dir = Path(spec.origin).parent
    print("[INFO] frechet_audio_distance dir =", pkg_dir)

    fad_py = pkg_dir / "fad.py"
    init_py = pkg_dir / "__init__.py"
    clap_py = pkg_dir / "clap_score.py"

    for p in [fad_py, init_py, clap_py]:
        if p.exists():
            backup(p)

    if init_py.exists():
        patch_init(init_py)

    if fad_py.exists():
        patch_file(fad_py)

    if clap_py.exists():
        # clap_score 也补一刀，防止有人 import 它
        patch_file(clap_py)

    print("[DONE] offline patch complete")

if __name__ == "__main__":
    main()
