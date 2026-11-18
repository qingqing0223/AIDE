import re
from pathlib import Path
import frechet_audio_distance as fad

pkg = Path(fad.__file__).resolve().parent
print("[INFO] frechet_audio_distance =", pkg)

# 1) __init__.py：只暴露 FrechetAudioDistance，避免 import clap_score 导致 laion_clap 触发
init_py = pkg / "__init__.py"
init_py.write_text(
    "from .fad import FrechetAudioDistance\n"
    "__all__ = ['FrechetAudioDistance']\n",
    encoding="utf-8"
)
print("[PATCH] rewrite", init_py)

# 2) fad.py：移除/屏蔽顶部 laion_clap 导入（避免 bert-base-uncased）
fad_py = pkg / "fad.py"
txt = fad_py.read_text(encoding="utf-8", errors="ignore")

# 干掉任何顶层 laion_clap import（非常关键）
txt2 = re.sub(r'^\s*(import|from)\s+laion_clap[^\n]*\n', '', txt, flags=re.M)

# 有些版本会从 clap_score 导入 ClapScore（同样会触发 laion_clap），也删掉
txt2 = re.sub(r'^\s*from\s+\.clap_score[^\n]*\n', '', txt2, flags=re.M)

# 加一个保护注释，方便你以后 grep
if "OFFLINE_PATCH_NO_LAION_CLAP" not in txt2:
    txt2 = "# OFFLINE_PATCH_NO_LAION_CLAP: do NOT import laion_clap at module import time\n" + txt2

if txt2 != txt:
    fad_py.write_text(txt2, encoding="utf-8")
    print("[PATCH] update", fad_py)
else:
    print("[SKIP] fad.py already patched")

print("[DONE] patch complete.")
