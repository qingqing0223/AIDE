from pathlib import Path
import re, shutil

SR = 16000  # 你的音频基本就是 16k（你前面也一直用 16000）

p = Path("run_guard_full_pipeline.py")
assert p.exists(), "找不到 run_guard_full_pipeline.py，请确认当前目录是 /autodl-tmp/speech_guard"

bak = p.with_suffix(".py.bak_clap_api")
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"[BAK] 备份已保存：{bak}")

txt = p.read_text(encoding="utf-8", errors="ignore")
new = txt

# 1) encode_text -> encode_texts（只替换 .encode_text( 这种调用）
new = re.sub(r"\.encode_text\(", ".encode_texts(", new)

# 2) .encode_audio(xxx) -> .encode_audio(xxx, sr=16000)（仅当括号里没有逗号）
def add_sr(m):
    inside = m.group(1).strip()
    if "," in inside:
        return m.group(0)
    return f".encode_audio({inside}, sr={SR})"

new = re.sub(r"\.encode_audio\(([^)\n]+)\)", add_sr, new)

if new == txt:
    print("[WARN] 没有改动：可能你已经修过/文件内容不同。")
else:
    p.write_text(new, encoding="utf-8")
    print("[OK] 已修复 run_guard_full_pipeline.py：encode_texts + encode_audio(sr=16000)")
