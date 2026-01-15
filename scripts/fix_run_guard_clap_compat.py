import re
from pathlib import Path
from datetime import datetime

p = Path("run_guard_full_pipeline.py")
assert p.exists(), "run_guard_full_pipeline.py 不存在，请确认当前目录在 /autodl-tmp/speech_guard"

src = p.read_text(encoding="utf-8")

# 备份
bak = p.with_suffix(f".py.bak_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
bak.write_text(src, encoding="utf-8")
print(f"[BK] backup -> {bak}")

# 如果已经打过补丁就不重复插入
if "CLAP API COMPAT (auto patch)" not in src:
    # 找到 import 段落末尾插入兼容函数
    lines = src.splitlines(True)
    ins = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            ins = i + 1

    compat = r'''
# ===== CLAP API COMPAT (auto patch) =====
def _clap__to_numpy(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(x, "detach"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return x.cpu().numpy()
        except Exception:
            pass
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.asarray(x)

def _clap_encode_text_any(clap, texts):
    # texts: str | list[str]
    if isinstance(texts, str):
        texts = [texts]
    if hasattr(clap, "encode_texts"):
        return _clap__to_numpy(clap.encode_texts(texts))
    if hasattr(clap, "encode_text"):
        return _clap__to_numpy(clap.encode_text(texts))
    raise AttributeError("ClapWrapper 没有 encode_text/encode_texts")

def _clap_encode_audio_path(clap, wav_path, target_sr=16000):
    """
    兼容不同 ClapWrapper:
    - encode_audio(path)
    - encode_audio(path, sr)
    - encode_audio(path, sr=)
    - encode_audio(waveform, sr)
    """
    import numpy as np
    from pathlib import Path

    wav_path = str(wav_path)
    tried = []

    # 1) 先尝试“直接传路径”
    candidates = [
        ("path_only", lambda: clap.encode_audio(wav_path)),
        ("path_list", lambda: clap.encode_audio([wav_path])),
        ("path_pos_sr", lambda: clap.encode_audio(wav_path, target_sr)),
        ("path_kw_sr", lambda: clap.encode_audio(wav_path, sr=target_sr)),
        ("path_kw_sample_rate", lambda: clap.encode_audio(wav_path, sample_rate=target_sr)),
    ]
    for name, fn in candidates:
        try:
            out = fn()
            return _clap__to_numpy(out)
        except TypeError as e:
            tried.append((name, str(e)))
        except Exception as e:
            tried.append((name, f"{type(e).__name__}: {e}"))

    # 2) 如果都不行，加载音频波形再试
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"CLAP encode_audio 无法用路径调用，且 librosa.load 失败：{e}\ntried={tried}")

    wave_candidates = [
        ("wave_pos_sr", lambda: clap.encode_audio(y, target_sr)),
        ("wave_kw_sr", lambda: clap.encode_audio(y, sr=target_sr)),
        ("wave_kw_sample_rate", lambda: clap.encode_audio(y, sample_rate=target_sr)),
    ]
    for name, fn in wave_candidates:
        try:
            out = fn()
            return _clap__to_numpy(out)
        except Exception as e:
            tried.append((name, f"{type(e).__name__}: {e}"))

    raise RuntimeError("CLAP encode_audio 全部兼容尝试失败：\n" + "\n".join([f"- {n}: {m}" for n,m in tried]))
# ===== END CLAP API COMPAT =====
'''
    lines.insert(ins, compat + "\n")
    src = "".join(lines)
else:
    src = src

# 修复你之前 sed 造成的致命错误：str(wav_path, sr=16000) 这种
src2 = re.sub(r"str\(\s*(\w+)\s*,\s*sr\s*=\s*\d+\s*\)", r"str(\1)", src)

# 把 clap.encode_text/encode_texts 的调用统一改成兼容函数（只改最常见的那种 batch）
src2 = re.sub(r"\bclap\.encode_texts?\(\s*([^)]+?)\s*\)", r"_clap_encode_text_any(clap, \1)", src2)

# 把“取音频 embedding”的调用改成兼容函数（只要这一行里出现 wav_path/audio_path 就替换）
def _audio_repl(m):
    inside = m.group(1)
    var = "wav_path"
    if "audio_path" in inside:
        var = "audio_path"
    elif "path" in inside and "wav_path" not in inside:
        # 兜底：尽量不乱改
        var = "wav_path"
    return f"_clap_encode_audio_path(clap, {var}, target_sr=16000)"

src2 = re.sub(r"\bclap\.encode_audio\(\s*([^)]+)\)", lambda m: _audio_repl(m) if ("wav_path" in m.group(1) or "audio_path" in m.group(1)) else m.group(0), src2)

# 写回
p.write_text(src2, encoding="utf-8")

# 打印替换结果提示
print("[OK] patched run_guard_full_pipeline.py")
