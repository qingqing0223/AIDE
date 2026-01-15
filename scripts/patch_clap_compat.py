import re, time, sys
from pathlib import Path

p = Path("run_guard_full_pipeline.py")
if not p.exists():
    print("[ERR] not found:", p.resolve())
    sys.exit(1)

src = p.read_text(encoding="utf-8", errors="ignore")
bak = Path(f"run_guard_full_pipeline.py.bak_auto_{time.strftime('%Y%m%d_%H%M%S')}")
bak.write_text(src, encoding="utf-8")
print("[BK] backup ->", bak)

# 1) 删除旧的兼容块（如果之前插过）
src = re.sub(r"\n# ==== CLAP API COMPAT.*?# ==== END CLAP API COMPAT ====\n", "\n", src, flags=re.S)

# 2) 找插入点：跳过文件头 docstring（如果有），再在 import 段后插入
lines = src.splitlines(True)
i = 0
if i < len(lines) and lines[i].lstrip().startswith(("'''", '"""')):
    q = lines[i].lstrip()[:3]
    i += 1
    while i < len(lines) and q not in lines[i]:
        i += 1
    if i < len(lines):
        i += 1

# 向后走过连续的 import/from/comment/blank 行
while i < len(lines):
    s = lines[i].lstrip()
    if s.startswith(("import ", "from ", "#")) or s.strip() == "":
        i += 1
    else:
        break

compat = r'''
# ==== CLAP API COMPAT (auto patch) ====
def _clap_to_numpy(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.asarray([])
        if len(x) == 1:
            return _clap_to_numpy(x[0])
    return np.asarray(x)

def _clap_encode_text(clap, texts, *args, **kwargs):
    # 支持 encode_text / encode_texts / get_text_embedding 等不同实现
    if isinstance(texts, str):
        texts_in, single = [texts], True
    else:
        texts_in, single = list(texts), False

    fn = getattr(clap, "encode_texts", None) or getattr(clap, "encode_text", None)
    if fn is None:
        fn = getattr(clap, "get_text_embedding", None) or getattr(clap, "get_text_embeddings", None)
    if fn is None:
        raise AttributeError("CLAP text API not found on ClapWrapper")

    out = fn(texts_in, *args, **kwargs)
    arr = _clap_to_numpy(out).astype("float32")
    return arr[0] if single and arr.ndim >= 2 else arr

def _clap_encode_audio(clap, audio_or_path, *args, **kwargs):
    """
    统一兼容：
      - 旧：encode_audio(path)
      - 新：encode_audio(waveform, sr)
    这里会：
      1) 如果传进来是文件路径 -> librosa.load 成 waveform
      2) 再尝试调用 encode_audio / get_audio_embedding_from_data / filelist API
      3) 如果还是不行：返回全 0 向量（不中断主流程）
    """
    import os, numpy as np

    # 解析 sr/target_sr（既支持位置参数也支持关键字）
    target_sr = kwargs.pop("target_sr", None)
    sr = kwargs.pop("sr", None)
    if sr is None and len(args) > 0 and isinstance(args[0], (int, float)):
        sr = int(args[0]); args = args[1:]
    if target_sr is None:
        target_sr = sr
    if target_sr is None:
        for k in ("sample_rate", "sampling_rate", "sr"):
            v = getattr(clap, k, None)
            if isinstance(v, (int, float)):
                target_sr = int(v); break
    if target_sr is None:
        target_sr = 48000  # CLAP 常见采样率

    path = None
    if isinstance(audio_or_path, (str, os.PathLike)):
        path = str(audio_or_path)
    else:
        try:
            from pathlib import Path
            if isinstance(audio_or_path, Path):
                path = str(audio_or_path)
        except Exception:
            pass

    # 1) path -> waveform
    if path is not None and os.path.isfile(path):
        import librosa
        y, _ = librosa.load(path, sr=target_sr, mono=True)
        audio_np = y.astype(np.float32)[None, :]
    else:
        audio_np = audio_or_path
        try:
            import torch
            if isinstance(audio_np, torch.Tensor):
                audio_np = audio_np.detach().cpu().numpy()
        except Exception:
            pass
        audio_np = np.asarray(audio_np, dtype=np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]

    # 2) 先试 encode_audio / get_audio_embedding_from_data（numpy & torch 都试一次）
    candidates = []
    try:
        import torch
        candidates.append(torch.from_numpy(audio_np))
    except Exception:
        pass
    candidates.append(audio_np)

    fn = getattr(clap, "encode_audio", None) or getattr(clap, "get_audio_embedding_from_data", None)
    last = None
    if fn is not None:
        for a in candidates:
            for call in (
                lambda: fn(a, target_sr),
                lambda: fn(a, sr=target_sr),
                lambda: fn(a, sample_rate=target_sr),
                lambda: fn(a, sampling_rate=target_sr),
                lambda: fn(a),
            ):
                try:
                    out = call()
                    return _clap_to_numpy(out).astype("float32")
                except Exception as e:
                    last = e

    # 3) 再试 filelist API（很多 CLAP wrapper 有）
    if path is not None:
        fn2 = getattr(clap, "get_audio_embedding_from_filelist", None)
        if fn2 is not None:
            try:
                out = fn2([path])
                return _clap_to_numpy(out).astype("float32")
            except Exception as e:
                last = e

    # 4) 最终 fallback：返回 0 向量，不让主流程崩
    dim = 512
    try:
        # 有的 wrapper 暴露 embedding_dim
        d = getattr(clap, "embed_dim", None) or getattr(clap, "embedding_dim", None)
        if isinstance(d, int) and d > 0:
            dim = d
    except Exception:
        pass
    print(f"[WARN] CLAP audio encode failed -> fallback zeros. last_err={last}")
    return np.zeros((1, dim), dtype=np.float32)

# ==== END CLAP API COMPAT ====
'''

lines.insert(i, compat + "\n")
src = "".join(lines)

# 3) 修复你现在最致命的错误：str(wav_path, sr=...) / str(wav_path, target_sr=...)
src = re.sub(r"str\(\s*([A-Za-z0-9_\.]+)\s*,\s*(sr|target_sr)\s*=\s*\d+\s*\)", r"str(\1)", src)

# 4) 把调用统一替换到兼容封装（只替换 clap.* 形式）
src = re.sub(r"\bclap\.encode_texts?\(", "_clap_encode_text(clap, ", src)
src = re.sub(r"\bclap\.encode_audio\(", "_clap_encode_audio(clap, ", src)

p.write_text(src, encoding="utf-8")
print("[OK] patched:", p)

# 5) 语法自检：过不了就回滚
import py_compile
try:
    py_compile.compile(str(p), doraise=True)
    print("[OK] py_compile pass")
except Exception as e:
    print("[ERR] py_compile failed, rollback ->", bak, "\n", e)
    p.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
    sys.exit(1)
