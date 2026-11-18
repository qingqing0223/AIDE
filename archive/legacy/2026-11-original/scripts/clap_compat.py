# -*- coding: utf-8 -*-
"""
CLAP API 兼容层：
- 兼容 encode_text / encode_texts
- 兼容 encode_audio 需要 sr（且 sr 可能不接受 keyword）
- 输入 wav_path，内部负责读音频 + 重采样到 target_sr
"""
from __future__ import annotations

import numpy as np

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch Tensor
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    # tuple/list: 取第一个像 embedding 的
    if isinstance(x, (list, tuple)) and len(x) > 0:
        for item in x:
            arr = _to_numpy(item)
            if isinstance(arr, np.ndarray):
                return arr
        return _to_numpy(x[0])
    # dict: 常见 key
    if isinstance(x, dict):
        for k in ["embedding", "embeddings", "audio_embedding", "text_embedding", "feat", "features"]:
            if k in x:
                arr = _to_numpy(x[k])
                if isinstance(arr, np.ndarray):
                    return arr
        # fallback: 取第一个 value
        for v in x.values():
            arr = _to_numpy(v)
            if isinstance(arr, np.ndarray):
                return arr
    return np.asarray(x)

def clap_encode_texts(clap, texts):
    # texts: list[str] or str
    if isinstance(texts, str):
        texts = [texts]
    # 优先 encode_texts，其次 encode_text
    if hasattr(clap, "encode_texts"):
        fn = clap.encode_texts
    elif hasattr(clap, "encode_text"):
        fn = clap.encode_text
    else:
        raise AttributeError("CLAP wrapper has neither encode_texts nor encode_text")

    out = fn(texts)
    arr = _to_numpy(out)
    if arr is None:
        raise RuntimeError("encode_text(s) returned None")
    return arr.astype(np.float32)

def _load_audio(path: str, target_sr: int = 16000):
    # 尽量用 soundfile；没有就用 librosa
    y = None
    sr = None
    try:
        import soundfile as sf
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            # 很极端情况下才会到这：用简单插值重采样兜底
            ratio = float(target_sr) / float(sr)
            n = int(len(y) * ratio)
            xp = np.linspace(0, 1, num=len(y), dtype=np.float32)
            xq = np.linspace(0, 1, num=n, dtype=np.float32)
            y = np.interp(xq, xp, y).astype(np.float32)
            sr = target_sr

    y = y.astype(np.float32)
    return y, sr

def clap_encode_audio_path(clap, wav_path, target_sr: int = 16000):
    wav_path = str(wav_path)

    # 先尝试：有的实现支持直接传 path
    if hasattr(clap, "encode_audio"):
        fn = clap.encode_audio
    else:
        raise AttributeError("CLAP wrapper has no encode_audio")

    # 1) 若 encode_audio(path) 直接可用，最快
    try:
        out = fn(wav_path)
        arr = _to_numpy(out)
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float32)
    except Exception:
        pass

    # 2) 读音频，传 waveform (+ sr)
    y, sr = _load_audio(wav_path, target_sr=target_sr)

    # 多种形状都试一下（不同 wrapper 要求不一样）
    candidates = [y, y[None, :]]

    last_err = None
    for audio in candidates:
        # 2.1 keyword 方式（如果支持）
        for kw in [{"sr": sr}, {"sample_rate": sr}, {"sampling_rate": sr}, {"fs": sr}]:
            try:
                out = fn(audio, **kw)
                arr = _to_numpy(out)
                if isinstance(arr, np.ndarray):
                    return arr.astype(np.float32)
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e

        # 2.2 positional 方式（最常见：encode_audio(audio, sr)）
        try:
            out = fn(audio, sr)
            arr = _to_numpy(out)
            if isinstance(arr, np.ndarray):
                return arr.astype(np.float32)
        except Exception as e:
            last_err = e

        # 2.3 最后兜底：encode_audio(audio)
        try:
            out = fn(audio)
            arr = _to_numpy(out)
            if isinstance(arr, np.ndarray):
                return arr.astype(np.float32)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"encode_audio failed for {wav_path}. last_err={last_err}")
