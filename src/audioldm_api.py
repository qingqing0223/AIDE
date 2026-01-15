from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

# 用官方 AudioLDM 包里的接口
from audioldm import text_to_audio


TARGET_SR = 16000


def call_audioldm(
    prompt: str,
    duration: float,
    model_name: str = "audioldm-s-full-v2",
) -> Tuple[np.ndarray, int]:
    """
    统一对外的 AudioLDM 封装。

    参数：
      - prompt: 文本描述
      - duration: 生成时长（秒）
      - model_name: 模型名字（这里只是为了兼容调用方，实际可以不用）

    返回：
      - audio: 1D numpy.float32 波形，范围 [-1, 1]
      - sr: 采样率
    """
    import inspect

    kwargs = {}

    # 自动适配不同版本 text_to_audio 的参数名，避免 unexpected keyword argument
    sig = inspect.signature(text_to_audio)
    params = sig.parameters

    # 大多数版本都有 duration
    if "duration" in params:
        kwargs["duration"] = float(duration)

    # 有些版本可以直接指定采样率
    if "sr" in params:
        kwargs["sr"] = TARGET_SR

    # 调用 AudioLDM 生成音频
    out = text_to_audio(prompt, **kwargs)

    # 有的版本返回 audio，有的返回 (audio, sr)
    if isinstance(out, tuple):
        audio, sr = out
    else:
        audio = out
        sr = TARGET_SR

    # 转成 1D float32，范围 [-1, 1]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    max_abs = float(np.max(np.abs(audio))) if audio.size > 0 else 0.0
    if max_abs > 0:
        audio = audio / max_abs
    audio = np.clip(audio, -1.0, 1.0)

    return audio, int(sr)


def save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """
    简单封装一下保存 wav，方便后面需要的话调用。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
