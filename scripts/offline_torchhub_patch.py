import os, shutil
from pathlib import Path
from urllib.parse import urlparse, unquote

def _candidates(fname: str):
    proj = Path(os.environ.get("PROJ", "/autodl-tmp/speech_guard"))
    torch_home = Path(os.environ.get("TORCH_HOME", str(proj / "torch_cache")))
    return [
        proj / fname,
        proj / "torch_cache" / "hub" / "checkpoints" / fname,
        torch_home / "hub" / "checkpoints" / fname,
        Path("/root/.cache/torch/hub/checkpoints") / fname,
    ]

def _offline_download_url_to_file(url, dst, hash_prefix=None, progress=True):
    fname = Path(urlparse(url).path).name
    fname = unquote(fname)  # 处理 %3D 这种编码
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    for c in _candidates(fname):
        if c.exists():
            shutil.copy2(c, dst)
            return

    raise RuntimeError(
        f"[OFFLINE] torch.hub wants to download: {url}\n"
        f"but local file '{fname}' not found.\n"
        "Please place it into one of:\n" +
        "\n".join(str(x.parent) for x in _candidates(fname))
    )

def patch_torchhub():
    import torch.hub as hub
    hub.download_url_to_file = _offline_download_url_to_file
    if hasattr(hub, "_download_url_to_file"):
        hub._download_url_to_file = _offline_download_url_to_file
