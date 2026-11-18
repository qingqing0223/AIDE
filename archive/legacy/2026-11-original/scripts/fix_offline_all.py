# fix_offline_all.py
from pathlib import Path
import re, site, shutil

def backup(p: Path):
    bk = p.with_suffix(p.suffix + ".bak")
    if not bk.exists():
        shutil.copy2(p, bk)

def patch_weights_only(text: str) -> str:
    # 给所有 torch.load(...) 补上 weights_only=False（如果没有的话）
    def repl(m):
        s = m.group(0)
        if "weights_only" in s:
            return s
        inner = m.group(1)
        return f"torch.load({inner}, weights_only=False)"
    return re.sub(r"torch\.load\(([^)]*?)\)", repl, text)

def patch_fad_py(p: Path):
    s = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    out = []
    changed = False
    for line in s:
        # 1) 强制 vggish 走本地 torch hub 缓存，不再访问 github
        if "torch.hub.load" in line and "torchvggish" in line:
            indent = line[: len(line) - len(line.lstrip())]
            new_line = indent + 'self.model = torch.hub.load(torch.hub.get_dir()+"/harritaylor_torchvggish_master", "vggish", source="local")\n'
            out.append(new_line)
            changed = True
            continue
        out.append(line)

    txt = "".join(out)
    txt2 = patch_weights_only(txt)
    if txt2 != txt:
        changed = True

    if changed:
        backup(p)
        p.write_text(txt2, encoding="utf-8")
        print("[OK] patched:", p)
    else:
        print("[SKIP] no change:", p)

def patch_repo_weights(repo: Path):
    if not repo.exists():
        print("[WARN] torchvggish repo not found:", repo)
        return
    n = 0
    for py in repo.rglob("*.py"):
        txt = py.read_text(encoding="utf-8", errors="ignore")
        txt2 = patch_weights_only(txt)
        if txt2 != txt:
            backup(py)
            py.write_text(txt2, encoding="utf-8")
            n += 1
    print(f"[OK] patched torchvggish repo files: {n}")

def main():
    # frechet_audio_distance 的 fad.py
    sp = Path(site.getsitepackages()[0])
    fad_py = sp / "frechet_audio_distance" / "fad.py"
    if not fad_py.exists():
        raise SystemExit(f"[ERR] not found: {fad_py}")
    patch_fad_py(fad_py)

    # torch hub 的本地 repo（依赖你提前 export TORCH_HOME 才会在对的位置）
    import torch
    hub_dir = Path(torch.hub.get_dir())
    repo = hub_dir / "harritaylor_torchvggish_master"
    patch_repo_weights(repo)

    print("[DONE] offline patch finished.")

if __name__ == "__main__":
    main()
