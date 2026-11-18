from __future__ import annotations

import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from .audio import fit_length, lsd_db, snr_db


def _read_wav(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def evaluate_perturbation_pairs(raw_dir: str | Path, adv_dir: str | Path) -> dict:
    raw_dir, adv_dir = Path(raw_dir), Path(adv_dir)
    rows = []
    for raw_path in sorted(raw_dir.glob("*_raw.wav")):
        audio_id = raw_path.name[:-8]
        adv_path = adv_dir / f"{audio_id}_adv.wav"
        if not adv_path.exists():
            continue
        raw = _read_wav(raw_path)
        adv = _read_wav(adv_path)
        length = min(raw.shape[-1], adv.shape[-1])
        raw, adv = raw[..., :length], adv[..., :length]
        rows.append({"audio_id": audio_id, "snr_db": snr_db(raw, adv), "lsd_db": lsd_db(raw, adv),
                     "linf": float((adv - raw).abs().max())})
    if not rows:
        raise RuntimeError(f"No matched raw/adv pairs in {raw_dir} and {adv_dir}")
    return {
        "num_pairs": len(rows),
        "snr_db_mean": sum(r["snr_db"] for r in rows) / len(rows),
        "lsd_db_mean": sum(r["lsd_db"] for r in rows) / len(rows),
        "linf_max": max(r["linf"] for r in rows),
        "per_sample": rows,
    }


def evaluate_prompt_clap(meta_root: str | Path, backend) -> dict:
    rows = []
    for meta_path in sorted(Path(meta_root).rglob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "prompt" not in meta or "output_path" not in meta:
            continue
        output_path = Path(meta["output_path"])
        if not output_path.exists():
            continue
        audio = _read_wav(output_path).to(backend.device)
        score = backend.clap_similarity(audio, [meta["prompt"]])
        rows.append({
            "file": str(output_path),
            "prompt": meta["prompt"],
            "clap": float(score.detach().mean().cpu()),
            "label_id": meta.get("label_id"),
            "is_harmful": meta.get("is_harmful"),
        })
    if not rows:
        return {"num_samples": 0, "clap_mean": None, "benign_clap_mean": None, "harmful_clap_mean": None}
    benign = [r["clap"] for r in rows if r["is_harmful"] == 0]
    harmful = [r["clap"] for r in rows if r["is_harmful"] == 1]
    return {
        "num_samples": len(rows),
        "clap_mean": sum(r["clap"] for r in rows) / len(rows),
        "benign_clap_mean": sum(benign) / len(benign) if benign else None,
        "harmful_clap_mean": sum(harmful) / len(harmful) if harmful else None,
        "per_sample": rows,
    }


def evaluate_audio_distribution(
    generated_dir: str | Path,
    reference_dir: str | Path,
    device: str = "cuda",
    limit_num: int | None = None,
) -> dict:
    """Run the vendored AudioLDM evaluation protocol for FD/FAD, IS and KL.

    The evaluator downloads/loads its pretrained feature extractor. It is
    intentionally optional and fails with an actionable message if its
    additional dependencies or pretrained weights are unavailable.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tools_dir = repo_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from audioldm_eval import EvaluationHelper
    except Exception as exc:
        raise RuntimeError(
            "Could not import the vendored AudioLDM evaluator. Install its optional "
            "dependencies and retry; see repro_v2/README.md."
        ) from exc
    helper = EvaluationHelper(sampling_rate=16000, device=device, backbone="mert")
    return helper.main(str(generated_dir), str(reference_dir), limit_num=limit_num)
