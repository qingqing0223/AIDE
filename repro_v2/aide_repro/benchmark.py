from __future__ import annotations

import json
from pathlib import Path

from .audio import write_jsonl
from .inference import infer_one, load_psm_checkpoint


def run_prompt_suite(
    backend,
    checkpoint_path: str | Path,
    prompt_jsonl: str | Path,
    pgd_audio_dir: str | Path,
    output_dir: str | Path,
    device: str,
    duration_seconds: float = 10.0,
    ddim_steps: int = 50,
    t_start: int | None = None,
    guidance_scale: float = 2.5,
    seed: int = 2025,
    save_paired: bool = True,
    local_files_only: bool = False,
    mode: str = "aide",
    hard_gate: bool = False,
    disable_channel_gate: bool = False,
) -> dict:
    psm, checkpoint = load_psm_checkpoint(checkpoint_path, backend.device, local_files_only)
    prompts = [json.loads(line) for line in Path(prompt_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    pgd_audio_dir = Path(pgd_audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(prompts):
        audio_id = row.get("audio_id")
        if not audio_id:
            raise ValueError("Evaluation prompt rows must include audio_id linking to a Stage-I sample.")
        raw_path = pgd_audio_dir / f"{audio_id}_raw.wav"
        delta_path = pgd_audio_dir / f"{audio_id}_delta.wav"
        if not raw_path.exists() or not delta_path.exists():
            raise FileNotFoundError(f"Missing Stage-I audio pair for {audio_id}")
        label_id = int(row.get("label_id", -1))
        harmful = int(row.get("is_harmful", label_id > 0))
        group = "harmful" if harmful else "benign"
        out_path = output_dir / group / f"{audio_id}_{index:06d}.wav"
        result = infer_one(
            backend, psm, raw_path, delta_path, str(row["text"]), out_path,
            duration_seconds=duration_seconds, ddim_steps=ddim_steps, t_start=t_start,
            guidance_scale=guidance_scale, seed=seed + index, mode=mode,
            hard_gate=hard_gate, disable_channel_gate=disable_channel_gate,
            save_paired=save_paired, local_files_only=local_files_only,
            psm_checkpoint_path=checkpoint_path,
        )
        result["audio_id"] = audio_id
        result["label_id"] = label_id
        result["is_harmful"] = harmful
        result["output_path"] = str(out_path.resolve())
        out_path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
    write_jsonl(output_dir / "suite_results.jsonl", results)
    return {"num_prompts": len(results), "output_dir": str(output_dir.resolve())}
