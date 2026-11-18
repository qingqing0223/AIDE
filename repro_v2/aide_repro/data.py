from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from .audio import audio_id_for, list_audio_files, load_audio, sha256_file, write_jsonl


def build_audio_manifest(
    input_root: str | Path,
    output_jsonl: str | Path,
    duration_seconds: float = 10.0,
    sample_rate: int = 16000,
) -> list[dict]:
    records = []
    for path in list_audio_files(input_root):
        audio = load_audio(path, sample_rate=sample_rate, duration_seconds=duration_seconds)
        info = sf.info(str(path))
        records.append({
            "audio_id": audio_id_for(path),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "source_sample_rate": int(info.samplerate),
            "source_channels": int(info.channels),
            "source_frames": int(info.frames),
            "source_duration_seconds": float(info.duration),
            "model_sample_rate": int(sample_rate),
            "model_samples": int(audio.shape[-1]),
            "duration_seconds": float(duration_seconds),
        })
    write_jsonl(output_jsonl, records)
    return records


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _first_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _extract_json_texts(value) -> list[str]:
    found = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            found.append(text)
    elif isinstance(value, list):
        for item in value:
            found.extend(_extract_json_texts(item))
    elif isinstance(value, dict):
        preferred = ("prompt", "question", "input", "text", "instruction", "query")
        for key in preferred:
            if key in value and isinstance(value[key], str) and value[key].strip():
                found.append(value[key].strip())
                return found
        for item in value.values():
            found.extend(_extract_json_texts(item))
    return found


def build_prompt_dataset(
    audiocaps_csv: str | Path,
    aart_csv: str | Path,
    red_instruct_root: str | Path,
    taxonomy_json: str | Path,
    output_dir: str | Path,
    seed: int = 2025,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> dict:
    """Build one-benign/six-harmful prompt data with explicit source taxonomy maps."""
    import pandas as pd

    taxonomy = json.loads(Path(taxonomy_json).read_text(encoding="utf-8"))
    labels = taxonomy["labels"]
    if len(labels) != 7 or labels[0] != "benign":
        raise ValueError("Taxonomy must contain exactly benign + six harmful labels.")
    records = []

    caps = pd.read_csv(audiocaps_csv)
    caption_col = _first_column(caps.columns, ("caption", "Caption", "text", "sentence"))
    if caption_col is None:
        raise ValueError(f"AudioCaps caption column not found: {list(caps.columns)}")
    for index, row in caps.iterrows():
        text = str(row[caption_col]).strip()
        if text and text.lower() != "nan":
            records.append({
                "text": text, "label_id": 0, "label_name": labels[0],
                "is_harmful": 0, "source": "AudioCaps", "source_id": str(index)
            })

    aart_map = {str(k).strip().lower(): int(v) for k, v in taxonomy.get("aart_label_map", {}).items()}
    aart = pd.read_csv(aart_csv)
    text_col = _first_column(aart.columns, ("prompt", "Prompt", "question", "Question", "input", "text", "Text"))
    label_col = _first_column(aart.columns, ("crime", "Crime", "category", "Category", "label", "Label"))
    if text_col is None or label_col is None:
        raise ValueError(f"AART needs a prompt column and a category column; found {list(aart.columns)}")
    unknown_aart = set()
    for index, row in aart.iterrows():
        text = str(row[text_col]).strip()
        raw_label = str(row[label_col]).strip().lower()
        if not text or text.lower() == "nan":
            continue
        if raw_label not in aart_map:
            unknown_aart.add(raw_label)
            continue
        label_id = aart_map[raw_label]
        if not 1 <= label_id <= 6:
            raise ValueError(f"AART mapping for {raw_label!r} must be 1..6.")
        records.append({
            "text": text, "label_id": label_id, "label_name": labels[label_id],
            "is_harmful": 1, "source": "AART", "source_id": str(index),
            "source_label": raw_label
        })
    if unknown_aart:
        raise ValueError(
            "Unmapped AART categories. Add explicit entries to taxonomy.json: "
            + ", ".join(sorted(unknown_aart)[:30])
        )

    red_map = {str(k).strip().lower(): int(v) for k, v in taxonomy.get("red_file_map", {}).items()}
    red_root = Path(red_instruct_root)
    if not red_root.is_dir():
        raise FileNotFoundError(f"RED-INSTRUCT directory not found: {red_root}")
    unknown_red = []
    for path in sorted(red_root.rglob("*.json")):
        key = path.stem.lower()
        if key not in red_map:
            unknown_red.append(key)
            continue
        label_id = red_map[key]
        if not 1 <= label_id <= 6:
            raise ValueError(f"RED-INSTRUCT mapping for {key!r} must be 1..6.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for index, text in enumerate(_extract_json_texts(payload)):
            records.append({
                "text": text, "label_id": label_id, "label_name": labels[label_id],
                "is_harmful": 1, "source": "RED-INSTRUCT",
                "source_id": f"{path.name}:{index}", "source_label": key
            })
    if unknown_red:
        raise ValueError(
            "Unmapped RED-INSTRUCT JSON filenames. Add explicit entries to taxonomy.json: "
            + ", ".join(unknown_red[:30])
        )

    # Exact normalized duplicate removal; conflicting labels are a hard error.
    unique = {}
    for record in records:
        key = _norm_text(record["text"])
        if not key:
            continue
        if key in unique and unique[key]["label_id"] != record["label_id"]:
            raise ValueError(f"Conflicting labels for duplicate prompt: {record['text'][:100]}")
        unique.setdefault(key, record)
    records = list(unique.values())

    by_label = defaultdict(list)
    for record in records:
        group = hashlib.sha256(_norm_text(record["text"]).encode("utf-8")).hexdigest()
        by_label[record["label_id"]].append((group, record))

    rng = random.Random(seed)
    splits = {"train": [], "dev": [], "test": []}
    for label_id, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        if n < 3:
            splits["train"].extend(record for _, record in items)
            continue
        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio))
        if n_train + n_dev >= n:
            n_train = n - 2
            n_dev = 1
        splits["train"].extend(record for _, record in items[:n_train])
        splits["dev"].extend(record for _, record in items[n_train:n_train + n_dev])
        splits["test"].extend(record for _, record in items[n_train + n_dev:])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, values in splits.items():
        rng.shuffle(values)
        write_jsonl(out / f"psm_{split}.jsonl", values)
    mapping = {"label2id": {name: i for i, name in enumerate(labels)},
               "id2label": {str(i): name for i, name in enumerate(labels)}}
    (out / "label_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    stats = {
        split: {name: sum(1 for r in values if r["label_name"] == name) for name in labels}
        for split, values in splits.items()
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return {"counts": {k: len(v) for k, v in splits.items()}, "class_counts": stats}


def attach_audio_ids(
    prompt_jsonl: str | Path,
    audio_manifest_jsonl: str | Path,
    output_jsonl: str | Path,
    seed: int = 2025,
) -> int:
    """Deterministically link each evaluation prompt to one protected audio ID."""
    prompts = [json.loads(line) for line in Path(prompt_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    audios = [json.loads(line) for line in Path(audio_manifest_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    audio_ids = sorted(row["audio_id"] for row in audios)
    if not audio_ids:
        raise ValueError("Audio manifest is empty.")
    for row in prompts:
        if not row.get("audio_id"):
            digest = hashlib.sha256((_norm_text(row["text"]) + str(seed)).encode("utf-8")).hexdigest()
            row["audio_id"] = audio_ids[int(digest[:12], 16) % len(audio_ids)]
        row.setdefault("is_harmful", int(row.get("label_id", 0) > 0))
    write_jsonl(output_jsonl, prompts)
    return len(prompts)
