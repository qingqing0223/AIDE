#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_gate_json.py

将 PSM/分类器输出（CSV/TSV/JSON/JSONL）转换为 pid -> gate 的 JSON。

支持输入：
- .csv / .tsv
- .json (dict 或 list)
- .jsonl (每行一个 dict)

自动推断字段：
- pid 字段: pid / id / prompt_id
- gate 字段优先级: gate / s / scalar_gate / p_harm / harm_prob / unsafe_prob / score

你也可以手动指定：
--pid_key, --gate_key

可选：
--prompts_file prompts_test.txt  用它过滤 pid（只保留出现过的 pid）
--clip  把 gate 裁剪到 [0,1]
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PID_KEYS = ["pid", "id", "prompt_id", "promptid", "key"]
GATE_KEYS_PRIORITY = [
    "gate", "s", "scalar_gate", "s_gate",
    "p_harm", "harm_prob", "unsafe_prob", "toxicity",
    "score", "risk"
]


def read_prompts_whitelist(prompts_file: Path) -> Optional[set[str]]:
    """
    prompts_test.txt 格式：pid \\t category \\t safety \\t text
    返回 pid 集合
    """
    if not prompts_file:
        return None
    if not prompts_file.exists():
        raise FileNotFoundError(f"prompts_file 不存在: {prompts_file}")

    keep: set[str] = set()
    with prompts_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 3)
            if len(parts) != 4:
                continue
            pid = parts[0].strip()
            if pid:
                keep.add(pid)
    return keep


def try_get(d: Dict[str, Any], keys: List[str]) -> Tuple[Optional[str], Optional[Any]]:
    for k in keys:
        if k in d:
            return k, d[k]
    # 尝试大小写不敏感匹配
    lower_map = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        kk = k.lower()
        if kk in lower_map:
            real_k = lower_map[kk]
            return str(real_k), d[real_k]
    return None, None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(x)
        return float(x)
    except Exception:
        return None


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def load_rows(path: Path) -> List[Dict[str, Any]]:
    """
    统一转成 list[dict]
    """
    suf = path.suffix.lower()

    if suf in [".csv", ".tsv"]:
        delim = "\t" if suf == ".tsv" else ","
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            return [dict(r) for r in reader]

    if suf == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            # 可能是 {"s039": 0.12, ...} 这种，直接转成 rows
            if all(isinstance(k, str) for k in obj.keys()) and all(
                isinstance(v, (int, float)) for v in obj.values()
            ):
                return [{"pid": k, "gate": v} for k, v in obj.items()]
            # 也可能是 {"rows":[...]}
            if "rows" in obj and isinstance(obj["rows"], list):
                return [r for r in obj["rows"] if isinstance(r, dict)]
            # 否则当成一条记录
            return [obj]
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        return []

    raise ValueError(f"不支持的文件类型: {path}")


def infer_pid_gate_keys(rows: List[Dict[str, Any]],
                        pid_key: str = "",
                        gate_key: str = "") -> Tuple[str, str]:
    if not rows:
        raise ValueError("输入数据为空，无法推断字段")

    # pid_key
    if pid_key:
        pk = pid_key
    else:
        pk, _ = try_get(rows[0], PID_KEYS)
        if not pk:
            # 再在所有行里找
            for r in rows[:50]:
                pk, _ = try_get(r, PID_KEYS)
                if pk:
                    break
    if not pk:
        raise ValueError(
            f"无法推断 pid 字段。请用 --pid_key 指定。候选字段: {PID_KEYS}"
        )

    # gate_key
    if gate_key:
        gk = gate_key
    else:
        gk, _ = try_get(rows[0], GATE_KEYS_PRIORITY)
        if not gk:
            for r in rows[:50]:
                gk, _ = try_get(r, GATE_KEYS_PRIORITY)
                if gk:
                    break
    if not gk:
        raise ValueError(
            f"无法推断 gate 字段。请用 --gate_key 指定。候选字段: {GATE_KEYS_PRIORITY}"
        )

    return pk, gk


def build_gate_map(rows: List[Dict[str, Any]],
                   pid_key: str,
                   gate_key: str,
                   whitelist: Optional[set[str]],
                   clip: bool) -> Dict[str, float]:
    out: Dict[str, float] = {}
    bad = 0
    kept = 0

    for r in rows:
        pid = str(r.get(pid_key, "")).strip()
        if not pid:
            bad += 1
            continue
        if whitelist is not None and pid not in whitelist:
            continue

        v = r.get(gate_key, None)
        fv = to_float(v)
        if fv is None:
            bad += 1
            continue

        if clip:
            fv = clamp01(fv)

        out[pid] = float(fv)
        kept += 1

    print(f"[INFO] 解析完成：kept={kept}, bad={bad}, total_rows={len(rows)}")
    if out:
        vals = list(out.values())
        print(f"[INFO] gate 统计：min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="输入文件（csv/tsv/json/jsonl）")
    ap.add_argument("--output", type=str, required=True, help="输出 gate json 路径")
    ap.add_argument("--pid_key", type=str, default="", help="手动指定 pid 字段名")
    ap.add_argument("--gate_key", type=str, default="", help="手动指定 gate 字段名")
    ap.add_argument("--prompts_file", type=str, default="", help="prompts_test.txt，用于过滤 pid")
    ap.add_argument("--clip", action="store_true", help="把 gate 裁剪到 [0,1]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"[ERROR] 输入不存在: {in_path}")

    whitelist = None
    if args.prompts_file:
        whitelist = read_prompts_whitelist(Path(args.prompts_file))
        print(f"[INFO] whitelist pid 数量: {len(whitelist)}")

    rows = load_rows(in_path)
    pid_key, gate_key = infer_pid_gate_keys(rows, pid_key=args.pid_key, gate_key=args.gate_key)
    print(f"[INFO] 使用字段：pid_key={pid_key}, gate_key={gate_key}")

    gate_map = build_gate_map(rows, pid_key, gate_key, whitelist=whitelist, clip=args.clip)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gate_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已写入: {out_path} (N={len(gate_map)})")


if __name__ == "__main__":
    main()
