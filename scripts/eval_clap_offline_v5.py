# -*- coding: utf-8 -*-
"""
eval_clap_offline_v5.py

离线计算每个模式的：
- CLAP：平均音频-文本余弦相似度
- IS_clap：基于 CLAP 的二分类 Inception Score（近似定义）
- KL_clap：p(匹配) 相对均匀分布的 KL（衡量“偏向程度”）

注意：
- 这里是一个简化、但稳定的版本，公式和原作者实现可能略有差异，
  但用于比较 B0/B1/我们的方法是没有问题的。
- 完全离线，不会再去访问网络（前提是 model_dir 指向本地模型目录）。
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from clap_compat import ClapWrapper


# ========= 工具函数 =========

def load_prompt_map(prompt_file: Path) -> Tuple[Dict[str, str], List[str]]:
    """
    读取 prompts_test.txt，尽量兼容多种格式：
    - id<TAB>text
    - id||text
    - id text
    - 纯文本（没有 id，就用行号做 id）

    返回：
    - id2text: {id_str: text}
    - texts_in_order: 按行顺序的文本列表（兜底用）
    """
    id2text: Dict[str, str] = {}
    texts: List[str] = []

    with open(prompt_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            pid = None
            text = None

            if "\t" in line:
                pid, text = line.split("\t", 1)
            elif "||" in line:
                pid, text = line.split("||", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].replace(".", "").isdigit():
                    pid, text = parts
                else:
                    pid = str(idx)
                    text = line

            pid = pid.strip()
            text = text.strip()
            id2text[pid] = text
            texts.append(text)

    return id2text, texts


def get_id_from_filename(path: Path) -> str:
    """
    从 wav 文件名里抽取前缀数字作为 id，例如：
    - '102963_safe_gen.wav' -> '102963'
    - '102963_h072_harm.wav' -> '102963'
    """
    stem = path.stem
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if digits:
        return "".join(digits)
    return ""


def binary_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    对 [p, 1-p] 和 [q, 1-q] 的 KL(p||q)，逐样本计算。
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p_vec = np.stack([p, 1 - p], axis=1)          # (N,2)
    q_vec = np.stack([q, 1 - q], axis=1)          # (N,2) 或 (1,2) broadcast
    kl = np.sum(p_vec * (np.log(p_vec) - np.log(q_vec)), axis=1)  # (N,)
    return kl


# ========= 主逻辑 =========

def compute_clap_for_mode(
    mode: str,
    mode_dir: Path,
    id2text: Dict[str, str],
    fallback_texts: List[str],
    clap: ClapWrapper,
    temperature: float = 0.5,
) -> Dict[str, float]:
    """
    对单个 mode（例如 B0_AudioLDM）计算 CLAP / IS_clap / KL_clap。
    """
    wav_paths = sorted(mode_dir.glob("*.wav"))
    if not wav_paths:
        print(f"[WARN] 模式 {mode} 在 {mode_dir} 下没有找到 wav，跳过")
        return {
            "mode": mode,
            "N_audio": 0,
            "CLAP": float("nan"),
            "IS_clap": float("nan"),
            "KL_clap": float("nan"),
        }

    print(f"[INFO] mode {mode}: 共 {len(wav_paths)} 条 wav，开始提取 CLAP 特征...")

    # 1) 为每条 wav 找到对应文本
    texts_for_audio: List[str] = []
    for idx, wav in enumerate(wav_paths):
        pid = get_id_from_filename(wav)
        if pid and pid in id2text:
            text = id2text[pid]
        else:
            # 兜底：按顺序循环使用 prompt 列表
            text = fallback_texts[idx % len(fallback_texts)]
        texts_for_audio.append(text)

    # 2) 文本特征（对唯一文本先编码，再复用）
    unique_texts = sorted(set(texts_for_audio))
    text2emb: Dict[str, np.ndarray] = {}
    print(f"[INFO] mode {mode}: 唯一文本数量 {len(unique_texts)}，开始编码文本特征...")
    for t in tqdm(unique_texts, desc=f"{mode} text", ncols=100):
        emb = clap.encode_text(t, keepdims=False)
        text2emb[t] = emb.astype("float32")

    text_feats = np.stack(
        [text2emb[t] for t in texts_for_audio], axis=0
    )  # (N, D)

    # 3) 音频特征
    audio_feats_list: List[np.ndarray] = []
    print(f"[INFO] mode {mode}: 开始编码音频特征...")
    for wav in tqdm(wav_paths, desc=f"{mode} audio", ncols=100):
        emb = clap.encode_audio(str(wav), keepdims=False)
        audio_feats_list.append(emb.astype("float32"))

    audio_feats = np.stack(audio_feats_list, axis=0)  # (N, D)

    # 4) 计算一一对应的余弦相似度
    sims = clap.batched_cosine_similarity(audio_feats, text_feats)  # (N,)
    clap_score = float(np.mean(sims))

    # 5) 把相似度当成二分类 logit，构造近似的 IS / KL
    #    logits = cos / temperature  ->  p(correct | x) = sigmoid(logits)
    logits = sims / float(temperature)
    p = 1.0 / (1.0 + np.exp(-logits))  # (N,)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    # 数据集边缘分布 p_bar
    p_bar = float(np.mean(p))
    p_bar_arr = np.full_like(p, p_bar)

    # KL(p || p_bar)，再做 IS = exp(E[KL])
    kl_to_marginal = binary_kl(p, p_bar_arr)
    IS_clap = float(np.exp(np.mean(kl_to_marginal)))

    # KL(p || 0.5)，衡量“偏向均匀”的程度
    uniform = np.full_like(p, 0.5)
    kl_to_uniform = binary_kl(p, uniform)
    KL_clap = float(np.mean(kl_to_uniform))

    print(
        f"[RESULT] {mode}: N_audio={len(wav_paths)}, "
        f"CLAP={clap_score:.4f}, IS_clap={IS_clap:.4f}, KL_clap={KL_clap:.4f}"
    )

    return {
        "mode": mode,
        "N_audio": len(wav_paths),
        "CLAP": clap_score,
        "IS_clap": IS_clap,
        "KL_clap": KL_clap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--real_subdir", type=str, default="real_for_eval")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="把 CLAP 相似度当成 logit 时使用的温度系数（越小越尖锐）",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    prompt_path = Path(args.prompt_file)

    # 1) 读 prompt
    id2text, texts_in_order = load_prompt_map(prompt_path)
    print(f"[INFO] 读取 prompt 数量：{len(texts_in_order)}")

    # 2) 初始化 CLAP
    print("[INFO] 初始化 CLAP 模型...")
    clap = ClapWrapper(
        model_dir=args.model_dir,
        device=args.device,
    )

    # 3) 逐个 mode 计算
    all_rows: List[Dict[str, float]] = []
    for mode in args.modes:
        mode_dir = results_root / mode
        row = compute_clap_for_mode(
            mode=mode,
            mode_dir=mode_dir,
            id2text=id2text,
            fallback_texts=texts_in_order,
            clap=clap,
            temperature=args.temperature,
        )
        all_rows.append(row)

    # 4) 写 CSV
    out_csv = Path(args.output_csv)
    print(f"[INFO] 写入结果到 {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "N_audio", "CLAP", "IS_clap", "KL_clap"])
        for row in all_rows:
            writer.writerow(
                [
                    row["mode"],
                    row["N_audio"],
                    row["CLAP"],
                    row["IS_clap"],
                    row["KL_clap"],
                ]
            )


if __name__ == "__main__":
    main()
