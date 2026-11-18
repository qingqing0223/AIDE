import os
import json
import random
import glob
from collections import defaultdict

import pandas as pd

# 用脚本所在目录自动推 ROOT，避免 /autodl-tmp 和 /root/autodl-tmp 不一致
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)  # /autodl-tmp/speech_guard
OUT_DIR = os.path.join(ROOT, "psm", "data")

os.makedirs(OUT_DIR, exist_ok=True)


def find_aart_csv():
    """
    尽量鲁棒地找到 AART 的 csv：
    1）在 ROOT/prompt/AART 下找 *.csv
    2）在 /root/autodl-tmp/speech_guard/prompt/AART 下找
    3）在 /autodl-tmp 和 /root/autodl-tmp 下递归搜索文件名包含 aart 的 csv
    """
    candidates = []

    base1 = os.path.join(ROOT, "prompt", "AART")
    pattern1 = os.path.join(base1, "*.csv")
    candidates.extend(glob.glob(pattern1))

    base2 = "/root/autodl-tmp/speech_guard/prompt/AART"
    pattern2 = os.path.join(base2, "*.csv")
    candidates.extend(glob.glob(pattern2))

    if not candidates:
        for base in ["/autodl-tmp", "/root/autodl-tmp"]:
            if not os.path.exists(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                for f in filenames:
                    if f.endswith(".csv") and "aart" in f.lower():
                        candidates.append(os.path.join(dirpath, f))

    if not candidates:
        raise FileNotFoundError(
            "没有找到 AART 的 csv 文件，请确认存放位置。\n"
            f"尝试过：{pattern1} 和 {pattern2}，以及 /autodl-tmp、/root/autodl-tmp 的递归搜索。"
        )

    candidates = sorted(set(candidates))
    print("找到以下 AART csv 候选：")
    for c in candidates:
        print("  ", c)

    chosen = candidates[0]
    print("将使用 AART 文件：", chosen)
    return chosen


def find_red_harmful_dir():
    """
    尽量鲁棒地找到 red-instruct 的 harmful_questions 目录：
    1）ROOT/prompt/red-instruct-main/harmful_questions
    2）/root/autodl-tmp/speech_guard/prompt/red-instruct-main/harmful_questions
    3）在 ROOT/prompt、/autodl-tmp、/root/autodl-tmp 下递归搜包含 adversarialqa.json 的 harmful_questions 目录
    """
    candidates = []

    c1 = os.path.join(ROOT, "prompt", "red-instruct-main", "harmful_questions")
    if os.path.isdir(c1):
        candidates.append(c1)

    c2 = "/root/autodl-tmp/speech_guard/prompt/red-instruct-main/harmful_questions"
    if os.path.isdir(c2):
        candidates.append(c2)

    if not candidates:
        # 在 ROOT/prompt 下找
        prompt_root = os.path.join(ROOT, "prompt")
        for base in [prompt_root, "/autodl-tmp", "/root/autodl-tmp"]:
            if not os.path.exists(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                if os.path.basename(dirpath) == "harmful_questions":
                    # 看看里面有没有 adversarialqa.json 之类的文件
                    has_key = any(
                        fn in filenames
                        for fn in [
                            "adversarialqa.json",
                            "dangerousqa.json",
                            "harmfulqa.json",
                        ]
                    )
                    if has_key:
                        candidates.append(dirpath)

    if not candidates:
        raise FileNotFoundError(
            "没有找到 red-instruct 的 harmful_questions 目录。\n"
            "请在 Jupyter 左侧确认它的完整路径，然后告诉我。"
        )

    candidates = sorted(set(candidates))
    print("找到以下 harmful_questions 目录候选：")
    for c in candidates:
        print("  ", c)

    chosen = candidates[0]
    print("将使用 harmful_questions 目录：", chosen)
    return chosen


def load_aart(path):
    print("正在读取 AART:", path)
    df = pd.read_csv(path)

    text_cols = ["prompt", "Prompt", "question", "Question", "input", "text", "Text"]
    label_cols = ["crime", "Crime", "category", "Category", "label", "Label"]

    text_col = None
    for c in text_cols:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"AART 中没有找到文本列，当前列名：{list(df.columns)}")

    label_col = None
    for c in label_cols:
        if c in df.columns:
            label_col = c
            break

    records = []
    if label_col is None:
        print("警告：AART 没有显式类别列，全部标成 aart_harmful")
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text:
                continue
            records.append(
                {"text": text, "label_name": "aart_harmful", "is_harmful": 1}
            )
    else:
        for _, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text:
                continue
            raw_label = str(row[label_col]).strip().lower().replace(" ", "_")
            if raw_label == "":
                raw_label = "harmful"
            label_name = f"aart_{raw_label}"
            records.append(
                {"text": text, "label_name": label_name, "is_harmful": 1}
            )
    return records


def load_red_instruct(red_root):
    records = []
    print("正在从 red-instruct 目录读取 JSON：", red_root)
    for fname in os.listdir(red_root):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(red_root, fname)
        label_name = os.path.splitext(fname)[0]

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    text = ""
                    for k in ["prompt", "Prompt", "question", "Question", "input", "text"]:
                        if k in item and isinstance(item[k], str):
                            text = item[k].strip()
                            break
                else:
                    continue

                if not text:
                    continue
                records.append(
                    {"text": text, "label_name": label_name, "is_harmful": 1}
                )
        else:
            print(f"警告：{fpath} JSON 结构特殊，请手动检查。")
    return records


def stratified_split(records, train_ratio=0.8, dev_ratio=0.1, seed=42):
    random.seed(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label_name"]].append(r)

    train, dev, test = [], [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        if n == 1:
            train.extend(items)
            continue
        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio))
        n_test = max(1, n - n_train - n_dev)
        train.extend(items[:n_train])
        dev.extend(items[n_train : n_train + n_dev])
        test.extend(items[n_train + n_dev : n_train + n_dev + n_test])
    return train, dev, test


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    aart_path = find_aart_csv()
    red_root = find_red_harmful_dir()

    print("==> 读取 AART 数据 ...")
    aart_records = load_aart(aart_path)
    print("AART 条数:", len(aart_records))

    print("==> 读取 red-instruct harmful_questions ...")
    red_records = load_red_instruct(red_root)
    print("red-instruct 条数:", len(red_records))

    all_records = aart_records + red_records
    print("合并总样本数:", len(all_records))

    train, dev, test = stratified_split(all_records)

    save_jsonl(os.path.join(OUT_DIR, "psm_train.jsonl"), train)
    save_jsonl(os.path.join(OUT_DIR, "psm_dev.jsonl"), dev)
    save_jsonl(os.path.join(OUT_DIR, "psm_test.jsonl"), test)

    labels = sorted({r["label_name"] for r in all_records})
    label_list_path = os.path.join(OUT_DIR, "label_list.txt")
    with open(label_list_path, "w", encoding="utf-8") as f:
        for lb in labels:
            f.write(lb + "\n")

    print("完成！输出文件：")
    print(" ", os.path.join(OUT_DIR, "psm_train.jsonl"))
    print(" ", os.path.join(OUT_DIR, "psm_dev.jsonl"))
    print(" ", os.path.join(OUT_DIR, "psm_test.jsonl"))
    print(" ", label_list_path)


if __name__ == "__main__":
    main()
