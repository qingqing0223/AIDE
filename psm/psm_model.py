import os
import json
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel

# 让 ROOT 自动等于 /autodl-tmp/speech_guard 或 /root/autodl-tmp/speech_guard 这种真实路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../speech_guard/psm
ROOT = os.path.dirname(THIS_DIR)                           # .../speech_guard


def _collect_roberta_roots():
    """
    收集所有可能放 roberta 的根目录，后面统一递归搜索：
      1) ROOT/huggingface_roberta_base
      2) /root/autodl-tmp/speech_guard/huggingface_roberta_base
      3) ROOT/roberta-base  （如果你将完整 roberta 解压到这里）
      4) /root/autodl-tmp/speech_guard/roberta-base
    """
    candidates = []

    cand1 = os.path.join(ROOT, "huggingface_roberta_base")
    cand2 = "/root/autodl-tmp/speech_guard/huggingface_roberta_base"
    cand3 = os.path.join(ROOT, "roberta-base")
    cand4 = "/root/autodl-tmp/speech_guard/roberta-base"

    for p in [cand1, cand2, cand3, cand4]:
        if os.path.isdir(p):
            candidates.append(p)

    # 再兜底：如果 ROOT 本身是 /root/autodl-tmp/speech_guard 或 /autodl-tmp/speech_guard，
    # 在 ROOT 下再找一层 roberta-base
    for name in ["roberta-base", "huggingface_roberta_base"]:
        p = os.path.join(ROOT, name)
        if os.path.isdir(p) and p not in candidates:
            candidates.append(p)

    return sorted(set(candidates))


def find_roberta_dir():
    """
    最鲁棒版：在所有候选根目录下递归寻找包含 config.json 的目录。
    一旦找到，就作为 transformers 的 from_pretrained 路径。
    """
    base_roots = _collect_roberta_roots()

    if not base_roots:
        raise FileNotFoundError(
            "没有在以下位置发现 roberta 权重根目录：\n"
            f"  1) {os.path.join(ROOT, 'huggingface_roberta_base')}\n"
            "  2) /root/autodl-tmp/speech_guard/huggingface_roberta_base\n"
            f"  3) {os.path.join(ROOT, 'roberta-base')}\n"
            "  4) /root/autodl-tmp/speech_guard/roberta-base\n"
            "请确认 roberta 权重已经解压到数据盘下。"
        )

    print("[PSM] 将在以下根目录中搜索 RoBERTa 模型：")
    for r in base_roots:
        print("   -", r)

    candidates = []
    for base in base_roots:
        for dirpath, dirnames, filenames in os.walk(base):
            if "config.json" in filenames:
                candidates.append(dirpath)

    if not candidates:
        msg = (
            "在上面的根目录中没有找到包含 config.json 的子目录。\n"
            "这通常意味着你目前只有 .git-lfs 的 blobs/refs 目录，模型文件没有真正下载下来。\n"
            "解决方案（任选一种）：\n"
            "  A) 在本地把 HuggingFace 的 roberta-base 完整下载好（包含 config.json、pytorch_model.bin 等），\n"
            "     压缩成 roberta-base.zip 上传到 /autodl-tmp/speech_guard/，然后解压到 roberta-base 目录。\n"
            "  B) 如果服务器能联网，可以在终端运行：\n"
            "       python -c \"from transformers import AutoModel; AutoModel.from_pretrained('roberta-base', cache_dir='/autodl-tmp/speech_guard/roberta-base')\"\n"
            "     下载完成后，本函数会自动在 /autodl-tmp/speech_guard/roberta-base 里找到模型。\n"
        )
        raise FileNotFoundError(msg)

    candidates = sorted(set(candidates))
    chosen = candidates[0]
    print("[PSM] 找到 RoBERTa 模型目录：", chosen)
    return chosen


# =================== 数据集与批处理 ===================

class PromptDataset(Dataset):
    """从 jsonl 读 {text, label_name, is_harmful}"""

    def __init__(self, jsonl_path: str, label2id: Dict[str, int]):
        self.samples = []
        self.label2id = label2id

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                label_name = obj["label_name"]
                is_harmful = int(obj.get("is_harmful", 1))
                if label_name not in label2id:
                    continue
                self.samples.append(
                    {
                        "text": text,
                        "label_id": label2id[label_name],
                        "is_harmful": is_harmful,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PSMCollator:
    """把一批样本打包成模型输入"""

    def __init__(self, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]):
        texts = [b["text"] for b in batch]
        label_ids = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
        is_harmful = torch.tensor([b["is_harmful"] for b in batch], dtype=torch.float32)

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_ids": label_ids,
            "is_harmful": is_harmful,
            "texts": texts,
        }


# =================== PSM 模型本体 ===================

class PSMModel(nn.Module):
    """
    Prompt Safety Module：
      - backbone: RoBERTa
      - classifier: 多类危险标签分类
      - gate: δ 门控 g∈[0,1]（ARR）
      - proj: 连续向量 p，用于“同语义拉近”的原型/对比损失
    """

    def __init__(self, num_labels: int, proj_dim: int = 256):
        super().__init__()
        roberta_dir = find_roberta_dir()
        self.tokenizer = AutoTokenizer.from_pretrained(
            roberta_dir, local_files_only=True
        )
        self.backbone = AutoModel.from_pretrained(
            roberta_dir, local_files_only=True
        )

        hidden = self.backbone.config.hidden_size  # 768

        self.classifier = nn.Linear(hidden, num_labels)

        self.gate_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # 输出 [0,1]
        )

        self.proj = nn.Linear(hidden, proj_dim)
        # 每个类别一个 prototype，用于拉近同类、拉远异类
        self.prototype = nn.Parameter(torch.randn(num_labels, proj_dim))

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0, :]  # [CLS] 向量

        logits = self.classifier(h_cls)          # [B, C]
        gate = self.gate_head(h_cls).squeeze(-1) # [B]
        p_vec = self.proj(h_cls)                 # [B, D]

        return {
            "logits": logits,
            "gate": gate,
            "p_vec": p_vec,
        }


def compute_psm_loss(model: PSMModel, batch, outputs,
                     lambda_proto: float = 0.5,
                     lambda_gate: float = 0.1,
                     temperature: float = 0.07):
    """
    联合损失：
      L = CE(分类) + λ_proto * L_proto + λ_gate * L_gate
    """

    label_ids = batch["label_ids"].to(outputs["logits"].device)
    is_harmful = batch["is_harmful"].to(outputs["logits"].device)

    logits = outputs["logits"]
    gate = outputs["gate"]
    p_vec = outputs["p_vec"]

    # 1. 分类损失
    ce_loss = F.cross_entropy(logits, label_ids)

    # 2. 原型对比损失
    p_norm = F.normalize(p_vec, dim=-1)
    proto_norm = F.normalize(model.prototype, dim=-1)
    proto_logits = torch.matmul(p_norm, proto_norm.T) / temperature  # [B, C]
    proto_loss = F.cross_entropy(proto_logits, label_ids)

    # 3. gate 回归损失
    gate_loss = F.mse_loss(gate, is_harmful)

    total = ce_loss + lambda_proto * proto_loss + lambda_gate * gate_loss
    info = {
        "ce": ce_loss.item(),
        "proto": proto_loss.item(),
        "gate": gate_loss.item(),
        "total": total.item(),
    }
    return total, info
