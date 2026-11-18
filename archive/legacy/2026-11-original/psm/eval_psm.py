import os
import json

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from psm_model import ROOT, PromptDataset, PSMCollator, PSMModel


def load_label2id():
    path = os.path.join(ROOT, "psm", "data", "label2id.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    # JSON 里 key 是字符串，转回 int
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def main():
    test_path = os.path.join(ROOT, "psm", "data", "psm_test.jsonl")
    label2id, id2label = load_label2id()

    test_ds = PromptDataset(test_path, label2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSMModel(num_labels=len(label2id), proj_dim=256)
    ckpt = os.path.join(ROOT, "psm", "checkpoints", "psm_roberta_best.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    collator = PSMCollator(model.tokenizer, max_len=128)
    loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label_ids"].to(device)

            outs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outs["logits"], dim=-1)

            all_labels.extend(label_ids.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(
        classification_report(
            all_labels, all_preds, target_names=target_names, digits=4
        )
    )


if __name__ == "__main__":
    main()
