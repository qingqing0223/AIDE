import os
import json
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

from psm_model import (
    ROOT,
    PromptDataset,
    PSMCollator,
    PSMModel,
    compute_psm_loss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_label2id():
    label_file = os.path.join(ROOT, "psm", "data", "label_list.txt")
    labels = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            lb = line.strip()
            if lb:
                labels.append(lb)
    labels = sorted(set(labels))
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}

    mapping = {"label2id": label2id, "id2label": id2label}
    map_path = os.path.join(ROOT, "psm", "data", "label2id.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"[PSM] 已保存 label2id 到 {map_path}")
    return label2id, id2label


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label_ids"].to(device)

            outs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outs["logits"], dim=-1)

            all_labels.extend(label_ids.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1


def main():
    set_seed(42)

    train_path = os.path.join(ROOT, "psm", "data", "psm_train.jsonl")
    dev_path = os.path.join(ROOT, "psm", "data", "psm_dev.jsonl")

    label2id, id2label = load_label2id()
    num_labels = len(label2id)
    print(f"[PSM] 标签数: {num_labels}")

    train_ds = PromptDataset(train_path, label2id)
    dev_ds = PromptDataset(dev_path, label2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSMModel(num_labels=num_labels, proj_dim=256)
    model.to(device)

    collator = PSMCollator(model.tokenizer, max_len=128)
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10

    ckpt_dir = os.path.join(ROOT, "psm", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(ROOT, "psm", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "psm_train_log.txt")

    best_f1 = 0.0
    with open(log_path, "w", encoding="utf-8") as log_f:
        for epoch in range(1, num_epochs + 1):
            model.train()
            total_loss = 0.0
            step = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss, info = compute_psm_loss(model, batch, outs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                step += 1
                total_loss += loss.item()
                if step % 20 == 0:
                    print(
                        f"[Epoch {epoch}] step {step} "
                        f"loss={loss.item():.4f} "
                        f"(ce={info['ce']:.4f}, proto={info['proto']:.4f}, gate={info['gate']:.4f})"
                    )

            avg_loss = total_loss / max(1, step)
            dev_acc, dev_f1 = evaluate(model, dev_loader, device)
            print(
                f"==> Epoch {epoch}: train_loss={avg_loss:.4f}, "
                f"dev_acc={dev_acc:.4f}, dev_macro_f1={dev_f1:.4f}"
            )
            log_f.write(
                f"{epoch}\t{avg_loss:.6f}\t{dev_acc:.6f}\t{dev_f1:.6f}\n"
            )
            log_f.flush()

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                ckpt = os.path.join(ckpt_dir, "psm_roberta_best.pt")
                torch.save(model.state_dict(), ckpt)
                print(f"*** 保存最优模型到 {ckpt} (dev_macro_f1={dev_f1:.4f})")

    print("训练完成。")


if __name__ == "__main__":
    main()
