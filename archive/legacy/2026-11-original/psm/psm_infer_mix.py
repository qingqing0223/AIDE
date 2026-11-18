import os
import json
import argparse

import torch
import torchaudio

from psm_model import ROOT, PSMModel


def load_label_mapping():
    path = os.path.join(ROOT, "psm", "data", "label2id.json")
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    label2id = mapping["label2id"]
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return label2id, id2label


def load_psm(device):
    label2id, id2label = load_label_mapping()
    model = PSMModel(num_labels=len(label2id), proj_dim=256)
    ckpt = os.path.join(ROOT, "psm", "checkpoints", "psm_roberta_best.pt")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, id2label


def predict_gate(model, prompt: str, device):
    tokenizer = model.tokenizer
    enc = tokenizer(
        [prompt],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        gate = out["gate"]
        pred = torch.argmax(logits, dim=-1)[0].item()
        g = gate[0].item()
    return pred, g


def mix_delta(raw_wav, noise_wav, out_wav, gate: float):
    wav_raw, sr1 = torchaudio.load(raw_wav)
    wav_noise, sr2 = torchaudio.load(noise_wav)
    assert sr1 == sr2, "raw 和 noise 采样率不一致"

    T = min(wav_raw.shape[-1], wav_noise.shape[-1])
    wav_raw = wav_raw[..., :T]
    wav_noise = wav_noise[..., :T]

    wav_mix = wav_raw + gate * wav_noise
    max_val = wav_mix.abs().max()
    if max_val > 1.0:
        wav_mix = wav_mix / max_val

    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    torchaudio.save(out_wav, wav_mix, sr1)
    return out_wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="raw.wav 路径")
    parser.add_argument("--noise", type=str, required=True, help="noise.wav 路径")
    parser.add_argument("--prompt", type=str, required=True, help="文本 prompt")
    parser.add_argument("--out", type=str, required=True, help="输出 mixed.wav 路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, id2label = load_psm(device)

    label_id, g = predict_gate(model, args.prompt, device)
    label_name = id2label[label_id]
    print(f"[PSM] 预测 label={label_name}, gate={g:.4f}")

    out_path = mix_delta(args.raw, args.noise, args.out, g)
    print("已生成混合语音:", out_path)


if __name__ == "__main__":
    main()
