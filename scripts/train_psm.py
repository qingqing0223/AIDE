from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from aide.audio import load_audio, pad_or_trim
from aide.audioldm import AudioLDMAdapter
from aide.psm import (
    PromptSafetyModule,
    PromptDataset,
    collate_prompts,
    psm_loss,
    save_psm_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", default="checkpoints/psm")
    p.add_argument("--backbone", default="roberta-base")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--channel-dim", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--mu-benign", type=float, default=1.0)
    p.add_argument("--mu-harmful", type=float, default=1.0)
    p.add_argument("--lambda-g", type=float, default=1.0)
    p.add_argument("--lambda-r", type=float, default=1.0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--model-name", default="audioldm-s-full")
    p.add_argument("--generation-steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--classification-only",
        action="store_true",
        help="Debug mode. Omits the paper's generation-coupled L_gen term.",
    )
    return p.parse_args()


def load_batch_audio(paths, device, length=160000):
    if any(not p for p in paths):
        raise ValueError(
            "Generation-coupled PSM training requires clean_path and "
            "protected_path in the prompt manifest."
        )

    wavs = []
    for path in paths:
        wav = load_audio(path)
        wav = pad_or_trim(wav, length)
        wavs.append(wav)

    return torch.cat(wavs, dim=0).to(device)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptSafetyModule(
        backbone=args.backbone,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        channel_dim=args.channel_dim,
    ).to(device)

    adapter = None
    if not args.classification_only:
        adapter = AudioLDMAdapter(
            checkpoint=args.checkpoint,
            model_name=args.model_name,
            device=str(device),
        )

    train_ds = PromptDataset(args.train)
    val_ds = PromptDataset(args.val)

    collate = lambda b: collate_prompts(
        b, model.tokenizer, max_length=128
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0

        for batch_idx, batch in enumerate(train_loader):
            labels = batch["labels"].to(device)
            harmful = batch["is_harmful"].to(device)

            out_psm = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )

            sim_clean = sim_gated = None

            if not args.classification_only:
                clean = load_batch_audio(
                    batch["clean_paths"], device
                )
                protected = load_batch_audio(
                    batch["protected_paths"], device
                )

                with torch.no_grad():
                    z_clean = adapter.encode_audio_condition(
                        clean, differentiable=False
                    )
                    z_protected = adapter.encode_audio_condition(
                        protected, differentiable=False
                    )

                    clean_gen = adapter.generate_differentiable(
                        z_clean.unsqueeze(1),
                        steps=args.generation_steps,
                        guidance_scale=args.guidance,
                        seed=args.seed + epoch,
                        eta=0.0,
                    )

                    sim_clean = adapter.differentiable_text_audio_similarity(
                        clean_gen.detach(),
                        batch["texts"],
                    ).detach()

                z_gated = adapter.fuse_conditions(
                    z_clean,
                    z_protected,
                    out_psm.scalar_gate,
                    out_psm.channel_gate,
                )

                gated_gen = adapter.generate_differentiable(
                    z_gated,
                    steps=args.generation_steps,
                    guidance_scale=args.guidance,
                    seed=args.seed + epoch,
                    eta=0.0,
                )

                sim_gated = adapter.differentiable_text_audio_similarity(
                    gated_gen,
                    batch["texts"],
                )

            losses = psm_loss(
                out_psm,
                labels,
                harmful,
                sim_clean=sim_clean,
                sim_gated=sim_gated,
                mu_benign=args.mu_benign,
                mu_harmful=args.mu_harmful,
                lambda_g=args.lambda_g,
                lambda_r=args.lambda_r,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(losses["total"].detach())

            if batch_idx % 20 == 0:
                print(
                    f"epoch={epoch+1} batch={batch_idx} "
                    f"total={float(losses['total'].detach()):.5f} "
                    f"cls={float(losses['classification'].detach()):.5f} "
                    f"gate={float(losses['gate'].detach()):.5f} "
                    f"gen={float(losses['generation'].detach()):.5f}"
                )

        metrics = evaluate_psm(
            model,
            val_loader,
            device,
        )
        metrics["train_loss"] = running / max(len(train_loader), 1)
        print(json.dumps(metrics, indent=2))

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_psm_checkpoint(
                out / "psm_best.pt",
                model,
                optimizer,
                metrics,
                vars(args),
            )


if __name__ == "__main__":
    main()
