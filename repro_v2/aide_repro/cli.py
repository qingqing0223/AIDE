from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .audio import load_audio
from .backend import AudioLDMBackend
from .benchmark import run_prompt_suite
from .data import build_audio_manifest, build_prompt_dataset
from .inference import infer_one, load_psm_checkpoint
from .metrics import evaluate_audio_distribution, evaluate_perturbation_pairs, evaluate_prompt_clap
from .pgd import LatentPGD
from .psm import PSMModel
from .trainer import PSMTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backend_from_args(args):
    return AudioLDMBackend(
        model_name=getattr(args, "model_name", "audioldm-s-full-v2"),
        device=getattr(args, "device", "auto"),
        checkpoint=getattr(args, "audio_checkpoint", None),
    )


def cmd_check_env(args):
    import torchaudio
    print(json.dumps({
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }, indent=2))
    try:
        import audioldm
        print("AudioLDM import: PASS")
    except Exception as exc:
        print(f"AudioLDM import: FAIL ({exc})")


def cmd_manifest(args):
    result = build_audio_manifest(args.input_root, args.output, args.duration, args.sample_rate)
    print(json.dumps({"num_files": len(result), "manifest": str(Path(args.output).resolve())}, indent=2))


def cmd_build_prompts(args):
    result = build_prompt_dataset(
        args.audiocaps_csv, args.aart_csv, args.red_root, args.taxonomy,
        args.output_dir, seed=args.seed
    )
    print(json.dumps(result, indent=2))


def cmd_pgd(args):
    set_seed(args.seed)
    backend = backend_from_args(args)
    optimizer = LatentPGD(
        backend, epsilon=args.epsilon, step_size=args.step_size, steps=args.steps,
        lambda_target=args.lambda_target, lambda_far=args.lambda_far,
        lambda_mask=args.lambda_mask, masking_offset_db=args.masking_offset_db,
        seed=args.seed,
    )
    result = optimizer.run_manifest(
        args.manifest, args.target_root, args.output_dir,
        sample_limit=args.sample_limit, duration_seconds=args.duration,
    )
    print(json.dumps(result, indent=2))


def cmd_train_psm(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    resume = torch.load(args.resume, map_location="cpu") if args.resume else None
    if resume:
        num_labels = int(resume["num_labels"])
        text_model = str(resume["text_model"])
        latent_channels = int(resume["latent_channels"])
    else:
        num_labels = 2 if args.binary_risk else int(args.num_labels)
        text_model = args.text_model
        latent_channels = int(args.latent_channels)
    model = PSMModel(
        num_labels=num_labels, latent_channels=latent_channels,
        text_model=text_model, local_files_only=args.local_files_only,
    )
    if resume:
        model.load_state_dict(resume["state_dict"], strict=True)
    if args.expand_to and args.expand_to > model.classifier.out_features:
        model.expand_classifier(args.expand_to)

    backend = None if args.no_generation_loss else backend_from_args(args)
    trainer = PSMTrainer(
        model=model, backend=backend, latent_dir=args.latent_dir,
        output_dir=args.output_dir, device=str(device),
        learning_rate=args.learning_rate, lambda_gate=args.lambda_gate,
        lambda_generation=args.lambda_generation, generation_every=args.generation_every,
        generation_ddim_steps=args.generation_ddim_steps,
        generation_t_start=args.generation_t_start,
        generation_guidance_scale=args.generation_guidance_scale,
        max_length=args.max_length, use_owm=args.owm,
        disable_generation_loss=args.no_generation_loss,
        disable_channel_gate=args.disable_channel_gate,
        binary_risk=args.binary_risk, seed=args.seed,
    )
    result = trainer.fit(args.train_jsonl, args.dev_jsonl, epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(result, indent=2))


def cmd_infer(args):
    set_seed(args.seed)
    backend = backend_from_args(args)
    psm, _ = load_psm_checkpoint(args.psm_checkpoint, backend.device, args.local_files_only)
    result = infer_one(
        backend, psm, args.audio, args.delta, args.prompt, args.output,
        duration_seconds=args.duration, ddim_steps=args.ddim_steps, t_start=args.t_start,
        guidance_scale=args.guidance_scale, seed=args.seed, mode=args.mode,
        hard_gate=args.hard_gate, disable_channel_gate=args.disable_channel_gate,
        save_paired=not args.no_paired, local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2))


def cmd_suite(args):
    set_seed(args.seed)
    backend = backend_from_args(args)
    result = run_prompt_suite(
        backend, args.psm_checkpoint, args.prompt_jsonl, args.pgd_audio_dir,
        args.output_dir, args.device, duration_seconds=args.duration,
        ddim_steps=args.ddim_steps, t_start=args.t_start,
        guidance_scale=args.guidance_scale, seed=args.seed,
        save_paired=not args.no_paired, local_files_only=args.local_files_only,
    )
    print(json.dumps(result, indent=2))


def cmd_eval_perturbation(args):
    result = evaluate_perturbation_pairs(args.raw_dir, args.adv_dir)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2))


def cmd_eval_distribution(args):
    result = evaluate_audio_distribution(args.generated_dir, args.reference_dir, args.device, args.limit_num)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def cmd_eval_clap(args):
    backend = backend_from_args(args)
    result = evaluate_prompt_clap(args.meta_root, backend)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sample"}, indent=2))


def cmd_gradient_check(args):
    backend = backend_from_args(args)
    waveform = load_audio(args.audio, duration_seconds=args.duration).to(backend.device)
    with torch.no_grad():
        latent = backend.encode_waveform(waveform)
    result = backend.assert_gradient_path(latent, args.prompt)
    print(json.dumps(result, indent=2))
    if not result["gradient_exists"] or not result["finite"] or result["gradient_norm"] <= 0:
        raise SystemExit("Gradient-path check FAILED. Do not start generation-coupled PSM training.")


def build_parser():
    parser = argparse.ArgumentParser(prog="aide-repro")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-env")
    p.set_defaults(func=cmd_check_env)

    p = sub.add_parser("manifest")
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("build-prompts")
    p.add_argument("--audiocaps-csv", required=True)
    p.add_argument("--aart-csv", required=True)
    p.add_argument("--red-root", required=True)
    p.add_argument("--taxonomy", default="configs/taxonomy.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_build_prompts)

    p = sub.add_parser("pgd")
    p.add_argument("--manifest", required=True)
    p.add_argument("--target-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--step-size", type=float, default=0.001)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lambda-target", type=float, default=1.0)
    p.add_argument("--lambda-far", type=float, default=0.2)
    p.add_argument("--lambda-mask", type=float, default=0.001)
    p.add_argument("--masking-offset-db", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_pgd)

    p = sub.add_parser("train-psm")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--dev-jsonl", required=True)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--text-model", default="roberta-base")
    p.add_argument("--latent-channels", type=int, default=8)
    p.add_argument("--num-labels", type=int, default=7)
    p.add_argument("--expand-to", type=int)
    p.add_argument("--resume")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--lambda-gate", type=float, default=1.0)
    p.add_argument("--lambda-generation", type=float, default=1.0)
    p.add_argument("--generation-every", type=int, default=1)
    p.add_argument("--generation-ddim-steps", type=int, default=8)
    p.add_argument("--generation-t-start", type=int, default=4)
    p.add_argument("--generation-guidance-scale", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--binary-risk", action="store_true")
    p.add_argument("--disable-channel-gate", action="store_true")
    p.add_argument("--no-generation-loss", action="store_true")
    p.add_argument("--owm", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_train_psm)

    p = sub.add_parser("infer")
    p.add_argument("--audio", required=True)
    p.add_argument("--delta", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--psm-checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--t-start", type=int)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--mode", choices=["aide", "clean", "protected"], default="aide")
    p.add_argument("--hard-gate", action="store_true")
    p.add_argument("--disable-channel-gate", action="store_true")
    p.add_argument("--no-paired", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("suite")
    p.add_argument("--prompt-jsonl", required=True)
    p.add_argument("--pgd-audio-dir", required=True)
    p.add_argument("--psm-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--t-start", type=int)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--no-paired", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--seed", type=int, default=2025)
    p.set_defaults(func=cmd_suite)

    p = sub.add_parser("eval-perturbation")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--adv-dir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_eval_perturbation)

    p = sub.add_parser("eval-distribution")
    p.add_argument("--generated-dir", required=True)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit-num", type=int)
    p.set_defaults(func=cmd_eval_distribution)

    p = sub.add_parser("eval-clap")
    p.add_argument("--meta-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_eval_clap)

    p = sub.add_parser("gradient-check")
    p.add_argument("--audio", required=True)
    p.add_argument("--prompt", default="A quiet rain sound.")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--model-name", default="audioldm-s-full-v2")
    p.add_argument("--audio-checkpoint")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_gradient_check)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
