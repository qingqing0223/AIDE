# AIDE reproducible implementation (v2)

This directory is the canonical implementation path for the KSEM 2026 paper. The original top-level scripts are retained as a historical snapshot and are not imported by this package.

## Implemented pipeline

1. Stage I — encoder-latent immunization: original AudioLDM VAE encoder, fixed target pool, sign-PGD minimizing target steering plus clean-latent separation, and a differentiable local time-frequency masking-threshold proxy.
2. Stage II — Prompt Safety Module: RoBERTa text backbone, seven-way risk classifier, scalar gate, latent-channel gate, Eq. (6) gate calibration, and Eq. (7) generation-coupled CLAP loss. AudioLDM UNet, CLAP, VAE and vocoder weights remain frozen.
3. Deployment — prompt-aware diffusion editing: Eq. (4) clean/protected latent interpolation before the DDIM reverse process, followed by the original AudioLDM VAE decoder and HiFi-GAN vocoder.

The implementation also includes an OWM gradient projector, incremental classifier expansion, perturbation metrics, AudioLDM distribution evaluation, benchmark generation, and smoke tests.

## Scientific boundaries and explicit engineering choices

The paper does not disclose the exact psychoacoustic threshold estimator, the complete names/mapping of all six harmful subclasses, or a fully specified source-latent injection API. Therefore:

- The local time-frequency masking envelope in aide_repro/pgd.py is an explicit engineering proxy for the paper's T_x, not a claim to reproduce an undisclosed implementation.
- configs/taxonomy.json deliberately leaves source-label maps empty. Fill them from the exact AART and RED-INSTRUCT releases. Unknown labels fail loudly; they are never silently merged into one class.
- The source-latent path follows the official AudioLDM style-transfer route: VAE-encode source audio, apply the DDIM forward-noising equation, then decode with text-conditioned DDIM. Eq. (4) is applied to the VAE latent before forward noising and reverse denoising. This is a concrete implementation choice for the paper's unspecified injection point.
- Generation-coupled PSM training uses a short deterministic DDIM trajectory by default (8 steps, t_start=4) for memory/runtime control. Final evaluation uses the configured full inference schedule. Record both settings in every experiment.
- The AudioLDM VAE mel frontend is reimplemented with differentiable torch operations and upstream AudioLDM mel/STFT settings. Validate latent shapes and reconstruction before large-scale runs.

Do not describe an engineering proxy or an unverified run as an original paper implementation/result.

## Environment

The paper reports Python 3.10, PyTorch 2.1.0, torchaudio 2.1.0, CUDA 12.1 and an NVIDIA A100. Use the matching PyTorch wheel for your CUDA host. A pinned installer is provided:

    bash scripts/install_env.sh

For CPU-only hosts, set TORCH_INDEX_URL to https://download.pytorch.org/whl/cpu before running the installer. The manual equivalent is:

    conda create -n aide python=3.10 -y
    conda activate aide
    pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    cd AIDE/repro_v2
    pip install -r requirements.txt

    git clone https://github.com/haoheliu/AudioLDM.git ../external/AudioLDM
    git -C ../external/AudioLDM checkout 4054cb418ba3d947d94f6ad302f1d6a320e2313c
    pip install -e ../external/AudioLDM --no-deps

    export AUDIOLDM_CACHE_DIR=/path/to/model_cache/audioldm
    python -m aide_repro.cli check-env

The upstream AudioLDM package downloads the audioldm-s-full-v2 checkpoint when first initialized unless a local checkpoint is supplied. Ensure the cache has sufficient disk space. Datasets, model weights, generated WAVs and machine-specific absolute paths are not committed.

## Data preparation

### 1. Audio manifest

The input directory should contain the split's WAV/FLAC files. Use the official train/validation/test split and do not combine splits.

    python -m aide_repro.cli manifest --input-root /data/AudioCaps/train_wav --output data/audio_manifest_train.jsonl --duration 10.0

The manifest records a stable audio ID, source SHA256, original sample rate/channels/duration, and model input length.

### 2. Fixed target pool

Provide the fixed target pool explicitly. The code selects one target per source audio deterministically from the pool using the run seed and audio ID. The selected target path and SHA256 are written to target_manifest.jsonl. Run separate controlled experiments for noise, VCTK speech, and mixed target pools; do not pool them without labeling the experiment.

### 3. Prompt taxonomy

The paper defines one benign category and six harmful subclasses. The released paper text does not fully enumerate all six class names. Edit configs/taxonomy.json after inspecting the exact dataset versions:

- aart_label_map: exact normalized AART category string to integer ID 1..6. If the AART CSV has no category column, provide an explicit __all__ key mapped to one chosen harmful subclass.
- red_file_map: RED-INSTRUCT JSON or JSONL filename stem to integer ID 1..6.

Then build prompt splits:

    python -m aide_repro.cli build-prompts --audiocaps-csv /data/AudioCaps/captions.csv --aart-csv /data/AART/prompts.csv --red-root /data/RED-INSTRUCT/harmful_questions --taxonomy configs/taxonomy.json --output-dir data/prompts

The builder adds AudioCaps captions as benign examples, maps harmful prompts using the explicit taxonomy, removes exact normalized duplicates, rejects conflicting labels, and performs deterministic stratified splits. Near-duplicate semantic clustering should be audited separately before final reporting.

## Run the complete chain

### Stage I: PGD

    python -m aide_repro.cli pgd --manifest data/audio_manifest_train.jsonl --target-root /data/target_pool/musan --output-dir runs/stage1_musan --model-name audioldm-s-full-v2 --epsilon 0.01 --step-size 0.001 --steps 40 --lambda-target 1.0 --lambda-far 0.2 --lambda-mask 0.001

Outputs:
- audio/<audio_id>_raw.wav
- audio/<audio_id>_delta.wav
- audio/<audio_id>_adv.wav
- latents/<audio_id>.pt with z_clean, z_protected, z_target
- pgd_metrics.jsonl and target_manifest.jsonl

Before a full run, use sample-limit 1 and inspect the output. Confirm the maximum perturbation is at most epsilon and the loss moves in the intended direction.

### Stage II: verify the real gradient path, then train PSM

    python -m aide_repro.cli gradient-check --audio runs/stage1_musan/audio/ONE_ID_raw.wav --model-name audioldm-s-full-v2

    python -m aide_repro.cli train-psm --train-jsonl data/prompts/psm_train.jsonl --dev-jsonl data/prompts/psm_dev.jsonl --latent-dir runs/stage1_musan/latents --output-dir runs/stage2_psm --model-name audioldm-s-full-v2 --text-model roberta-base --epochs 10 --batch-size 4 --generation-ddim-steps 8 --generation-t-start 4 --generation-every 1 --owm

The generation-coupled loss is enabled by default. It creates paired clean/gated audio with the same noise tensor, computes AudioLDM CLAP cosine similarities, and backpropagates through the differentiable DDIM loop, VAE decoder, vocoder and CLAP audio branch into the scalar/channel gates. The diffusion/CLAP/VAE/vocoder weights are frozen. The run writes best.pt and last.pt as resumable training checkpoints, plus best_inference.pt as a smaller deployment checkpoint without optimizer/OWM state.

If the gradient check reports a missing, zero or non-finite gradient, stop. Do not train PSM or claim Eq. (7) is active until the path passes.

### Deployment

    python -m aide_repro.cli infer --audio runs/stage1_musan/audio/ONE_ID_raw.wav --delta runs/stage1_musan/audio/ONE_ID_delta.wav --prompt "A calm environmental sound description." --psm-checkpoint runs/stage2_psm/best_inference.pt --output runs/inference/example.wav --model-name audioldm-s-full-v2 --ddim-steps 50 --guidance-scale 2.5

Infer writes a JSON sidecar with the predicted category distribution, scalar gate, channel gate, seed and sampler settings. By default it also writes paired clean/protected/gated generations using the same initial noise.

### Prompt-suite benchmark

The evaluation prompt JSONL must contain audio_id, text, label_id and is_harmful. audio_id links each prompt to a Stage-I source audio.

    python -m aide_repro.cli suite --prompt-jsonl data/prompts/psm_test_with_audio_ids.jsonl --pgd-audio-dir runs/stage1_musan/audio --psm-checkpoint runs/stage2_psm/best_inference.pt --output-dir runs/test_suite --ddim-steps 50 --guidance-scale 2.5

## Metrics and ablations

    python -m aide_repro.cli eval-perturbation --raw-dir runs/stage1_musan/audio --adv-dir runs/stage1_musan/audio --output runs/stage1_musan/perturbation_metrics.json

    python -m aide_repro.cli eval-clap --meta-root runs/test_suite --output runs/test_suite/clap_metrics.json --model-name audioldm-s-full-v2

    python -m aide_repro.cli eval-distribution --generated-dir runs/test_suite/benign --reference-dir /data/reference_audio/benign --output runs/test_suite/benign_distribution.json

Use the same filenames, sample rate, duration, sampler settings and random seeds for paired comparisons. The AudioLDM evaluator requires its pretrained feature-extractor dependencies/weights; install them before final metric runs.

Suggested ablation mapping:
- Full AIDE: default training and infer mode aide.
- No perturbation: infer mode clean.
- Perturbation-only/static protection (no PSM): infer mode protected.
- No soft gate: infer with hard-gate.
- No channel gate: infer with disable-channel-gate.
- No multi-class risk modeling: train with binary-risk.
- No generation-coupled/consistency term: train with no-generation-loss.
- No OWM: train without owm; compare with an otherwise identical OWM run.
- Incremental class expansion: resume an OWM-trained checkpoint with resume BASE/best.pt, expand-to NEW_CLASS_COUNT and owm; provide new JSONL labels in the appended ID range. When OWM and classifier expansion are combined, the runner freezes the pretrained RoBERTa backbone and applies the saved OWM projections to the trainable shared/head layers. This is the bounded-memory incremental implementation choice; record it in the experiment report. Include old-class and new-class examples in the dev JSONL to obtain separate old_class_mean_f1 and new_class_mean_f1 values.

The paper's Table 2 includes a “w/o Consistency Loss” row, but the method equations do not define a separately named fourth loss. In this implementation that ablation disables Eq. (7) generation-coupled supervision; state this mapping in reports.

## Interface and numerical checks

- AudioLDM VAE latent is expected to have 8 channels for the selected checkpoint. The PSM channel gate is therefore [B,8] and is broadcast as [B,8,1,1]; the code checks tensor compatibility at the fusion point.
- Clean/protected latents are encoded with the same frozen VAE, deterministic posterior mode, preprocessing and target frame count.
- DDIM sampling uses the upstream schedule arrays and eta=0. Paired calls must share the exact noise tensor.
- decode_latent bypasses the upstream convenience decoder's no_grad wrapper and calls the frozen VAE decoder and vocoder directly so gradients can reach the gate.
- The old main-branch PGD ascent-sign bug is not carried forward: the new optimizer minimizes the stated objective with delta -= step_size * sign(grad).
- The saved delta is a residual, not the target-pool waveform.

## One-command orchestration

After filling the exact taxonomy maps and setting the dataset paths, the complete train/test chain can be launched with the orchestration script:

    export TRAIN_AUDIO_ROOT=/data/train_audio
    export TEST_AUDIO_ROOT=/data/test_audio
    export TARGET_POOL=/data/target_pool
    export AUDIOCAPS_CSV=/data/AudioCaps/captions.csv
    export AART_CSV=/data/AART/prompts.csv
    export RED_ROOT=/data/RED-INSTRUCT
    export RUN_ROOT=/data/aide_runs
    bash scripts/run_full_pipeline.sh

The script keeps train and test audio manifests/latents separate, runs the Stage-I gradient check before PSM training, and only proceeds when that check passes. Optional REFERENCE_BENIGN_DIR and REFERENCE_HARMFUL_DIR variables enable FD/IS/KL evaluation. Use scripts/run_inference_ablations.sh for clean, static-protection, full, hard-gate and no-channel-gate comparisons.

## Tests

    pip install pytest
    pytest -q

Unit tests cover loss formulas, masking-loss finiteness and OWM projection. The integration gradient check requires the real AudioLDM checkpoint and model initialization.

## Reproducibility record

For every run, archive:
- Git commit SHA and git status --short
- Python, PyTorch, torchaudio, CUDA, GPU name
- AudioLDM source commit and checkpoint SHA256
- Dataset manifests and SHA256s
- taxonomy JSON and prompt split stats
- full command line, seed, PGD and DDIM parameters
- PGD metrics, PSM training JSONL, checkpoint SHA256
- generated WAVs, metadata sidecars and metric JSON

Large datasets/checkpoints/results stay outside GitHub. This repository contains the complete code and reproducibility instructions, not those external artifacts.
