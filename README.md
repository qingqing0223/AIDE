# Audio Immunization Against Harmful Audio Editing with Diffusion Models

📌 Anonymous code release for double-blind review.

This repository provides the core implementation of AIDE, a prompt-aware safety framework for text-to-audio diffusion models.
It includes: (i) prompt-aware input perturbation, (ii) a lightweight Prompt Safety Module (PSM), and (iii) evaluation scripts.


---

<p align="center">
  <img src="https://raw.githubusercontent.com/qingqing0223/AIDE/main/assets/framework_crop.png" width="900" />
</p>



---

## 📝 Abstract

Text-to-audio diffusion systems have become a foundation for prompt-guided audio generation and editing. In public deployment, malicious prompts shift deployed generators toward harmful outputs. This behavior introduce safety risks for released speech conditions. Existing defense approaches mainly apply the same suppression strategy to all prompts regardless of their content, which degrades benign generation quality. This paper proposes Audio Immunization Against Harmful Audio Editing with Diffusion Models (AIDE), a prompt-aware framework for protecting audio editing based on diffusion models with unchanged generator parameters. AIDE frames protection as the coordination between encoder-latent immunization and risk control during denoising. For each input speech condition, AIDE optimizes an imperceptible perturbation that forms a protected condition in encoder latent space. A lightweight prompt safety module maps prompt risk to continuous gates and regulates how strongly the protected condition is expressed during denoising. The framework suppresses harmful prompt realization while preserving benign controllability under the original generator. Experiments on AudioCaps and VCTK with the Audio Latent Diffusion Model (AudioLDM) show that AIDE maintains audio quality close to the unprotected generator and reduces harmful-prompt alignment compared with perturbation-only protection. The results indicate that coupling encoder-space immunization with prompt-conditioned continuous control provides a practical path toward reliable audio protection in deployed diffusion systems.



## 🚦 Project Status

Full release after the official publication.

| Component | Status | Timeline |
|---|---:|---:|
| Paper | Submitted | - |
| Code  | Initial Release (Anonymous) | 2026 |
| Full release (optional artifacts/demos) | Planned | After publication |

---

## 📂 Repository Structure

```text
AIDE/
├── src/                    # Core implementation
├── scripts/                # Entry scripts (train / infer / eval)
├── pgd/                    # Perturbation utilities
├── psm/                    # Prompt Safety Module (PSM)
├── tools/                  # Auxiliary tools
└── assets/                 # Figures used in README
```
## 🛠️ Environment Setup

We recommend using Conda to manage the environment.
1) Create environment
conda create -n aide python=3.10 -y
conda activate aide
2) Install dependencies
pip install -U pip
pip install torch torchvision torchaudio
pip install numpy scipy tqdm pyyaml
pip install librosa soundfile audioread
pip install transformers sentencepiece
3) Quick sanity check
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import librosa, soundfile; print('audio libs ok')"

## 💾 Data Preparation

This repository does NOT provide datasets, checkpoints, or generated results.
Typical evaluation datasets:
·AudioCaps (environmental audio captioning benchmark)
·VCTK (multi-speaker speech corpus)


## 🚀 Training

AIDE typically follows a two-stage protocol:
·Trainaudio immunization (generate perturbations / immunized latents)
·Train PSM (prompt classifier + continuous gate calibration)


## 🧪 Inference and Evaluation

·Run guarded pipeline
· Batch / multi-prompt runs
· Offline evaluation
|Outputs (generated audio / caches / logs / metrics tables) are NOT included in this repo.

---

## 🔒 Anonymity Notes

This repository is intended for double-blind review:
No author names, institutions, emails, or personal links are included.
Please avoid committing logs, absolute paths, or any identifying metadata.
> Note (Anonymous Release): This repo contains the core code. Large checkpoints / datasets are not included.


## 🙏 Credits

This anonymous implementation is built upon and inspired by prior open-source work in:
text-to-audio diffusion (AudioLDM-style pipelines)
contrastive audio-text encoders (CLAP-style)
adversarial audio perturbations and perceptual constraints
prompt safety classification and continual learning techniques
We thank the original authors and open-source contributors for releasing their code and models.


## 📌 Citation

@inproceedings{aide2026, 
  title     = {Audio Immunization Against Harmful Audio Editing with Diffusion Models},   
  author    = {Anonymous Authors},   
  booktitle = {Anonymous Submission},   
  year      = {2026}   
}
