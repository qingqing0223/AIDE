# Environment

Recommended baseline:
- Python 3.10
- PyTorch/CUDA matched to the AutoDL image
- official AudioLDM source
- transformers 4.38+
- torchaudio
- librosa
- soundfile
- scipy
- scikit-learn

Install:
```bash
pip install -r requirements.txt
pip install -r requirements-audioldm.txt
pip install -e .
```

Optional generation metrics:
```bash
pip install -r requirements-eval.txt
```

Do not put pretrained checkpoints into Git. Record checkpoint path, source revision and SHA-256 in each experiment run.
