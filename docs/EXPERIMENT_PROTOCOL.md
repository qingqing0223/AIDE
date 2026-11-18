# Experiment protocol

Main: clean AudioLDM / PGD-only / AIDE.

Datasets: AudioCaps and VCTK 0.92.

Ablations: no target term, no clean-separation term, no masking, scalar-only gate, channel-only gate, no PSM.

Metrics: FD, IS, KL, CLAP, harmful-prompt H-CLAP, SNR, LSD, PSM accuracy, Macro-F1 and Weighted-F1.

Each run records Git SHA, checkpoint hash, dataset/prompt manifest hashes, seed, CUDA information and config.
