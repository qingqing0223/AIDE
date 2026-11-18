import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_step_metrics(root: Path, step: int):
    """
    读取某个 stepsX 目录下的 train/metrics.jsonl，
    计算该步数对应的平均 SNR / LSD。
    """
    metrics_path = root / f"steps{step}" / "train" / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.jsonl 不存在: {metrics_path}")

    snrs = []
    lsds = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 这里假定 delta_pgd.py 写入的键是 snr_db / lsd_db
            if "snr_db" in rec:
                snrs.append(float(rec["snr_db"]))
            if "lsd_db" in rec:
                lsds.append(float(rec["lsd_db"]))

    if not snrs or not lsds:
        raise RuntimeError(f"{metrics_path} 里没有 snr_db / lsd_db 数据")

    mean_snr = sum(snrs) / len(snrs)
    mean_lsd = sum(lsds) / len(lsds)
    return mean_snr, mean_lsd


def main():
    project_root = Path("/root/autodl-tmp/speech_guard")
    steps_root = project_root / "out" / "pgd_steps"
    out_dir = steps_root  # 图就直接放在 pgd_steps 下面

    # 这里的列表要和你实际跑的 steps 一一对应
    step_list = [1, 3, 5, 7, 10]

    mean_snrs = []
    mean_lsds = []

    for s in step_list:
        mean_snr, mean_lsd = load_step_metrics(steps_root, s)
        mean_snrs.append(mean_snr)
        mean_lsds.append(mean_lsd)
        print(f"[INFO] steps={s}: mean SNR={mean_snr:.3f} dB, mean LSD={mean_lsd:.3f} dB")

    # ===== 画图：两行子图，共用一个 x 轴 =====
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    ax1.plot(step_list, mean_snrs, marker="o")
    ax1.set_ylabel("Mean SNR (dB)")
    ax1.set_title("PGD δ 训练曲线：步数 vs 平均 SNR / LSD")
    ax1.grid(True, alpha=0.3)

    ax2.plot(step_list, mean_lsds, marker="o")
    ax2.set_xlabel("PGD steps")
    ax2.set_ylabel("Mean LSD (dB)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "pgd_train_curve_steps.png"
    fig.savefig(out_path, dpi=150)
    print(f"[OK] 训练曲线已保存到: {out_path}")


if __name__ == "__main__":
    main()
