# -*- coding: utf-8 -*-
# merge_all_metrics.py
from __future__ import annotations
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clap_csv", required=True)
    ap.add_argument("--fd_csv", required=True)
    ap.add_argument("--gate_csv", default="", help="可选：门控统计（没有就不填）")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    clap = pd.read_csv(args.clap_csv)
    fd = pd.read_csv(args.fd_csv)

    df = pd.merge(fd, clap, on=["mode"], how="outer", suffixes=("", "_clap"))
    # N_audio 以非空为准
    if "N_audio_clap" in df.columns:
        df["N_audio"] = df["N_audio"].fillna(df["N_audio_clap"])
        df.drop(columns=["N_audio_clap"], inplace=True)

    if args.gate_csv:
        gate = pd.read_csv(args.gate_csv)
        df = pd.merge(df, gate, on=["mode"], how="left")

    df.to_csv(args.out_csv, index=False)
    print(f"[OK] merged -> {args.out_csv}")


if __name__ == "__main__":
    main()
