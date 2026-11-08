"""
Paired comparison of two runs on the SAME test positions.

    python compare_runs.py finetune_results/test_preds_n0_s0.csv ^
                           finetune_results/test_preds_n100_s0_lr2e-05_fbn_nodo.csv

Prints the change in Top-1, Top-3 and NLL (fine-tuned minus baseline) with a 95%
bootstrap CI that resamples whole GAMES. Because both models are scored on the
same positions, this is much more sensitive than comparing two separate
confidence intervals.
"""
import sys

import numpy as np
import pandas as pd


def main(base_path, ft_path, n_boot=5000, seed=0):
    a, b = pd.read_csv(base_path), pd.read_csv(ft_path)
    assert len(a) == len(b), "files have a different number of positions"
    assert (a["fen"].values == b["fen"].values).all() and (a["move"].values == b["move"].values).all(), \
        "files are not aligned on the same positions"

    rng = np.random.default_rng(seed)
    games = a["game_id"].values

    for col, label, scale in [("top1_hit", "Top-1 (points)", 100),
                              ("top3_hit", "Top-3 (points)", 100),
                              ("nll", "NLL (lower is better)", 1)]:
        x, y = a[col].astype(float).values, b[col].astype(float).values
        d = y - x
        g = pd.DataFrame({"d": d, "g": games}).groupby("g")["d"].agg(["sum", "count"])
        s, c = g["sum"].to_numpy(), g["count"].to_numpy()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{label:22s} {x.mean()*scale:7.2f} -> {y.mean()*scale:7.2f}   "
              f"diff {d.mean()*scale:+.3f}  (95% CI {lo*scale:+.3f} to {hi*scale:+.3f})")

    gained = int(((a["top1_hit"] == False) & (b["top1_hit"] == True)).sum())
    lost = int(((a["top1_hit"] == True) & (b["top1_hit"] == False)).sum())
    print(f"Top-1: fine-tuned model fixed {gained} positions and broke {lost}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])