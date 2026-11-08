"""
Control experiment analysis: train on guilleo vs. train on a different player, test on guilleo.

    python compare_control.py                 # control tag defaults to control_Optemm
    python compare_control.py control_Optemm

Everything is paired on the SAME test positions (seed-averaged), with 95% intervals that resample test games.
Prints, for each training size:
    own      = guilleo-trained minus zero-shot
    control  = control-trained minus zero-shot
    own-ctrl = guilleo-trained minus control-trained   <- the personalization effect
and saves finetune_results/control_comparison.png.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "finetune_results")
TAG = sys.argv[1] if len(sys.argv) > 1 else "control_Optemm"
FX = sys.argv[2] if len(sys.argv) > 2 else ""          # e.g. 4 for runs made with --fixed_epochs 4
FXS = f"_fx{FX}" if FX else ""
LR = "2e-05"
SIZES = [10, 25, 50, 100, 200]
METRICS = [("top1_hit", "Top-1 change (points)", 100), ("top3_hit", "Top-3 change (points)", 100),
           ("nll", "NLL change (negative = better)", 1)]

base = pd.read_csv(os.path.join(RES, "test_preds_n0_s0.csv"))
games = base["game_id"].to_numpy()
rng = np.random.default_rng(0)


def load(n, suffix):
    pat = os.path.join(RES, f"test_preds_n{n}_s*_lr{LR}_fbn_nodo{suffix}.csv")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    dfs = [pd.read_csv(f) for f in files]
    for d in dfs:
        assert (d["fen"].values == base["fen"].values).all(), "result files are not aligned"
    return {c: np.mean([d[c].astype(float).values for d in dfs], axis=0) for c, _, _ in METRICS}, len(files)


def boot(diff, n_boot=5000):
    g = pd.DataFrame({"d": diff, "g": games}).groupby("g")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    b = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return diff.mean(), lo, hi


zero = {c: base[c].astype(float).values for c, _, _ in METRICS}
res = {k: {c: [] for c, _, _ in METRICS} for k in ("own", "ctl", "diff")}
sizes_done = []
for n in SIZES:
    own, ctl = load(n, FXS), load(n, f"{FXS}_{TAG}")
    if own is None or ctl is None:
        print(f"size {n}: missing own or control files, skipped")
        continue
    sizes_done.append(n)
    (own_m, n_own), (ctl_m, n_ctl) = own, ctl
    line = []
    for c, label, sc in METRICS:
        a = boot(own_m[c] - zero[c]); b = boot(ctl_m[c] - zero[c]); d = boot(own_m[c] - ctl_m[c])
        res["own"][c].append(a); res["ctl"][c].append(b); res["diff"][c].append(d)
        name = label.split(" change")[0]
        line.append(f"{name}: own {a[0]*sc:+.3f} | ctrl {b[0]*sc:+.3f} | own-ctrl {d[0]*sc:+.3f} [{d[1]*sc:+.3f}, {d[2]*sc:+.3f}]")
    print(f"{n:4d} games ({n_own} own seeds, {n_ctl} control seeds)")
    for l in line:
        print("      " + l)

if sizes_done:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (c, label, sc) in zip(axes, METRICS):
        ax.axhline(0, color="gray", ls="--")
        for key, color, name, off in (("own", "tab:blue", "trained on guilleo", 0.97), ("ctl", "tab:orange", f"trained on control ({TAG.replace('control_', '')})", 1.03)):
            m = np.array([r[0] for r in res[key][c]]) * sc
            lo = np.array([r[1] for r in res[key][c]]) * sc
            hi = np.array([r[2] for r in res[key][c]]) * sc
            ax.errorbar(np.array(sizes_done) * off, m, yerr=[m - lo, hi - m], marker="o", capsize=3, color=color, label=name)
        ax.set_xscale("log"); ax.set_xticks(sizes_done); ax.set_xticklabels([str(s) for s in sizes_done])
        ax.set_xlabel("training games"); ax.set_ylabel(label); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Change vs. zero-shot Maia-2 on guilleo's test games: own games vs. another player's games")
    fig.tight_layout()
    out = os.path.join(RES, f"control_comparison{FXS}.png")
    fig.savefig(out, dpi=200)
    print("saved", out)