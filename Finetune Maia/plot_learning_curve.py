"""
Learning-curve figure + table from finetune_results/learning_curve.csv.

    python plot_learning_curve.py            # uses lr 2e-5, freeze_bn + no_dropout runs
    python plot_learning_curve.py 5e-5       # another learning rate

Each fine-tuned size is plotted as the mean over seeds (dots = individual seeds,
bars = standard deviation across seeds). The dashed line / grey band is the
zero-shot Maia-2 baseline on the same test positions with its bootstrap 95% CI.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "finetune_results")
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5



def load_summary(path):
    """Read learning_curve.csv even if its early rows were written with fewer columns
    (before the freeze_bn / no_dropout columns existed)."""
    import csv
    new_cols = ["n_games", "seed", "lr", "freeze_backbone", "freeze_bn", "no_dropout", "best_epoch",
                "val_nll", "top1", "top1_lo", "top1_hi", "top3", "top3_lo", "top3_hi",
                "nll", "nll_lo", "nll_hi"]
    old_cols = [c for c in new_cols if c not in ("freeze_bn", "no_dropout")]
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            if len(r) == len(new_cols):
                rows.append(dict(zip(new_cols, r)))
            elif len(r) == len(old_cols):
                d = dict(zip(old_cols, r))
                d["freeze_bn"], d["no_dropout"] = "False", "False"
                rows.append(d)
    out = pd.DataFrame(rows)
    for c in ("n_games", "seed", "best_epoch"):
        out[c] = out[c].astype(int)
    for c in new_cols[7:] + ["lr"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("freeze_backbone", "freeze_bn", "no_dropout"):
        out[c] = out[c].astype(str).str.lower().eq("true")
    return out


df = load_summary(os.path.join(RES, "learning_curve.csv"))
for col in ("freeze_bn", "no_dropout", "freeze_backbone"):
    if col not in df:
        df[col] = False
    df[col] = df[col].fillna(False).astype(bool)

# the same run may appear twice if it was repeated: keep the latest
df = df.drop_duplicates(subset=["n_games", "seed", "lr", "freeze_backbone", "freeze_bn", "no_dropout"],
                        keep="last")
base = df[df["n_games"] == 0].iloc[-1]
ft = df[(df["n_games"] > 0) & np.isclose(df["lr"], LR) & df["freeze_bn"]
        & df["no_dropout"] & ~df["freeze_backbone"]]
if ft.empty:
    sys.exit(f"No fine-tuned rows found for lr={LR} with --freeze_bn --no_dropout.")

metrics = [("top1", "Top-1 accuracy (%)"), ("top3", "Top-3 accuracy (%)"), ("nll", "NLL (lower is better)")]

# ---- table ----
print(f"Baseline (zero-shot): " + ", ".join(f"{m} {base[m]:.2f}" for m, _ in metrics))
rows = []
for n, g in ft.groupby("n_games"):
    row = {"games": n, "seeds": len(g)}
    for m, _ in metrics:
        dec = 3 if m == "nll" else 2
        row[m] = f"{g[m].mean():.{dec}f} ± {g[m].std(ddof=0):.{dec}f}"
        row["d_" + m] = f"{g[m].mean() - base[m]:+.{dec}f}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

# ---- figure ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (m, label) in zip(axes, metrics):
    ax.axhline(base[m], color="gray", linestyle="--", label="zero-shot Maia-2")
    grp = ft.groupby("n_games")[m]
    xs, mean, std = grp.mean().index.values, grp.mean().values, grp.std(ddof=0).fillna(0).values
    ax.errorbar(xs, mean, yerr=std, marker="o", capsize=4, color="tab:blue", label="fine-tuned (mean ± sd over seeds)")
    ax.scatter(ft["n_games"], ft[m], color="tab:blue", alpha=0.3, s=15)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("training games from the player")
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Personalized Maia-2: accuracy vs. number of training games (held-out test games)")
fig.tight_layout()
out = os.path.join(RES, "learning_curve.png")
fig.savefig(out, dpi=200)
print("saved", out)


# ---------------------------------------------------------------------------
# Figure 2: paired difference vs zero-shot (needs the per-position result files)
# ---------------------------------------------------------------------------
import glob

base_file = os.path.join(RES, "test_preds_n0_s0.csv")
if os.path.isfile(base_file):
    b = pd.read_csv(base_file)
    games = b["game_id"].to_numpy()
    rng = np.random.default_rng(0)
    tag = f"_lr{LR:g}_fbn_nodo"
    cols = [("top1_hit", "Top-1 change (points)", 100), ("top3_hit", "Top-3 change (points)", 100),
            ("nll", "NLL change (negative = better)", 1)]
    sizes = sorted(ft["n_games"].unique())
    res = {c: [] for c, _, _ in cols}
    for n in sizes:
        files = sorted(glob.glob(os.path.join(RES, f"test_preds_n{n}_s*{tag}.csv")))
        dfs = [pd.read_csv(f) for f in files]
        for d in dfs:
            assert (d["fen"].values == b["fen"].values).all(), "result files are not aligned"
        for c, _, sc in cols:
            avg = np.mean([d[c].astype(float).values for d in dfs], axis=0)
            diff = avg - b[c].astype(float).values
            g = pd.DataFrame({"d": diff, "g": games}).groupby("g")["d"].agg(["sum", "count"])
            sm, ct = g["sum"].to_numpy(), g["count"].to_numpy()
            idx = rng.integers(0, len(g), size=(5000, len(g)))
            boot = sm[idx].sum(axis=1) / ct[idx].sum(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[c].append((diff.mean() * sc, lo * sc, hi * sc))

    print("\nPaired change vs zero-shot (seed-averaged, 95% CI resampling games):")
    for i, n in enumerate(sizes):
        print(f"{int(n):4d} games: " + " | ".join(
            f"{lab.split(' change')[0]} {res[c][i][0]:+.3f} [{res[c][i][1]:+.3f}, {res[c][i][2]:+.3f}]"
            for c, lab, _ in cols))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (c, lab, _) in zip(axes, cols):
        m_, lo_, hi_ = (np.array([r[k] for r in res[c]]) for k in range(3))
        ax.axhline(0, color="gray", linestyle="--")
        ax.errorbar(sizes, m_, yerr=[m_ - lo_, hi_ - m_], marker="o", capsize=4, color="tab:blue")
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(int(x)) for x in sizes])
        ax.set_xlabel("training games from the player")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    fig.suptitle("Fine-tuned minus zero-shot Maia-2, same test positions (paired; bars = 95% CI over test games)")
    fig.tight_layout()
    out2 = os.path.join(RES, "learning_curve_paired.png")
    fig.savefig(out2, dpi=200)
    print("saved", out2)
else:
    print("test_preds_n0_s0.csv not found: skipping the paired figure")