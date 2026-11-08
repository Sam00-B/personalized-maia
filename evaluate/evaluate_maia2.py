import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from maia2 import model, inference

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
DATA_CSV = os.path.join(ROOT, "test set", "test_elo.csv")

m = model.from_pretrained(type="rapid", device="cpu")
prepared = inference.prepare()

df = pd.read_csv(DATA_CSV).dropna(subset=["fen", "move", "elo_self", "elo_oppo"]).reset_index(drop=True)
print(f"{len(df)} positions from {df['game_id'].nunique()} games")

hit1, hit3, nll = [], [], []
for fen, move, es, eo in tqdm(zip(df["fen"], df["move"], df["elo_self"], df["elo_oppo"]), total=len(df)):
    probs, _ = inference.inference_each(m, prepared, fen, int(es), int(eo))
    ranked = sorted(probs, key=probs.get, reverse=True)
    hit1.append(ranked[0] == move)
    hit3.append(move in ranked[:3])
    nll.append(-np.log(max(probs.get(move, 0.0), 1e-9)))

df["top1_hit"], df["top3_hit"], df["nll"] = hit1, hit3, nll
df.to_csv(os.path.join(BASE, "results_maia2_zeroshot_test.csv"), index=False)


def boot(col, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    g = df.groupby("game_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    v = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return np.percentile(v, [2.5, 97.5])


for label, col, scale in [("Top-1", "top1_hit", 100), ("Top-3", "top3_hit", 100), ("Mean NLL", "nll", 1)]:
    lo, hi = boot(col)
    print(f"{label}: {df[col].mean()*scale:.2f}  (95% CI {lo*scale:.2f} to {hi*scale:.2f})")