import os
import chess
import chess.engine
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE) 
LC0_DIR = os.path.join(ROOT, "lc0-v0.32.1-windows-cpu-dnnl")
LC0_PATH = os.path.join(LC0_DIR, "lc0.exe")
WEIGHTS_DIR = os.path.join(ROOT, "maia-chess", "weights")

DATA_CSV = os.path.join(ROOT, "test set", "test.csv")
MODELS = ["maia-1500", "maia-1600"]
TOP_K = 3


def evaluate(weights_path, df):
    engine = chess.engine.SimpleEngine.popen_uci(
        [LC0_PATH, f"--weights={weights_path}", "--threads=1", "--backend=blas"],
        cwd=LC0_DIR,
    )
    hit1, hitk, short_lists = [], [], 0
    for fen, move in tqdm(zip(df["fen"], df["move"]), total=len(df)):
        board = chess.Board(fen)
        info = engine.analyse(board, chess.engine.Limit(nodes=1), multipv=TOP_K)
        preds = [i["pv"][0].uci() for i in info if i.get("pv")]
        if len(preds) < min(TOP_K, board.legal_moves.count()):
            short_lists += 1
        hit1.append(bool(preds) and move == preds[0])
        hitk.append(move in preds)
    engine.quit()
    return np.array(hit1), np.array(hitk), short_lists


def bootstrap_ci(df, col, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    g = df.groupby("game_id")[col].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    accs = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return np.percentile(accs, [2.5, 97.5]) * 100


if __name__ == "__main__":
    df = pd.read_csv(DATA_CSV).dropna(subset=["fen", "move"]).reset_index(drop=True)
    print(f"Evaluating on {len(df)} positions from {df['game_id'].nunique()} games")

    for name in MODELS:
        w = os.path.join(WEIGHTS_DIR, f"{name}.pb.gz")
        if not os.path.isfile(w):
            print(f"Skipping {name}: weights not found at {w}")
            continue
        print(f"\n=== {name} ===")
        h1, hk, short = evaluate(w, df)
        out = df.copy()
        out["top1_hit"], out["topk_hit"] = h1, hk
        out.to_csv(os.path.join(BASE, f"results_{name}_{os.path.basename(DATA_CSV)}"),
                   index=False)

        for label, col in [("Top-1", "top1_hit"), ("Top-3", "topk_hit")]:
            lo, hi = bootstrap_ci(out, col)
            print(f"{label}: {out[col].mean()*100:.2f}%  (95% CI {lo:.2f} to {hi:.2f})")
        print(f"Positions where lc0 returned fewer lines than expected: {short}")