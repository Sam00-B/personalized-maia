"""
Analyze ONE of guilleo's games with three Maia-2 models:
    1. original (pretrained, no fine-tuning)
    2. trained on guilleo's games
    3. trained on Optemm's games (the control)

Put this file in your `Finetune Maia` folder (next to finetune_maia2.py) and run, for example:

    python analyze_one_game.py                    # a random TEST game (different every time)
    python analyze_one_game.py --seed 7           # a reproducible random choice
    python analyze_one_game.py --game_id abcd1234 # a specific game from the test set

The game is always taken from the TEST set, so none of the three models has seen it in training.
Needed model files (create them with finetune_maia2.py ... --save_model):
    finetune_results/models/guilleo_200games_s0_lr2e-05_fbn_nodo.pt
    finetune_results/models/Optemm_200games_s0_lr2e-05_fbn_nodo_fx4.pt      (use the _fx4 file: trained 4 epochs, no early stop)

Output: a move-by-move table in the terminal, a CSV, and a figure in finetune_results/.
"""
import argparse
import os
import random

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from maia2 import model as maia_model
from maia2.inference import preprocessing
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
RES = os.path.join(BASE, "finetune_results")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None, help="seed for choosing the random game")
ap.add_argument("--game_id", default=None, help="analyze this game id (must be in the test set)")
ap.add_argument("--guilleo_model", default=os.path.join(RES, "models", "guilleo_200games_s0_lr2e-05_fbn_nodo.pt"))
ap.add_argument("--optemm_model", default=os.path.join(RES, "models", "Optemm_200games_s0_lr2e-05_fbn_nodo_fx4.pt"))
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
all_moves = get_all_possible_moves()
all_moves_dict = {m: i for i, m in enumerate(all_moves)}
rev = {i: m for m, i in all_moves_dict.items()}
elo_dict = create_elo_dict()


def load_model(path=None):
    m = maia_model.from_pretrained(type="rapid", device=device)
    if path:
        m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def move_probs(model, fen, e_self, e_oppo):
    """Probability of every legal move (unrounded), in real board orientation."""
    board_t, s_cat, o_cat, legal = preprocessing(fen, int(e_self), int(e_oppo), elo_dict, all_moves_dict)
    logits, _, _ = model(board_t.unsqueeze(0).to(device), torch.tensor([s_cat]).to(device),
                         torch.tensor([o_cat]).to(device))
    logits = logits[0].masked_fill(legal.to(device) == 0, float("-inf"))
    probs = torch.softmax(logits, dim=-1).cpu()
    black = fen.split(" ")[1] == "b"
    out = {}
    for idx in legal.nonzero().flatten().tolist():
        mv = rev[idx]
        out[mirror_move(mv) if black else mv] = float(probs[idx])
    return out


def san(fen, uci):
    b = chess.Board(fen)
    return b.san(chess.Move.from_uci(uci))


# ---------------------------------------------------------------- choose the game
test = pd.read_csv(os.path.join(ROOT, "test set", "test_elo.csv"))
game_ids = sorted(test["game_id"].unique())
if args.game_id:
    if args.game_id not in game_ids:
        raise SystemExit(f"Game {args.game_id} is not in the test set. Test game ids start with: {game_ids[:5]}")
    gid = args.game_id
else:
    gid = random.Random(args.seed).choice(game_ids)
game = test[test["game_id"] == gid].sort_values("ply").reset_index(drop=True)
print(f"Game: https://lichess.org/{gid}   ({len(game)} guilleo moves, test set, played as {game['color'][0]})")
print(f"Date {game['date'][0]} | guilleo rated {game['elo_self'][0]} vs opponent {game['elo_oppo'][0]}\n")

# ---------------------------------------------------------------- load the three models
models = {"original": load_model()}
for name, path in (("guilleo-trained", args.guilleo_model), ("Optemm-trained", args.optemm_model)):
    if os.path.isfile(path):
        models[name] = load_model(path)
    else:
        print(f"WARNING: {name} model not found at {path}, skipping it.")

# ---------------------------------------------------------------- analyze every move
rows = []
for _, r in game.iterrows():
    row = {"ply": int(r["ply"]), "move_no": int(r["ply"]) // 2 + 1, "played": san(r["fen"], r["move"])}
    for name, m in models.items():
        probs = move_probs(m, r["fen"], r["elo_self"], r["elo_oppo"])
        ranked = sorted(probs, key=probs.get, reverse=True)
        p = probs.get(r["move"], 0.0)
        row[f"{name}|p"] = p
        row[f"{name}|rank"] = ranked.index(r["move"]) + 1 if r["move"] in ranked else None
        row[f"{name}|top1"] = san(r["fen"], ranked[0])
        row[f"{name}|top1_p"] = probs[ranked[0]]
        row[f"{name}|top3"] = ", ".join(f"{san(r['fen'], u)} {probs[u]:.2f}" for u in ranked[:3])
    rows.append(row)
df = pd.DataFrame(rows)

# ---------------------------------------------------------------- terminal table
names = list(models)
show = pd.DataFrame({"move": df["move_no"].astype(str) + ". " + df["played"]})
for n in names:
    show[f"{n}: p(played)"] = df[f"{n}|p"].map(lambda x: f"{x:.2f}")
    show[f"{n}: rank"] = pd.to_numeric(df[f"{n}|rank"]).astype("Int64").astype(str)
    show[f"{n}: top-1 guess"] = df[f"{n}|top1"] + " (" + df[f"{n}|top1_p"].map(lambda x: f"{x:.2f}") + ")"
print(show.to_string(index=False))

print("\nSummary for this game (lower NLL is better):")
summ = []
for n in names:
    ranks = df[f"{n}|rank"].astype(float)
    nll = -np.log(np.maximum(df[f"{n}|p"].to_numpy(), 1e-9))
    summ.append({"model": n, "moves": len(df), "top-1 hits": int((ranks == 1).sum()),
                 "top-3 hits": int((ranks <= 3).sum()), "mean NLL": round(float(nll.mean()), 3)})
print(pd.DataFrame(summ).to_string(index=False))

# moves where a model changes the outcome compared with the original
for n in names[1:]:
    fixed = df[(df["original|rank"] != 1) & (df[f"{n}|rank"] == 1)]
    broke = df[(df["original|rank"] == 1) & (df[f"{n}|rank"] != 1)]
    print(f"\n{n} vs original: fixed {len(fixed)} move(s) "
          f"({', '.join(fixed['move_no'].astype(str) + '. ' + fixed['played']) or 'none'}), "
          f"broke {len(broke)} ({', '.join(broke['move_no'].astype(str) + '. ' + broke['played']) or 'none'})")

# ---------------------------------------------------------------- save CSV + figure
os.makedirs(RES, exist_ok=True)
csv_path = os.path.join(RES, f"single_game_{gid}.csv")
df.to_csv(csv_path, index=False)

colors = {"original": "gray", "guilleo-trained": "tab:blue", "Optemm-trained": "tab:orange"}
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
x = df["move_no"].to_numpy()
for n in names:
    axes[0].plot(x, df[f"{n}|p"], marker="o", ms=4, lw=1.4, color=colors[n], label=n)
axes[0].set_ylabel("probability given to the move guilleo played")
axes[0].set_title(f"Game {gid}: how likely each model thought guilleo's actual move was")
axes[0].legend(); axes[0].grid(alpha=0.3)
width = 0.38
for k, n in enumerate(names[1:]):
    diff = np.log(np.maximum(df[f"{n}|p"], 1e-9)) - np.log(np.maximum(df["original|p"], 1e-9))
    axes[1].bar(x + (k - 0.5) * width, diff, width=width, color=colors[n], label=f"{n} vs original")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_ylabel("change in log-probability\n(above 0 = more like guilleo)")
axes[1].set_xlabel("guilleo's move number")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig_path = os.path.join(RES, f"single_game_{gid}.png")
fig.savefig(fig_path, dpi=200)
print(f"\nSaved: {csv_path}\nSaved: {fig_path}")