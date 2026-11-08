import os
import chess.pgn
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
USERNAME = "guilleo"
PGN_PATH = os.path.join(ROOT, "player_games.pgn")

SPLIT_FILES = [
    os.path.join("test set", "test.csv"),
    os.path.join("val set", "val.csv"),
    os.path.join("training set", "train_pool.csv"),
] + [os.path.join("training set", f"train_{n}.csv") for n in (10, 25, 50, 100, 200)]

# Build game_id -> (elo_self, elo_oppo) from the PGN headers only (fast)
rows = []
with open(PGN_PATH, "r", encoding="utf-8") as f:
    while True:
        h = chess.pgn.read_headers(f)
        if h is None:
            break
        gid = h.get("Site", "").split("/")[-1]
        white, black = h.get("White", "").lower(), h.get("Black", "").lower()
        if white == USERNAME.lower():
            me, opp = h.get("WhiteElo"), h.get("BlackElo")
        elif black == USERNAME.lower():
            me, opp = h.get("BlackElo"), h.get("WhiteElo")
        else:
            continue
        rows.append({"game_id": gid, "elo_self": me, "elo_oppo": opp})

elo = pd.DataFrame(rows)
elo["elo_self"] = pd.to_numeric(elo["elo_self"], errors="coerce")
elo["elo_oppo"] = pd.to_numeric(elo["elo_oppo"], errors="coerce")
print(f"Games with Elo info: {len(elo)}; missing opponent Elo: {elo['elo_oppo'].isna().sum()}")

for rel in SPLIT_FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        print("Skipping (not found):", rel)
        continue
    df = pd.read_csv(path).drop(columns=["elo_self", "elo_oppo"], errors="ignore")
    out = df.merge(elo, on="game_id", how="left")
    out_path = path.replace(".csv", "_elo.csv")
    out.to_csv(out_path, index=False)
    print(f"{rel}: {len(out)} rows, missing Elo rows: {out['elo_oppo'].isna().sum()}")