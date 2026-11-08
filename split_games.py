'''This splits by game and chronologically: the newest games become the test set, the next newest the validation set, and the older games form the training pool. That avoids leaking positions from the same game across sets, and it mimics real use (train on the past, predict the future). The training subsets are nested, so the 25-game set is contained in the 50-game set and so on, which keeps the learning curve clean.'''
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE, "player_positions.csv")

TEST_GAMES = 60
VAL_GAMES = 40
TRAIN_SIZES = [10, 25, 50, 100, 200]

df = pd.read_csv(INPUT_CSV)

# Lichess returns games newest-first, and the CSV keeps that order
game_ids = list(dict.fromkeys(df["game_id"]))
print(f"Total games: {len(game_ids)}")

test_ids = game_ids[:TEST_GAMES]
val_ids = game_ids[TEST_GAMES:TEST_GAMES + VAL_GAMES]
pool_ids = game_ids[TEST_GAMES + VAL_GAMES:]   # still newest-first


def save(name, ids):
    sub = df[df["game_id"].isin(ids)]
    path = os.path.join(BASE, name)
    sub.to_csv(path, index=False)
    print(f"{name:18s} games={len(ids):4d} positions={len(sub):6d} "
          f"dates {sub['date'].min()} to {sub['date'].max()}")


save("test.csv", test_ids)
save("val.csv", val_ids)
save("train_pool.csv", pool_ids)

for n in TRAIN_SIZES:
    if n <= len(pool_ids):
        # the n most recent games before the validation window (nested subsets)
        save(f"train_{n}.csv", pool_ids[:n])
    else:
        print(f"Skipping train_{n}.csv (pool only has {len(pool_ids)} games)")