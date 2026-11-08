"""
Collect a CONTROL player's games and build nested training sets for the control experiment.

Save this file in your `data` folder (next to collect_player_games.py) and run:
    python collect_control_player.py

It never touches guilleo's files. Output (under  data/control set/<USERNAME>/ ):
    train_10_elo.csv, train_25_elo.csv, train_50_elo.csv, train_100_elo.csv, train_200_elo.csv
Each file has the same columns as guilleo's *_elo.csv files:
    game_id, date, ply, color, player_elo, fen, move, elo_self, elo_oppo

Games that also appear in guilleo's data (for example games between the two players) are removed,
so nothing from guilleo's test games can leak into the control training sets.
"""
import os
import requests
import chess
import chess.pgn
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

USERNAME = "Optemm"        # control player (public Lichess account)
PERF_TYPE = "rapid"
MAX_GAMES = 400
SKIP_OPENING_PLY = 10
SIZES = [10, 25, 50, 100, 200]

GUILLEO_POSITIONS_CSV = os.path.join(BASE, "player_positions.csv")   # used only to exclude shared games
PGN_PATH = os.path.join(BASE, f"games_{USERNAME}.pgn")
OUT_DIR = os.path.join(BASE, "control set", USERNAME)


def download_games():
    url = f"https://lichess.org/api/games/user/{USERNAME}"
    params = {"max": MAX_GAMES, "rated": "true", "perfType": PERF_TYPE, "moves": "true", "tags": "true",
              "clocks": "false", "evals": "false", "opening": "false"}
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": "USD-class-project (student research)"}
    print(f"Downloading up to {MAX_GAMES} {PERF_TYPE} games for {USERNAME}...")
    r = requests.get(url, params=params, headers=headers, timeout=180)
    if r.status_code != 200:
        print(f"Failed (HTTP {r.status_code})")
        print("URL sent:", r.url)
        print("Response:", r.text[:300])
        return False
    with open(PGN_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)
    return True


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def convert():
    shared = set()
    if os.path.isfile(GUILLEO_POSITIONS_CSV):
        shared = set(pd.read_csv(GUILLEO_POSITIONS_CSV, usecols=["game_id"])["game_id"])
        print(f"Excluding games that also appear in guilleo's data (guilleo has {len(shared)} games).")
    else:
        print("WARNING: guilleo's player_positions.csv not found, so shared games cannot be excluded.")

    rows, order = [], []          # `order` keeps games newest-first, as Lichess returns them
    dropped_shared = dropped_noelo = 0
    me = USERNAME.lower()
    with open(PGN_PATH, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            if h.get("Variant", "Standard") != "Standard":
                continue
            white, black = h.get("White", "").lower(), h.get("Black", "").lower()
            if me == white:
                color, e_self, e_opp = chess.WHITE, to_int(h.get("WhiteElo")), to_int(h.get("BlackElo"))
            elif me == black:
                color, e_self, e_opp = chess.BLACK, to_int(h.get("BlackElo")), to_int(h.get("WhiteElo"))
            else:
                continue
            game_id = h.get("Site", "").split("/")[-1]
            if game_id in shared:
                dropped_shared += 1
                continue
            if e_self is None or e_opp is None:
                dropped_noelo += 1
                continue

            board = game.board()
            ply, used = 0, False
            for move in game.mainline_moves():
                if board.turn == color and ply >= SKIP_OPENING_PLY:
                    rows.append({"game_id": game_id, "date": h.get("UTCDate", ""), "ply": ply,
                                 "color": "white" if color == chess.WHITE else "black",
                                 "player_elo": e_self, "fen": board.fen(), "move": move.uci(),
                                 "elo_self": e_self, "elo_oppo": e_opp})
                    used = True
                board.push(move)
                ply += 1
            if used:
                order.append(game_id)

    df = pd.DataFrame(rows)
    print(f"Dropped as shared with guilleo: {dropped_shared} | dropped for missing rating: {dropped_noelo}")
    print(f"Games used: {len(order)}")
    print(f"Positions: {len(df)}")
    if df.empty:
        return
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Elo range: {df['player_elo'].min()} to {df['player_elo'].max()} (mean {df['player_elo'].mean():.0f})")
    print(f"Moves as White: {(df['color'] == 'white').sum()}, as Black: {(df['color'] == 'black').sum()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    for n in SIZES:
        if n > len(order):
            print(f"Skipping train_{n}_elo.csv (only {len(order)} games available)")
            continue
        sub = df[df["game_id"].isin(order[:n])]       # the n most recent games (nested sets)
        path = os.path.join(OUT_DIR, f"train_{n}_elo.csv")
        sub.to_csv(path, index=False)
        print(f"train_{n}_elo.csv: {n} games, {len(sub)} positions")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    if os.path.exists(PGN_PATH) or download_games():
        convert()