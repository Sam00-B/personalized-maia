import io
import os
import time
import requests
import chess
import chess.pgn
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
import requests
'''
base = "https://lichess.org/api/games/user/guilleo"
ua = {"User-Agent": "USD-class-project (student research)"}

tests = {
    "1. minimal, no headers": dict(params={"max": 2, "perfType": "rapid"}, headers={}),
    "2. minimal + User-Agent": dict(params={"max": 2, "perfType": "rapid"}, headers=ua),
    "3. minimal + UA + Accept": dict(params={"max": 2, "perfType": "rapid"},
                                     headers={**ua, "Accept": "application/x-chess-pgn"}),
    "4. full params + UA": dict(params={"max": 2, "rated": "true", "perfType": "rapid",
                                        "moves": "true", "tags": "true", "clocks": "false",
                                        "evals": "false", "opening": "false"}, headers=ua),
}

for name, kw in tests.items():
    r = requests.get(base, timeout=60, **kw)
    print(name, "->", r.status_code, "|", r.text[:60].replace("\n", " "))'''

USERNAME = "guilleo"  # Lichess username to collect games for
PERF_TYPE = "rapid"        # "rapid", "blitz", or "classical"
MAX_GAMES = 400
SKIP_OPENING_PLY = 10

PGN_PATH = os.path.join(BASE, "player_games.pgn")
CSV_PATH = os.path.join(BASE, "player_positions.csv")

def download_games():
    url = f"https://lichess.org/api/games/user/{USERNAME}"
    print(repr(USERNAME), url)
    params = {
        "max": MAX_GAMES,
        "rated": "true",
        "perfType": PERF_TYPE,
        "moves": "true",
        "tags": "true",
        "clocks": "false",
        "evals": "false",
        "opening": "false",
    }
    headers = {    "Accept": "application/x-chess-pgn",
                    "User-Agent": "USD-class-project (student research)",}
    print(f"Downloading up to {MAX_GAMES} {PERF_TYPE} games for {USERNAME}...")
    r = requests.get(url, params=params, headers=headers, timeout=120)
    if r.status_code == 429:
        print("Rate limited. Wait a minute and try again.")
        return False
    if r.status_code != 200:
        print(f"Failed (HTTP {r.status_code})")
        print("URL sent:", r.url)
        print("Response:", r.text[:300])
        return False
    with open(PGN_PATH, "w", encoding="utf-8") as f:
        f.write(r.text)
    return True


def convert_to_positions():
    rows = []
    games_used = 0
    with open(PGN_PATH, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            if h.get("Variant", "Standard") != "Standard":
                continue
            white = h.get("White", "").lower()
            black = h.get("Black", "").lower()
            me = USERNAME.lower()
            if me == white:
                color = chess.WHITE
            elif me == black:
                color = chess.BLACK
            else:
                continue

            game_id = h.get("Site", "").split("/")[-1]
            board = game.board()
            ply = 0
            used = False
            for move in game.mainline_moves():
                if board.turn == color and ply >= SKIP_OPENING_PLY:
                    rows.append({
                        "game_id": game_id,
                        "date": h.get("UTCDate", ""),
                        "ply": ply,
                        "color": "white" if color == chess.WHITE else "black",
                        "player_elo": h.get("WhiteElo" if color == chess.WHITE else "BlackElo", ""),
                        "fen": board.fen(),
                        "move": move.uci(),
                    })
                    used = True
                board.push(move)
                ply += 1
            if used:
                games_used += 1

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Games used: {games_used}")
    print(f"Positions: {len(df)}")
    if len(df):
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Elo range: {df['player_elo'].min()} to {df['player_elo'].max()}")
        print(f"Moves as White: {(df['color']=='white').sum()}, "
              f"as Black: {(df['color']=='black').sum()}")
    print(f"Saved to {CSV_PATH}")


if __name__ == "__main__":
    if os.path.exists(PGN_PATH) or download_games():
        convert_to_positions()