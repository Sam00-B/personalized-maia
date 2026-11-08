"""
Maia Single-Game Evaluator
============================
This script tests Maia's move-prediction accuracy against ONE specific,
real game -- e.g. a random game you (or someone else) played on Lichess
or Chess.com -- rather than a big mixed dataset.

For every move where it was your chosen player's turn (White or Black),
it asks Maia what move IT would have picked, then checks whether that
matches what the real human actually played. At the end it reports:
  - Top-1 accuracy: how often Maia's #1 guess matched the real move
  - Top-K accuracy: how often the real move was in Maia's top K guesses

WHERE TO GET THE GAME FROM:
  Easiest: open the game on Lichess or Chess.com, use "Download PGN" /
  "Share -> PGN", and save it as a plain text file (e.g. "mygame.pgn").
  Point PGN_FILE_PATH below at that file.

  Alternative (Lichess only): if you know the game's ID (the string of
  letters/numbers in the game's URL, e.g. lichess.org/abcd1234 -> "abcd1234"),
  set GAME_SOURCE = "lichess_id" and LICHESS_GAME_ID below, and this
  script will fetch the PGN directly over the internet using Lichess's
  public API. Requires the "requests" package (pip install requests).
"""

import io                    # lets us treat a PGN string like a file, for chess.pgn
import os                    # for building file paths that work on any machine
import chess                 # Core chess logic: boards, moves, FEN parsing
import chess.pgn             # Parses PGN game notation into move sequences
import chess.engine           # Lets Python talk to a UCI chess engine (Lc0)
from tqdm import tqdm         # Adds a progress bar while looping


# ---------------------------------------------------------------------------
# STEP 0: Configuration -- tweak these values as needed
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

LC0_DIR = os.path.join(BASE, "lc0-v0.32.1-windows-cpu-dnnl")
LC0_PATH = os.path.join(LC0_DIR, "lc0.exe")
WEIGHTS_PATH = os.path.join(BASE, "maia-chess", "weights", "maia-1500.pb.gz")

# --- Where to get the game from ---
GAME_SOURCE = "file"          # "file" or "lichess_id"
PGN_FILE_PATH = os.path.join(BASE, "mygame.pgn")   # used if GAME_SOURCE == "file"
'''LICHESS_GAME_ID = "abcd1234"  '''                      # used if GAME_SOURCE == "lichess_id"

# --- What to evaluate ---
TARGET_COLOR = "white"   # "white" or "black" -- whose moves to check Maia against
SKIP_OPENING_PLY = 10    # skip the first N half-moves (memorized opening theory) --
                         # set to 0 if you want every move checked, openings included
TOP_K = 3                # how many of Maia's top guesses to consider a "hit"


# ---------------------------------------------------------------------------
# STEP 1: Load the game, either from a local file or from Lichess directly
# ---------------------------------------------------------------------------
def load_game_from_file(pgn_path):
    """Reads a single game from a local .pgn file."""
    if not os.path.exists(pgn_path):
        print(f"PGN file not found at: {pgn_path}")
        return None

    with open(pgn_path, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    if game is None:
        print("Could not parse a game from that PGN file -- is it empty or malformed?")
    return game


def load_game_from_lichess(game_id):
    """
    Fetches a game's PGN directly from Lichess's public API using its
    game ID, then parses it. Requires an internet connection and the
    'requests' package.
    """
    try:
        import requests
    except ImportError:
        print("The 'requests' package is required for this. Install it with:")
        print("    pip install requests")
        return None

    url = f"https://lichess.org/game/export/{game_id}?literate=false"
    print(f"Fetching PGN from: {url}")

    response = requests.get(url, headers={"Accept": "application/x-chess-pgn"})
    if response.status_code != 200:
        print(f"Failed to fetch game (HTTP {response.status_code}). "
              f"Double-check the game ID.")
        return None

    pgn_text = response.text
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    if game is None:
        print("Could not parse the PGN returned by Lichess.")
    return game


# ---------------------------------------------------------------------------
# STEP 2: Walk through the game move-by-move, testing Maia at each of the
#         target player's turns
# ---------------------------------------------------------------------------
def evaluate_game(game, target_color, skip_opening_ply, top_k):
    """
    Steps through every move of the game in order. Whenever it's
    `target_color`'s turn to move (and we're past the opening-skip
    threshold), it asks Maia/Lc0 for its top-K predicted moves and
    checks whether the human's actual move matches.
    """
    print("Initializing lc0 engine...")
    engine = chess.engine.SimpleEngine.popen_uci(
        [LC0_PATH, f"--weights={WEIGHTS_PATH}", "--threads=1", "--backend=blas"],
        cwd=LC0_DIR
    )

    target = chess.WHITE if target_color.lower() == "white" else chess.BLACK

    board = game.board()          # starting position
    ply = 0                       # counts every half-move from the start
    total = 0
    top_1_correct = 0
    top_k_correct = 0
    mismatches = []               # keep a record of moves Maia got wrong, for review

    # game.mainline_moves() gives the actual sequence of moves that was
    # played in the real game, in order
    moves = list(game.mainline_moves())

    for move in tqdm(moves, desc="Walking through game"):
        is_targets_turn = (board.turn == target)

        if is_targets_turn and ply >= skip_opening_ply:
            fen_before_move = board.fen()
            played_move_uci = move.uci()

            analysis = engine.analyse(
                board,
                chess.engine.Limit(nodes=1),
                multipv=top_k
            )

            predicted_moves = []
            for info in analysis:
                if "pv" in info and len(info["pv"]) > 0:
                    predicted_moves.append(info["pv"][0].uci())

            if predicted_moves:
                total += 1
                is_top1 = played_move_uci == predicted_moves[0]
                is_topk = played_move_uci in predicted_moves

                if is_top1:
                    top_1_correct += 1
                if is_topk:
                    top_k_correct += 1

                if not is_top1:
                    mismatches.append({
                        "ply": ply,
                        "move_number": (ply // 2) + 1,
                        "fen": fen_before_move,
                        "played": played_move_uci,
                        "maia_predicted": predicted_moves,
                    })

        # Advance the board by actually playing the move, regardless of
        # whose turn it was -- we need the board to stay in sync with
        # the real game for every subsequent position
        board.push(move)
        ply += 1

    engine.quit()

    return total, top_1_correct, top_k_correct, mismatches


# ---------------------------------------------------------------------------
# STEP 3: Print a neatly formatted results summary
# ---------------------------------------------------------------------------
def print_results(total, top_1_correct, top_k_correct, top_k, target_color, mismatches):
    top_1_acc = (top_1_correct / total) * 100 if total > 0 else 0
    top_k_acc = (top_k_correct / total) * 100 if total > 0 else 0

    print("\n" + "=" * 55)
    print("        MAIA SINGLE-GAME EVALUATION RESULTS        ")
    print("=" * 55)
    print(f"Player Evaluated          : {target_color.capitalize()}")
    print(f"Total Moves Evaluated     : {total}")
    print(f"Top-1 Accuracy            : {top_1_acc:6.2f}%  ({top_1_correct}/{total})")
    print(f"Top-{top_k} Accuracy            : {top_k_acc:6.2f}%  ({top_k_correct}/{total})")
    print("=" * 55)

    if mismatches:
        print(f"\nMoves where the real player's move did NOT match Maia's "
              f"Top-1 guess ({len(mismatches)} shown):")
        print("-" * 55)
        for m in mismatches:
            print(f"Move {m['move_number']} ({target_color}): "
                  f"played {m['played']}, Maia's top guesses were {m['maia_predicted']}")


# ---------------------------------------------------------------------------
# STEP 4: Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if GAME_SOURCE == "file":
        game = load_game_from_file(PGN_FILE_PATH)
    elif GAME_SOURCE == "lichess_id":
        game = load_game_from_lichess(LICHESS_GAME_ID)
    else:
        print(f"Unknown GAME_SOURCE: {GAME_SOURCE!r} -- must be 'file' or 'lichess_id'")
        game = None

    if game is not None:
        total, top_1_correct, top_k_correct, mismatches = evaluate_game(
            game, TARGET_COLOR, SKIP_OPENING_PLY, TOP_K
        )

        if total == 0:
            print("No moves were evaluated -- check TARGET_COLOR and "
                  "SKIP_OPENING_PLY against how long the game actually was.")
        else:
            print_results(total, top_1_correct, top_k_correct, TOP_K, TARGET_COLOR, mismatches)