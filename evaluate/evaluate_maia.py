"""
Maia Chess Model Evaluator
===========================
This script measures how well a Maia neural network model (running through
the Lc0 chess engine) predicts human moves from a dataset of real games.

It expects the dataset to already be filtered down to your target Elo
bucket with opening moves skipped -- that filtering is done once, ahead
of time, by prepare_dataset.py (which writes out
"maia-chess-testing-set-filtered.csv"). This script just:

  1. Loads that filtered CSV with pandas
  2. Randomly samples up to MAX_SAMPLES positions from it
  3. Runs Maia/Lc0 on each sampled position
  4. Reports Top-1 and Top-K move-prediction accuracy

Splitting the work this way means you can re-run evaluations with
different sample sizes or top_k values without re-scanning the full
raw dataset each time.
"""

import chess                # Core chess logic: boards, moves, FEN parsing
import chess.engine          # Lets Python talk to a UCI chess engine (Lc0)
import os                    # For building file paths that work on any machine
import pandas as pd          # For loading and randomly sampling the dataset
from tqdm import tqdm        # Adds a progress bar while looping


# ---------------------------------------------------------------------------
# STEP 0: Configuration -- tweak these values as needed
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

LC0_DIR = os.path.join(BASE, "lc0-v0.32.1-windows-cpu-dnnl")
LC0_PATH = os.path.join(LC0_DIR, "lc0.exe")
WEIGHTS_PATH = os.path.join(BASE, "maia-chess", "weights", "maia-1500.pb.gz")

# This should point at the FILTERED file produced by prepare_dataset.py,
# not the raw decompressed one -- it's already narrowed to your target
# Elo bucket with opening moves skipped.
DATASET_PATH = os.path.join(BASE, "maia-chess-testing-set-filtered.csv")

TOP_K = 3               # How many of Maia's top guesses to consider
MAX_SAMPLES = 500       # How many positions to randomly sample and evaluate
RANDOM_SEED = 42        # Fixed seed so results are reproducible between runs


# ---------------------------------------------------------------------------
# STEP 1: Load the filtered dataset and randomly sample rows from it
# ---------------------------------------------------------------------------
def load_sample(csv_path, max_samples, seed=42):
    """
    Loads the pre-filtered CSV and returns a random sample of up to
    `max_samples` rows as a list of (fen, played_move_uci) tuples.

    Since this file is already filtered down to a manageable size (unlike
    the huge raw dataset), it's fine to load it fully into memory with
    pandas and sample directly -- no need for reservoir sampling here.
    """
    print(f"Loading filtered dataset from:\n  {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Filtered dataset contains {len(df):,} rows.")

    # Don't ask for more samples than actually exist
    n = min(max_samples, len(df))

    # random_state makes the sample reproducible between runs
    sample_df = df.sample(n=n, random_state=seed)

    print(f"Randomly sampled {n} rows for evaluation.")

    samples = []
    for _, row in sample_df.iterrows():
        fen = row.get("board") if "board" in row else row.get("fen")
        played_move_uci = row.get("move") if "move" in row else row.get("played_move")
        if pd.isna(fen) or pd.isna(played_move_uci):
            continue
        samples.append((fen, played_move_uci))

    return samples


# ---------------------------------------------------------------------------
# STEP 2: Run Maia/Lc0 on the sampled positions and score accuracy
# ---------------------------------------------------------------------------
def evaluate_positions(samples, top_k=3):
    """
    Starts the Lc0 engine (loaded with Maia weights) and, for each
    (fen, played_move) pair in `samples`, asks it for its top-K move
    predictions, then checks whether the human's move matches.
    """
    print("Initializing lc0 engine...")
    engine = chess.engine.SimpleEngine.popen_uci(
        [LC0_PATH, f"--weights={WEIGHTS_PATH}", "--threads=1", "--backend=blas"],
        cwd=LC0_DIR
    )

    total = 0
    top_1_correct = 0
    top_k_correct = 0

    for fen, played_move_uci in tqdm(samples, desc="Evaluating Positions"):
        board = chess.Board(fen)

        # Ask the engine to analyze this position and return its
        # top `top_k` candidate moves (multipv = "multiple principal variations")
        analysis = engine.analyse(
            board,
            chess.engine.Limit(nodes=1),   # nodes=1 -> pure network eval, no search
            multipv=top_k
        )

        # Pull just the first move of each candidate line (its "best next move")
        predicted_moves = []
        for info in analysis:
            if "pv" in info and len(info["pv"]) > 0:
                predicted_moves.append(info["pv"][0].uci())

        if not predicted_moves:
            continue

        # --- Compare human move to engine predictions ---
        if played_move_uci == predicted_moves[0]:
            top_1_correct += 1          # matched the #1 guess exactly
        if played_move_uci in predicted_moves:
            top_k_correct += 1          # matched any of the top-K guesses

        total += 1

    engine.quit()

    return total, top_1_correct, top_k_correct


# ---------------------------------------------------------------------------
# STEP 3: Print a neatly formatted results summary
# ---------------------------------------------------------------------------
def print_results(total, top_1_correct, top_k_correct, top_k):
    top_1_acc = (top_1_correct / total) * 100 if total > 0 else 0
    top_k_acc = (top_k_correct / total) * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print("           MAIA EVALUATION RESULTS           ")
    print("=" * 50)
    print(f"Total Positions Evaluated : {total}")
    print(f"Top-1 Accuracy            : {top_1_acc:6.2f}%  ({top_1_correct}/{total})")
    print(f"Top-{top_k} Accuracy            : {top_k_acc:6.2f}%  ({top_k_correct}/{total})")
    print("=" * 50)


# ---------------------------------------------------------------------------
# STEP 4: Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Filtered dataset not found at: {DATASET_PATH}")
        print("Run prepare_dataset.py first to generate it.")
    else:
        samples = load_sample(DATASET_PATH, max_samples=MAX_SAMPLES, seed=RANDOM_SEED)

        if not samples:
            print("No usable rows found in the filtered dataset.")
        else:
            total, top_1_correct, top_k_correct = evaluate_positions(samples, top_k=TOP_K)
            print_results(total, top_1_correct, top_k_correct, TOP_K)