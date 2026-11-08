"""
Prepare Maia Testing Dataset
=============================
This script does three things, in order:

  1. DECOMPRESS: takes "maia-chess-testing-set.csv.bz2" and extracts the
     plain CSV file next to it, so pandas/your evaluator can read it.

  2. INSPECT: prints out the column names and the first few rows of the
     extracted CSV, so you can confirm the columns are named what your
     evaluator script expects ("board"/"fen" and "move"/"played_move").
     If they're named something else, you'll see that here BEFORE
     wasting time running the full evaluation.

  3. FILTER: uses pandas to read through the (potentially huge) raw CSV
     in chunks, keeps only rows where the player about to move is close
     to your target Elo AND the move happens after the opening (skips
     memorized opening theory), and writes those matching rows out to
     a much smaller, separate CSV file. Your evaluator script then just
     reads this pre-filtered file directly — no need to re-scan or
     re-filter the full raw dataset every time you run an evaluation.
"""

import bz2         # lets Python read .bz2 compressed files
import shutil      # used to efficiently copy data from one file to another
import os          # for building file paths and checking if files exist
import csv         # used to peek at the CSV's header + first rows
import pandas as pd  # used for the chunked filtering step


# ---------------------------------------------------------------------------
# STEP 0: File paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

# The compressed file you downloaded (must be in the same folder as this
# script, or give a full path here instead)
COMPRESSED_FILE = os.path.join(BASE, "maia-chess-testing-set.csv.bz2")

# Where the decompressed CSV will be written
DECOMPRESSED_FILE = os.path.join(BASE, "maia-chess-testing-set.csv")

# Where the filtered (Elo-matched, opening-skipped) CSV will be written —
# this is the file your evaluator script should read from
FILTERED_FILE = os.path.join(BASE, "maia-chess-testing-set-filtered.csv")

# Filtering settings — should match the Maia model you plan to test
TARGET_ELO = 1500   # matches the dataset's rounded_active_elo column
                     # (each player's rating rounded to the nearest 100)
MIN_PLY = 10         # skip the first 10 ply (half-moves) of every game,
                     # since those are usually memorized opening theory
CHUNK_SIZE = 100_000  # how many rows pandas reads into memory at a time —
                      # keeps memory usage low even on huge raw files


# ---------------------------------------------------------------------------
# STEP 1: Decompress the .bz2 file into a plain .csv file
# ---------------------------------------------------------------------------
def decompress_bz2(compressed_path, output_path):
    """
    Reads a .bz2 compressed file in small chunks and writes the
    decompressed contents to a new plain file. Skips the work entirely
    if the decompressed file already exists.
    """
    if os.path.exists(output_path):
        print(f"Already decompressed — found existing file:\n  {output_path}")
        return

    if not os.path.exists(compressed_path):
        print(f"ERROR: Could not find compressed file at:\n  {compressed_path}")
        print("Make sure the .bz2 file is in the same folder as this script,")
        print("or edit COMPRESSED_FILE above to point to its actual location.")
        return

    print(f"Decompressing:\n  {compressed_path}\ninto:\n  {output_path}")

    # bz2.open lets us read the compressed file as if it were a normal file
    with bz2.open(compressed_path, "rb") as f_in:
        # shutil.copyfileobj streams data in chunks instead of loading the
        # whole file into memory at once — important since these datasets
        # can be large
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Decompression complete.")


# ---------------------------------------------------------------------------
# STEP 2: Inspect the resulting CSV's structure
# ---------------------------------------------------------------------------
def inspect_csv(csv_path, num_preview_rows=5):
    """
    Opens the CSV and prints:
      - the column headers (so you can check they match what your
        evaluator script expects: "board"/"fen" and "move"/"played_move")
      - the first few rows of actual data, so you can see the real
        values (e.g. is "board" a full FEN string? is "move" in UCI
        format like "e2e4"?)
    """
    if not os.path.exists(csv_path):
        print(f"Cannot inspect — file not found:\n  {csv_path}")
        return

    print("\n" + "=" * 60)
    print("           CSV STRUCTURE PREVIEW")
    print("=" * 60)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Print the column names found in the header row
        print(f"Columns found: {reader.fieldnames}")

        # Print the expected columns for comparison
        expected = ["board (or fen)", "move (or played_move)"]
        print(f"Columns your evaluator script expects: {expected}")

        print("\nFirst few rows of data:")
        print("-" * 60)

        # Print a handful of rows so you can see actual values
        for i, row in enumerate(reader):
            if i >= num_preview_rows:
                break
            print(row)

    print("=" * 60)
    print("\nIf 'Columns found' does NOT include something matching")
    print("'board'/'fen' and 'move'/'played_move', you'll need to update")
    print("the row.get(...) lines in your evaluator script to use the")
    print("actual column names shown above.")


# ---------------------------------------------------------------------------
# STEP 3: Filter the dataset down to matching Elo + post-opening positions
# ---------------------------------------------------------------------------
def filter_dataset(input_path, output_path, target_elo, min_ply, chunk_size=100_000):
    """
    Reads `input_path` in chunks (so we never load the whole huge raw file
    into memory at once), keeps only rows where:
      - rounded_active_elo == target_elo  (player about to move is near
        your target rating)
      - move_ply >= min_ply               (skips memorized opening moves)

    and writes the matching rows out to `output_path` as a new, much
    smaller CSV file.
    """
    if os.path.exists(output_path):
        print(f"Filtered file already exists — skipping filter step:\n  {output_path}")
        return

    if not os.path.exists(input_path):
        print(f"Cannot filter — input file not found:\n  {input_path}")
        return

    print(f"\nFiltering dataset for Elo == {target_elo} and move_ply >= {min_ply}...")
    print(f"Reading in chunks of {chunk_size:,} rows to keep memory usage low.")

    total_rows_seen = 0
    total_rows_kept = 0
    first_chunk = True  # tracks whether we still need to write the header row

    # chunksize makes pandas return an iterator of DataFrames instead of
    # loading the entire file at once — essential for very large CSVs
    reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        total_rows_seen += len(chunk)

        # Keep only rows matching both conditions.
        # rounded_active_elo may be read as float (e.g. 1500.0) if the
        # column has any missing values, so we cast to int for a clean
        # comparison against target_elo.
        matches = chunk[
            (chunk["rounded_active_elo"].fillna(-1).astype(int) == target_elo)
            & (chunk["move_ply"].fillna(-1).astype(int) >= min_ply)
        ]

        total_rows_kept += len(matches)

        # Append this chunk's matches to the output file. We write the
        # header only once (on the very first chunk that produces output),
        # and use mode="a" (append) for every write after that so each
        # chunk's rows get added to the same growing file.
        matches.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        if len(matches) > 0:
            first_chunk = False

    print(f"Scanned {total_rows_seen:,} total rows.")
    print(f"Kept {total_rows_kept:,} rows matching Elo {target_elo} and "
          f"move_ply >= {min_ply}.")
    print(f"Filtered dataset written to:\n  {output_path}")


# ---------------------------------------------------------------------------
# STEP 4: Run all steps in order
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    decompress_bz2(COMPRESSED_FILE, DECOMPRESSED_FILE)
    inspect_csv(DECOMPRESSED_FILE)
    filter_dataset(
        DECOMPRESSED_FILE,
        FILTERED_FILE,
        target_elo=TARGET_ELO,
        min_ply=MIN_PLY,
        chunk_size=CHUNK_SIZE,
    )