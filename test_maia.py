import chess
import chess.engine
import os

'''BASE = os.path.dirname(os.path.abspath(__file__))

LC0_DIR = os.path.join(BASE, "lc0-v0.32.1-windows-cpu-dnnl")
LC0_PATH = os.path.join(LC0_DIR, "lc0.exe")
WEIGHTS_PATH = os.path.join(BASE, "maia-chess", "weights", "maia-1500.pb.gz")

def test_pipeline():
    print("Initializing lc0 with Maia-1500 weights...")
    print("LC0_PATH:", LC0_PATH)
    print("Exe exists:", os.path.isfile(LC0_PATH))
    print("Weights exist:", os.path.isfile(WEIGHTS_PATH))

    engine = chess.engine.SimpleEngine.popen_uci(
    [LC0_PATH, f"--weights={WEIGHTS_PATH}", "--threads=1", "--backend=blas"],
    cwd=LC0_DIR,)

    board = chess.Board()
    print(f"\nStarting FEN: {board.fen()}")

    result = engine.play(board, chess.engine.Limit(nodes=1))
    print(f"Maia-1500 predicted move: {result.move}")

    engine.quit()

if __name__ == "__main__":
    test_pipeline()'''
from maia2 import model, inference

m = model.from_pretrained(type="rapid", device="cpu")   # first run downloads the weights
prepared = inference.prepare()
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
move_probs, win_prob = inference.inference_each(m, prepared, fen, 1600, 1600)
top = sorted(move_probs.items(), key=lambda x: -x[1])[:3]
print(top, win_prob)