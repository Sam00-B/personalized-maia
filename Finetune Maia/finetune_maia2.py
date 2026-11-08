"""
Fine-tune Maia-2 (rapid) on one player's games and evaluate on held-out games.

Usage (from the folder that holds this script; ROOT = its parent "data" folder):

    python finetune_maia2.py --n_games 0                 # zero-shot baseline (no training)
    python finetune_maia2.py --n_games 100 --seed 0      # fine-tune on train_100_elo.csv
    python finetune_maia2.py --n_games 100 --lr 5e-5     # try another learning rate

Expected files under ROOT:
    training set/train_<N>_elo.csv
    val set/val_elo.csv
    test set/test_elo.csv
Each needs columns: game_id, fen, move, elo_self, elo_oppo.

Rules this script follows:
  * Train only on the training file.
  * Pick the best epoch with the VALIDATION file (lowest validation NLL; epoch 0 =
    the untouched pretrained model, so training can never make the kept model
    worse on validation than the baseline).
  * Report the TEST file once at the end.
"""

import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from maia2 import model as maia_model
from maia2.inference import preprocessing
from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT_DIR = os.path.join(BASE, "finetune_results")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset(df, all_moves_dict, elo_dict):
    """Turn rows (fen, move, elo_self, elo_oppo) into tensors Maia-2 understands.

    Maia-2 always sees the board from White's point of view, so for Black-to-move
    positions the board is mirrored inside `preprocessing`; the played move must
    be mirrored the same way to find its class index.
    """
    boards, e_self, e_oppo, legal, target = [], [], [], [], []
    keep_rows = []
    skipped = 0
    for i, (fen, move, es, eo) in enumerate(
        zip(df["fen"], df["move"], df["elo_self"], df["elo_oppo"])
    ):
        b, s_cat, o_cat, lm = preprocessing(fen, int(es), int(eo), elo_dict, all_moves_dict)
        m = mirror_move(move) if fen.split(" ")[1] == "b" else move
        idx = all_moves_dict.get(m)
        if idx is None or lm[idx] == 0:
            skipped += 1
            continue
        boards.append(b)
        e_self.append(s_cat)
        e_oppo.append(o_cat)
        legal.append(lm)
        target.append(idx)
        keep_rows.append(i)

    if skipped:
        print(f"  (skipped {skipped} rows whose move was not legal / not in the move list)")

    ds = TensorDataset(
        torch.stack(boards),
        torch.tensor(e_self, dtype=torch.long),
        torch.tensor(e_oppo, dtype=torch.long),
        torch.stack(legal),
        torch.tensor(target, dtype=torch.long),
    )
    meta = df.iloc[keep_rows][["game_id", "fen", "move"]].reset_index(drop=True)
    return ds, meta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, ds, device, batch_size=256):
    """Returns per-position arrays: top1_hit, top3_hit, nll (unrounded probabilities)."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    top1, top3, nll = [], [], []
    for boards, es, eo, legal, tgt in loader:
        boards, es, eo = boards.to(device), es.to(device), eo.to(device)
        legal, tgt = legal.to(device), tgt.to(device)
        logits, _, _ = model(boards, es, eo)
        logits = logits.masked_fill(legal == 0, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        nll.append(-logp.gather(1, tgt[:, None]).squeeze(1))
        top = logits.topk(3, dim=-1).indices
        top1.append(top[:, 0] == tgt)
        top3.append((top == tgt[:, None]).any(dim=1))
    return (
        torch.cat(top1).cpu().numpy(),
        torch.cat(top3).cpu().numpy(),
        torch.cat(nll).cpu().numpy(),
    )


def bootstrap_ci(values, game_ids, n_boot=2000, seed=42):
    """95% CI of the mean, resampling whole GAMES (positions in a game are correlated)."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"v": values.astype(float), "g": game_ids}).groupby("g")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return np.percentile(means, [2.5, 97.5])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def set_train_mode(model, args):
    """model.train(), but optionally keep BatchNorm statistics and dropout fixed.

    Freezing weights (requires_grad=False) does NOT stop BatchNorm from updating
    its running mean/variance in train mode, which silently shifts the model on a
    small, player-specific dataset. --freeze_bn keeps those statistics fixed;
    --no_dropout removes the dropout noise while fine-tuning.
    """
    model.train()
    if args.freeze_bn:
        for mod in model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
    if args.no_dropout:
        for mod in model.modules():
            if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d)):
                mod.eval()

def fine_tune(model, train_ds, val_ds, args, device):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, generator=gen)

    n_bn = sum(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in model.modules())
    n_do = sum(isinstance(x, (torch.nn.Dropout, torch.nn.Dropout2d)) for x in model.modules())
    n_tr = sum(p.numel() for p in params)
    print(f"BatchNorm modules: {n_bn} | Dropout modules: {n_do} | trainable params: {n_tr:,}")
    _, _, v_nll = evaluate(model, val_ds, device)
    best_val = float(v_nll.mean())
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, bad = 0, 0
    print(f"epoch  0 (pretrained)  val NLL {best_val:.4f}")

    n_epochs = args.fixed_epochs if args.fixed_epochs > 0 else args.epochs
    for epoch in range(1, n_epochs + 1):
        set_train_mode(model, args)
        total, n = 0.0, 0
        for boards, es, eo, legal, tgt in loader:
            boards, es, eo = boards.to(device), es.to(device), eo.to(device)
            legal, tgt = legal.to(device), tgt.to(device)
            logits, _, _ = model(boards, es, eo)
            logits = logits.masked_fill(legal == 0, float("-inf"))
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * len(tgt)
            n += len(tgt)

        _, _, v_nll = evaluate(model, val_ds, device)
        val = float(v_nll.mean())
        flag = ""
        if val < best_val - 1e-4:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            flag = "  <- best"
        else:
            bad += 1
        print(f"epoch {epoch:2d}  train loss {total / n:.4f}  val NLL {val:.4f}{flag}")
        if bad >= args.patience and args.fixed_epochs == 0:
            print("early stopping")
            break

    if args.fixed_epochs > 0:
        return n_epochs, val          # keep the final model, no selection
    model.load_state_dict(best_state)
    return best_epoch, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_games", type=int, required=True, help="0 = zero-shot (no training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="freeze the CNN + patch embedding, train only the rest")
    ap.add_argument("--fixed_epochs", type=int, default=0,
                    help="if > 0: train exactly this many epochs and keep the FINAL model (no early stopping or "
                         "validation-based selection). Used for the stricter control comparison.")
    ap.add_argument("--save_model", action="store_true",
                    help="save the fine-tuned weights to finetune_results/models/ (about 90 MB per run)")
    ap.add_argument("--freeze_bn", action="store_true",
                    help="keep BatchNorm running statistics fixed while fine-tuning")
    ap.add_argument("--no_dropout", action="store_true",
                    help="disable dropout while fine-tuning")
    ap.add_argument("--train_dir", default="training set",
                    help="folder (under ROOT) holding train_<N>_elo.csv; use a control player folder for the control experiment")
    ap.add_argument("--tag", default="", help="label for the run, e.g. control_Optemm (keeps result files separate)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    all_moves = get_all_possible_moves()
    all_moves_dict = {m: i for i, m in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    def load(path):
        df = pd.read_csv(path).dropna(subset=["fen", "move", "elo_self", "elo_oppo"])
        return df.reset_index(drop=True)

    test_df = load(os.path.join(ROOT, "test set", "test_elo.csv"))
    test_ds, test_meta = build_dataset(test_df, all_moves_dict, elo_dict)

    m = maia_model.from_pretrained(type="rapid", device=device)

    best_epoch, best_val = 0, float("nan")
    if args.n_games > 0:
        train_df = load(os.path.join(ROOT, args.train_dir, f"train_{args.n_games}_elo.csv"))
        val_df = load(os.path.join(ROOT, "val set", "val_elo.csv"))
        train_ds, _ = build_dataset(train_df, all_moves_dict, elo_dict)
        val_ds, _ = build_dataset(val_df, all_moves_dict, elo_dict)
        print(f"train positions {len(train_ds)} | val positions {len(val_ds)} | "
              f"test positions {len(test_ds)}")

        if args.freeze_backbone:
            for mod in (m.chess_cnn, m.to_patch_embedding):
                for p in mod.parameters():
                    p.requires_grad = False

        best_epoch, best_val = fine_tune(m, train_ds, val_ds, args, device)
    else:
        print(f"zero-shot evaluation | test positions {len(test_ds)}")

    # ---- final evaluation on the held-out test games (done once) ----
    t1, t3, nll = evaluate(m, test_ds, device)
    gids = test_meta["game_id"].to_numpy()
    row = {
        "n_games": args.n_games, "seed": args.seed, "lr": args.lr,
        "freeze_backbone": args.freeze_backbone, "freeze_bn": args.freeze_bn,
        "no_dropout": args.no_dropout,
        "best_epoch": best_epoch, "val_nll": best_val,
    }
    for name, vals, scale in [("top1", t1, 100), ("top3", t3, 100), ("nll", nll, 1)]:
        lo, hi = bootstrap_ci(vals, gids)
        row[name] = vals.mean() * scale
        row[name + "_lo"], row[name + "_hi"] = lo * scale, hi * scale
        print(f"TEST {name:5s}: {row[name]:.2f}  (95% CI {lo * scale:.2f} to {hi * scale:.2f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    preds = test_meta.copy()
    preds["top1_hit"], preds["top3_hit"], preds["nll"] = t1, t3, nll
    tag = f"_lr{args.lr:g}" + ("_fbn" if args.freeze_bn else "") + ("_nodo" if args.no_dropout else "") \
        + ("_frz" if args.freeze_backbone else "") + (f"_fx{args.fixed_epochs}" if args.fixed_epochs else "")
    name = f"test_preds_n{args.n_games}_s{args.seed}" + (tag if args.n_games > 0 else "") + (f"_{args.tag}" if (args.tag and args.n_games > 0) else "") + ".csv"
    preds.to_csv(os.path.join(OUT_DIR, name), index=False)
    print("per-position results:", name)

    if args.save_model:
        if args.n_games == 0:
            print("--save_model ignored for the zero-shot run (nothing was trained)")
        else:
            who = "guilleo" if args.train_dir == "training set" else os.path.basename(os.path.normpath(args.train_dir))
            mdir = os.path.join(OUT_DIR, "models")
            os.makedirs(mdir, exist_ok=True)
            mpath = os.path.join(mdir, f"{who}_{args.n_games}games_s{args.seed}{tag}.pt")
            torch.save(m.state_dict(), mpath)
            print("saved model weights:", mpath)

    summary_name = "learning_curve" + (f"_fx{args.fixed_epochs}" if args.fixed_epochs else "") \
        + (f"_{args.tag}" if args.tag else "") + ".csv"
    summary = os.path.join(OUT_DIR, summary_name)
    pd.DataFrame([row]).to_csv(summary, mode="a", header=not os.path.exists(summary), index=False)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()