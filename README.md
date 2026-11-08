# Personalized Maia: Few-Shot Alignment to Individual Chess Players

Course project for CSC 457 (Data Analysis, Decision-Making and Data Visualization), University of South Dakota, Fall 2026.

**Question.** Maia models predict the move a *typical* player of a given rating would make. Can fine-tuning Maia-2 on a small number of games from **one** player make it predict that player's moves better, and how many games does it take?

## Summary of results

Setup: one public Lichess rapid player (about 1600), 399 games (11,446 positions), split chronologically by game. Test = newest 60 games (1,744 positions), never used for training or model selection. Metrics: Top-1, Top-3 and NLL (negative log-likelihood of the played move), with 95% bootstrap intervals that resample games.

| Model | Top-1 (%) | Top-3 (%) | NLL |
|---|---|---|---|
| Maia-1500 (lc0), zero-shot | 50.34 | 79.30 | n/a |
| Maia-2 rapid, zero-shot | 51.43 | 80.50 | 1.443 |
| Maia-2 fine-tuned on 100 games of the player (3-seed mean) | 52.66 | 81.35 | 1.413 |
| Maia-2 fine-tuned on 200 games of the player (3-seed mean) | 52.66 | 81.80 | 1.402 |

* NLL improves steadily with more games (about -0.011 at 25 games to -0.040 at 200 games; paired intervals exclude zero from 25 games on). Top-1 and Top-3 gains are small (about 1 point) and less consistent.
* Fine-tuning needed BatchNorm statistics frozen and dropout disabled; without that, no gain appeared.
* Control experiment: fine-tuning on a *different* player's games did not help (early stopping kept the pretrained model) and, with a fixed number of epochs, made predictions of the original player worse. This supports, but does not prove, that part of the gain is player-specific.

Limitations: one player, 60 test games, one control player, opening moves (first 10 plies) excluded.

## Repository layout

Scripts use paths relative to their own folder and its parent, so keep this layout (the heavy folders are not in the repo, see `.gitignore`):

```
data/                          <- repository root
  prepare_dataset.py           decompress and filter the Maia 1500-Elo test set
  collect_player_games.py      download a player's games from the Lichess API, build positions
  collect_control_player.py    same for the control player, builds nested training sets
  split_games.py               chronological split by game: test / validation / training sets
  add_elo.py                   add own and opponent rating columns
  evaluate_maia.py             Maia-1500 (lc0) on the general test set
  evaluate_maia_single_game.py Maia-1500 (lc0) on one game
  Finetune Maia/
    finetune_maia2.py          fine-tune Maia-2 and evaluate (options: --n_games, --seed, --lr,
                               --freeze_bn, --no_dropout, --fixed_epochs, --train_dir, --tag, --save_model)
    compare_runs.py            paired comparison of two runs
    compare_control.py         own games vs control player
    plot_learning_curve.py     learning-curve figures and tables
    analyze_one_game.py        compare original / trained models move by move on one test game
  player baseline/
    evaluate_player_baseline.py  Maia-1500 / 1600 (lc0) baselines on the player's test games
    evaluate_maia2.py            zero-shot Maia-2 baseline
```

## Setup

1. Python 3.12 and a virtual environment: `python -m venv .venv`, then activate it.
2. `python -m pip install -r requirements.txt` (install a CUDA build of PyTorch if you have an NVIDIA GPU).
3. For the lc0 baselines only: download Leela Chess Zero (v0.32.1, CPU build) and the Maia-1500 / Maia-1600 weights from the [Maia Chess repository](https://github.com/CSSLab/maia-chess). Maia-2 weights download automatically on first use.

## Reproducing the experiments

```
python collect_player_games.py                 # set USERNAME inside the script first
python split_games.py
python add_elo.py
python collect_control_player.py               # set USERNAME to the control player

cd "Finetune Maia"
python finetune_maia2.py --n_games 0                                   # zero-shot baseline
python finetune_maia2.py --n_games 100 --seed 0 --lr 2e-5 --freeze_bn --no_dropout
python compare_runs.py finetune_results/test_preds_n0_s0.csv finetune_results/test_preds_n100_s0_lr2e-05_fbn_nodo.csv
python plot_learning_curve.py
python compare_control.py control_Optemm       # after running the control loop with --tag control_Optemm
```

Data are public Lichess games downloaded through the Lichess API. Set the player usernames in the collection scripts; they are not part of this repository's results.

## Credits and references

* Maia: McIlroy-Young et al., KDD 2020. Maia-2: Tang et al., NeurIPS 2024. Individual models: McIlroy-Young et al., KDD 2022. Code and weights: https://github.com/CSSLab/maia-chess and https://github.com/CSSLab/maia2 (see their licenses; weights and data are not redistributed here).
* Leela Chess Zero (lc0): https://lczero.org
* Game data: https://lichess.org (public games).
