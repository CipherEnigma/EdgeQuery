import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from loop import run_multi_seed
from scorers import random_scorer, uncertainty_scorer, structural_scorer, oracle_voi_scorer
from eval import auc, plot_curves

SEEDS = [0, 1, 2]
BUDGET = 20


def main():
    y, A_true, X = make_csbm(seed=0)

    scorers = {
        "random": random_scorer,
        "structural": structural_scorer,
        "uncertainty": uncertainty_scorer,
        "oracle_voi": partial(oracle_voi_scorer, n_sample=15, epochs=25),
    }

    results = {}
    aucs = {}
    for name, scorer in scorers.items():
        curves = run_multi_seed(y, A_true, X, scorer, budget=BUDGET, seeds=SEEDS)
        results[name] = curves
        aucs[name] = [auc(c) for c in curves]
        print(f"{name}: AUC mean={np.mean(aucs[name]):.2f} (+/- {np.std(aucs[name]):.2f}), "
              f"final acc mean={curves[:, -1].mean():.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "e1_killtest.png")
    plot_curves(results, out_path, title="E1 kill test: all strategies")
    print(f"Saved plot to {out_path}")

    random_auc = np.mean(aucs["random"])
    print("\nAUC gap vs random baseline:")
    winner, best_gap = None, -np.inf
    for name in scorers:
        if name == "random":
            continue
        gap = np.mean(aucs[name]) - random_auc
        print(f"  {name}: {gap:+.2f}")
        if gap > best_gap:
            best_gap, winner = gap, name

    if best_gap > 0:
        print(f"\nWinner: {winner} beats random by {best_gap:+.2f} AUC -> PURSUE")
    else:
        print("\nNo strategy beats random -> PIVOT")


if __name__ == "__main__":
    main()
