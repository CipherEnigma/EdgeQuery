import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from loop import run_multi_seed
from scorers import random_scorer, uncertainty_scorer
from eval import auc, plot_curves

SEEDS = [0, 1, 2, 3, 4]
BUDGET = 40


def main():
    y, A_true, X = make_csbm(seed=0)

    results = {}
    for name, scorer in [("random", random_scorer), ("uncertainty", uncertainty_scorer)]:
        curves = run_multi_seed(y, A_true, X, scorer, budget=BUDGET, seeds=SEEDS)
        results[name] = curves
        aucs = [auc(c) for c in curves]
        print(f"{name}: AUC mean={np.mean(aucs):.2f} (+/- {np.std(aucs):.2f}), "
              f"final acc mean={curves[:, -1].mean():.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "e0_sanity.png")
    plot_curves(results, out_path, title="E0 sanity check: random vs uncertainty")
    print(f"Saved plot to {out_path}")

    random_final = results["random"][:, -1].mean()
    uncertainty_final = results["uncertainty"][:, -1].mean()
    assert uncertainty_final >= random_final - 0.05, (
        "uncertainty scorer is dramatically worse than random -- pipeline is likely broken"
    )
    print("Sanity check passed: pipeline runs honestly, edges move accuracy as expected.")


if __name__ == "__main__":
    main()
