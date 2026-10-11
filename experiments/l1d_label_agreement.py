"""Fast check: does label_agreement_scorer find real edges faster than structural?
Model-free (uses only train labels + observed graph), so simulated cheaply like
l1b/l1c. If this clears the bar, it goes to a real 30-seed accuracy confirmatory run.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from scorers import random_scorer, structural_scorer, label_agreement_scorer
from l1b_link_heuristics import edge_curve, paired

N_SEEDS = 50
CHECKPOINTS = [10, 15, 25, 50, 100, 200]
OBSERVE_FRAC = 0.15

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "label_agree": label_agreement_scorer,
}


def main():
    y, A, X = make_csbm(seed=0)
    curves = {
        name: np.array([edge_curve(y, A, X, scorer, max(CHECKPOINTS), OBSERVE_FRAC, s) for s in range(N_SEEDS)])
        for name, scorer in SCORERS.items()
    }
    gains = {k: v - v[:, :1] for k, v in curves.items()}

    print(f"label_agreement_scorer check, {N_SEEDS} seeds, observe_frac={OBSERVE_FRAC}\n")
    for ref in ["random", "structural"]:
        print(f"=== vs {ref} ===")
        print(f"{'budget':>6} {'gain':>6} {'ref':>6} {'diff':>7} {'t':>6} {'p':>8}")
        for b in CHECKPOINTS:
            md, _, t, p = paired(gains["label_agree"][:, b].astype(float), gains[ref][:, b].astype(float))
            flag = " *" if p < 0.05 else ""
            print(f"{b:>6} {gains['label_agree'][:, b].mean():>6.2f} {gains[ref][:, b].mean():>6.2f} "
                  f"{md:>+7.2f} {t:>6.2f} {p:>8.4f}{flag}")
        print()


if __name__ == "__main__":
    main()
