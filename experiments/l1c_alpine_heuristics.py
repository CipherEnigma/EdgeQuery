"""ALPINE's two model-free query strategies (max-deg, page-rank; Chen et al. 2020) vs
random and structural, using the same 50-seed no-training edge simulation as
l1b_link_heuristics.py. ALPINE found both beat random on real networks but trail the
embedding-based strategies; this checks where they land on our locked synthetic setup."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from scorers import random_scorer, structural_scorer, max_degree_scorer, pagerank_scorer
from l1b_link_heuristics import edge_curve, paired

N_SEEDS = 50
CHECKPOINTS = [10, 15, 25, 50, 100, 200]
OBSERVE_FRAC = 0.15

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "max_deg": max_degree_scorer,
    "page_rank": pagerank_scorer,
}


def main():
    y, A, X = make_csbm(seed=0)
    curves = {
        name: np.array([edge_curve(y, A, X, scorer, max(CHECKPOINTS), OBSERVE_FRAC, s) for s in range(N_SEEDS)])
        for name, scorer in SCORERS.items()
    }
    gains = {k: v - v[:, :1] for k, v in curves.items()}

    print(f"ALPINE model-free strategies, {N_SEEDS} seeds, observe_frac={OBSERVE_FRAC}\n")
    for ref in ["random", "structural"]:
        print(f"=== vs {ref} ===")
        print(f"{'name':>11} {'budget':>6} {'gain':>6} {'ref':>6} {'diff':>7} {'t':>6} {'p':>8}")
        for name in ["max_deg", "page_rank"]:
            for b in CHECKPOINTS:
                md, _, t, p = paired(gains[name][:, b].astype(float), gains[ref][:, b].astype(float))
                flag = " *" if p < 0.05 else ""
                print(f"{name:>11} {b:>6} {gains[name][:, b].mean():>6.2f} {gains[ref][:, b].mean():>6.2f} "
                      f"{md:>+7.2f} {t:>6.2f} {p:>8.4f}{flag}")
        print()


if __name__ == "__main__":
    main()
