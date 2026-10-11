"""Cheap check: do Adamic-Adar or Jaccard beat plain structural_scorer?

All three scorers here are model-free (no GCN training), so this simulates edge
acquisition directly -- same technique diag_foundation.py used to get 50 seeds cheaply.
Paired against random (the Phase 1 comparison) AND against structural (the real
question: does a better graph-only heuristic beat what we already confirmed works).
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from scorers import random_scorer, structural_scorer, adamic_adar_scorer, jaccard_scorer

N_SEEDS = 50
CHECKPOINTS = [10, 15, 25, 50, 100, 200]
OBSERVE_FRAC = 0.15

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "adamic_adar": adamic_adar_scorer,
    "jaccard": jaccard_scorer,
}


def edge_curve(y, A, X, scorer, budget, obs, seed):
    env = GraphEnv(y, A, X, observe_frac=obs, seed=seed)
    curve = [int(env.A_obs.sum() / 2)]
    for _ in range(budget):
        cands = env.candidates()
        s = scorer(env, cands, None, None, None)
        tied = np.flatnonzero(s == s.max())
        env.reveal(cands[int(env.rng.choice(tied))])
        curve.append(int(env.A_obs.sum() / 2))
    return np.array(curve)


def paired(a, b):
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return d.mean(), d.std(ddof=1), t, p


def main():
    y, A, X = make_csbm(seed=0)
    curves = {
        name: np.array([edge_curve(y, A, X, scorer, max(CHECKPOINTS), OBSERVE_FRAC, s) for s in range(N_SEEDS)])
        for name, scorer in SCORERS.items()
    }
    gains = {k: v - v[:, :1] for k, v in curves.items()}

    print(f"Model-free scorer comparison, {N_SEEDS} seeds, observe_frac={OBSERVE_FRAC}\n")
    for ref in ["random", "structural"]:
        print(f"=== vs {ref} ===")
        print(f"{'name':>13} {'budget':>6} {'mean gain':>10} {'ref gain':>9} {'diff':>7} {'sd':>6} {'t':>6} {'p':>8}")
        for name in SCORERS:
            if name == ref:
                continue
            for b in CHECKPOINTS:
                md, sd, t, p = paired(gains[name][:, b].astype(float), gains[ref][:, b].astype(float))
                flag = " *" if p < 0.05 else ""
                print(f"{name:>13} {b:>6} {gains[name][:, b].mean():>10.2f} {gains[ref][:, b].mean():>9.2f} "
                      f"{md:>+7.2f} {sd:>6.2f} {t:>6.2f} {p:>8.4f}{flag}")
        print()


if __name__ == "__main__":
    main()
