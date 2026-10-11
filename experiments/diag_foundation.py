"""Diagnose why Phase 1 found no significant structural-vs-random difference.

random_scorer and structural_scorer never use the model, so their edge-acquisition
sequence is identical with or without GCN training. That lets us simulate edge
curves cheaply, with many more seeds than a full run_loop could afford, and read the
high-resolution real-edge metric at every budget checkpoint.

Test 1 (regression to the mean): how often does a 3-seed test show a strong-looking
          structural win, when the full-power effect is what it is?
Test 2 (density retune): current config vs original config, same protocol.
Test 3 (budget dilution): paired tests at budgets 10..200 from the same curves.
"""
import os
import sys
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from scorers import random_scorer, structural_scorer

N_SEEDS = 50
BUDGET = 200
CHECKPOINTS = [10, 15, 25, 50, 100, 200]

CONFIGS = {
    "current (p_in=0.13, p_out=0.006, obs=0.15)": dict(p_in=0.13, p_out=0.006, obs=0.15),
    "original (p_in=0.11, p_out=0.005, obs=0.10)": dict(p_in=0.11, p_out=0.005, obs=0.10),
}

PHASE1_FINALS = {
    "random": [32, 32, 34, 38, 35, 33, 33, 31, 33, 33],
    "structural": [34, 36, 32, 32, 38, 32, 31, 39, 35, 35],
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
    all_gains = {}
    for label, cfg in CONFIGS.items():
        y, A, X = make_csbm(p_in=cfg["p_in"], p_out=cfg["p_out"], seed=0)
        curves = {}
        for name, scorer in [("random", random_scorer), ("structural", structural_scorer)]:
            curves[name] = np.array([edge_curve(y, A, X, scorer, BUDGET, cfg["obs"], s) for s in range(N_SEEDS)])
        gains = {k: v - v[:, :1] for k, v in curves.items()}
        all_gains[label] = gains

        n_true = int(A.sum() / 2)
        start = curves["random"][:, 0].mean()
        pool = 90 * 89 // 2 - start
        base_rate = (n_true - start) / pool
        print(f"\n=== {label} ===")
        print(f"true edges={n_true}, revealed at start={start:.0f}, random base rate={100*base_rate:.2f}% per query")

        if label.startswith("current"):
            match = all(
                curves[name][:10, -1].tolist() == PHASE1_FINALS[name] for name in PHASE1_FINALS
            )
            print(f"reproduces Phase 1 seeds 0-9 exactly: {match}")

        print(f"{'budget':>6} {'random gain':>12} {'struct gain':>12} {'diff':>7} {'sd':>6} {'t':>6} {'p':>8}")
        for b in CHECKPOINTS:
            r, s = gains["random"][:, b], gains["structural"][:, b]
            md, sd, t, p = paired(s.astype(float), r.astype(float))
            flag = " *" if p < 0.05 else ""
            print(f"{b:>6} {r.mean():>12.2f} {s.mean():>12.2f} {md:>+7.2f} {sd:>6.2f} {t:>6.2f} {p:>8.4f}{flag}")

    print("\n=== Test 1: regression to the mean ===")
    print("Earlier 'strong' results, re-tested with a paired t-test on their own per-seed data:")
    for desc, s, r in [
        ("E2 diag, old config, budget=15, 5 seeds (hits)", [3, 2, 2, 3, 1], [0, 0, 2, 1, 1]),
        ("E3, old config, budget=200, 3 seeds (final edges)", [25, 23, 23], [23, 18, 21]),
    ]:
        md, sd, t, p = paired(np.array(s, float), np.array(r, float))
        print(f"  {desc}: diff={md:+.2f}, t={t:.2f}, p={p:.4f}")

    for label, gains in all_gains.items():
        for b in [15, 200]:
            d = (gains["structural"][:, b] - gains["random"][:, b]).astype(float)
            subset_means = np.array([d[list(c)].mean() for c in combinations(range(N_SEEDS), 3)])
            full = d.mean()
            print(f"  {label}, budget={b}: full {N_SEEDS}-seed diff={full:+.2f}; "
                  f"3-seed subsets: {100*(subset_means >= 2).mean():.0f}% show diff>=+2, "
                  f"{100*(subset_means <= -2).mean():.0f}% show diff<=-2, "
                  f"range [{subset_means.min():+.1f}, {subset_means.max():+.1f}]")


if __name__ == "__main__":
    main()
