import os
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from loop import run_loop, full_graph_accuracy
from scorers import random_scorer, uncertainty_scorer, structural_scorer, oracle_voi_scorer
from eval import normalized_advantage

OBSERVE_FRACS = [0.05, 0.10, 0.15, 0.20, 0.30]
SEEDS = [0, 1, 2]
BUDGET = 15

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "uncertainty": uncertainty_scorer,
    "oracle_voi": partial(oracle_voi_scorer, n_sample=40, epochs=25),
}


def main():
    y, A_true, X = make_csbm(seed=0)

    # full_graph_accuracy only depends on the train/test split, which is fixed by `seed`
    # alone (computed before observe_frac has any effect) -- so compute it once per seed
    # and reuse across the whole observe_frac sweep.
    full_acc_by_seed = {}
    for seed in SEEDS:
        env = GraphEnv(y, A_true, X, observe_frac=0.3, seed=seed)
        full_acc_by_seed[seed] = full_graph_accuracy(env, A_true, seed=seed)
    print("full-graph accuracy per seed:", {s: f"{a:.3f}" for s, a in full_acc_by_seed.items()})

    # results[scorer][observe_frac] = list of normalized advantages, one per seed
    results = {name: {frac: [] for frac in OBSERVE_FRACS} for name in SCORERS}

    for frac in OBSERVE_FRACS:
        for name, scorer in SCORERS.items():
            for seed in SEEDS:
                env = GraphEnv(y, A_true, X, observe_frac=frac, seed=seed)
                result = run_loop(env, scorer, budget=BUDGET, seed=seed)
                observed_acc = result["accuracy"][0]
                final_acc = result["accuracy"][-1]
                na = normalized_advantage(final_acc, observed_acc, full_acc_by_seed[seed])
                results[name][frac].append(na)
            vals = [v for v in results[name][frac] if not np.isnan(v)]
            mean_na = np.mean(vals) if vals else float("nan")
            print(f"observe_frac={frac:.2f} {name:>12s}: normalized_advantage mean={mean_na:.3f} "
                  f"(n_valid_seeds={len(vals)}/{len(SEEDS)})")

    print("\nExcess normalized advantage vs random (the 'hump' signal):")
    excess_by_frac = {name: [] for name in SCORERS if name != "random"}
    for frac in OBSERVE_FRACS:
        random_vals = np.array(results["random"][frac])
        row = [f"observe_frac={frac:.2f}"]
        for name in SCORERS:
            if name == "random":
                continue
            strategy_vals = np.array(results[name][frac])
            diffs = strategy_vals - random_vals
            diffs = diffs[~np.isnan(diffs)]
            mean_diff = diffs.mean() if len(diffs) else float("nan")
            excess_by_frac[name].append(mean_diff)
            row.append(f"{name}={mean_diff:+.3f}")
        print("  " + " | ".join(row))

    out_path = os.path.join(os.path.dirname(__file__), "e2_observe_frac_sweep.png")
    plot_sweep(excess_by_frac, out_path)
    print(f"\nSaved plot to {out_path}")


def plot_sweep(excess_by_frac, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for name, values in excess_by_frac.items():
        plt.plot(OBSERVE_FRACS, values, marker="o", label=name)
    plt.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("observe_frac (fraction of true edges revealed upfront)")
    plt.ylabel("Excess normalized advantage vs random")
    plt.title("Signal x headroom: where does task-aware acquisition help?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
