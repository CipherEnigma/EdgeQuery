import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from loop import run_loop, full_graph_accuracy
from scorers import random_scorer, structural_scorer, uncertainty_scorer
from eval import auc, mean_and_se

SEEDS = [0, 1, 2]
BUDGET = 200
OBSERVE_FRAC = 0.1  # plenty of headroom; this is where structural's hit-rate advantage was found

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "uncertainty": uncertainty_scorer,
}


def main():
    y, A_true, X = make_csbm(seed=0)

    acc_results = {}    # name -> array (n_seeds, budget+1)
    edges_results = {}  # name -> array (n_seeds, budget+1)
    full_acc_by_seed = {}

    for name, scorer in SCORERS.items():
        acc_curves = []
        edge_curves = []
        for seed in SEEDS:
            env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=seed)
            if seed not in full_acc_by_seed:
                full_acc_by_seed[seed] = full_graph_accuracy(env, A_true, seed=seed)
            result = run_loop(env, scorer, budget=BUDGET, seed=seed)
            acc_curves.append(result["accuracy"])
            edge_curves.append(result["n_real_edges"])
        acc_results[name] = np.array(acc_curves)
        edges_results[name] = np.array(edge_curves)

        final_acc = acc_results[name][:, -1]
        final_edges = edges_results[name][:, -1]
        aucs = [auc(c) for c in acc_results[name]]
        print(f"{name}: final real edges mean={final_edges.mean():.1f} (per seed: {final_edges.tolist()}), "
              f"final acc mean={final_acc.mean():.3f}, AUC mean={np.mean(aucs):.2f}")

    print("\nfull-graph accuracy ceiling per seed:", {s: f"{a:.3f}" for s, a in full_acc_by_seed.items()})

    print("\nAUC gap vs random baseline:")
    random_auc = np.mean([auc(c) for c in acc_results["random"]])
    for name in SCORERS:
        if name == "random":
            continue
        gap = np.mean([auc(c) for c in acc_results[name]]) - random_auc
        print(f"  {name}: {gap:+.2f}")

    print("\nFinal real-edge count gap vs random baseline (the higher-resolution metric):")
    random_final_edges = edges_results["random"][:, -1].mean()
    for name in SCORERS:
        if name == "random":
            continue
        gap = edges_results[name][:, -1].mean() - random_final_edges
        print(f"  {name}: {gap:+.1f} more real edges than random")

    out_path = os.path.join(os.path.dirname(__file__), "e3_big_budget.png")
    plot_dual(acc_results, edges_results, out_path)
    print(f"\nSaved plot to {out_path}")


def plot_dual(acc_results, edges_results, save_path):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, curves in acc_results.items():
        mean, se = mean_and_se(curves)
        x = np.arange(len(mean))
        ax1.plot(x, mean, label=name)
        ax1.fill_between(x, mean - se, mean + se, alpha=0.2)
    ax1.set_xlabel("Loop step (queries spent)")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Accuracy vs budget spent")
    ax1.legend()

    for name, curves in edges_results.items():
        mean, se = mean_and_se(curves)
        x = np.arange(len(mean))
        ax2.plot(x, mean, label=name)
        ax2.fill_between(x, mean - se, mean + se, alpha=0.2)
    ax2.set_xlabel("Loop step (queries spent)")
    ax2.set_ylabel("Real edges in observed graph")
    ax2.set_title("Real edges acquired vs budget spent")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
