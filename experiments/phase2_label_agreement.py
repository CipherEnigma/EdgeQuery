"""Phase 2 candidate: label_agreement_scorer vs structural vs random, on actual
downstream accuracy (not just edge hit-rate). 30 seeds, budget=200, paired t-tests --
same discipline as phase1_confirmatory.py. Only run once l1d_label_agreement.py
confirms label_agreement finds real edges faster than structural.
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from loop import run_loop, full_graph_accuracy
from scorers import random_scorer, structural_scorer, label_agreement_scorer
from eval import auc

SEEDS = list(range(30))
BUDGET = 200
OBSERVE_FRAC = 0.15

SCORERS = {
    "random": random_scorer,
    "structural": structural_scorer,
    "label_agree": label_agreement_scorer,
}


def main():
    y, A_true, X = make_csbm(seed=0)

    acc_results, edges_results, full_acc_by_seed = {}, {}, {}
    for name, scorer in SCORERS.items():
        acc_curves, edge_curves = [], []
        for seed in SEEDS:
            env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=seed)
            if seed not in full_acc_by_seed:
                full_acc_by_seed[seed] = full_graph_accuracy(env, A_true, seed=seed)
            result = run_loop(env, scorer, budget=BUDGET, seed=seed)
            acc_curves.append(result["accuracy"])
            edge_curves.append(result["n_real_edges"])
        acc_results[name] = np.array(acc_curves)
        edges_results[name] = np.array(edge_curves)
        print(f"{name}: final edges mean={edges_results[name][:,-1].mean():.1f}, "
              f"final acc mean={acc_results[name][:,-1].mean():.3f}, "
              f"AUC mean={np.mean([auc(c) for c in acc_results[name]]):.2f}")

    print(f"\nPaired significance tests vs structural (n={len(SEEDS)} seeds):")
    for name in ["random", "label_agree"]:
        for metric_name, values, ref_values in [
            ("final_edges", edges_results[name][:, -1], edges_results["structural"][:, -1]),
            ("final_acc", acc_results[name][:, -1], acc_results["structural"][:, -1]),
            ("auc", np.array([auc(c) for c in acc_results[name]]), np.array([auc(c) for c in acc_results["structural"]])),
        ]:
            diffs = values - ref_values
            t_stat, p_value = stats.ttest_rel(values, ref_values)
            sig = "  ***" if p_value < 0.05 else ""
            print(f"  {name} vs structural, {metric_name}: mean diff={diffs.mean():+.3f}, "
                  f"t={t_stat:.2f}, p={p_value:.4f}{sig}")


if __name__ == "__main__":
    main()
