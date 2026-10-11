"""Layer 2 gate: does a lower label rate make edge value heterogeneous?

Rerun with 10 torch seeds (10-seed-mean edge values have split-half-corrected
reliability ~0.87, see evaluator_check.py; 3 seeds gave ~0.67) and a label-rate sweep.

For every hidden real edge, add it alone to the starting observed graph, retrain, and
measure the drop in test cross-entropy (continuous, unlike accuracy). Repeat over several
torch seeds so per-edge training noise can be estimated and subtracted before judging how
spread out the values really are. Compared across labelling configs on the same graph.

Passes if, with anchors: (1) noise-corrected edge-value spread is clearly larger than in
the current 40%-label setup, and (2) value concentrates on edges bridging far-from-label
nodes toward labelled structure -- which structural's shared-neighbour count can't see.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from model import GCN, normalize_adj, train_model

TORCH_SEEDS = list(range(10))
ENV_SEEDS = [0, 1]
OBSERVE_FRAC = 0.15
CONFIGS = {
    "current (40% labels)": None,
    "12 per class": 12,
    "6 per class": 6,
    "3 per class": 3,
}


def evaluate(A_obs, Xt, yt, train_idx, test_idx, seed):
    A_norm = torch.tensor(normalize_adj(A_obs), dtype=torch.float32)
    torch.manual_seed(seed)
    model = GCN(Xt.shape[1], 16, int(yt.max()) + 1)
    train_model(model, Xt, A_norm, yt, train_idx, epochs=100)
    model.eval()
    with torch.no_grad():
        out = model(Xt, A_norm)
        nll = F.cross_entropy(out[test_idx], yt[test_idx]).item()
        acc = (out[test_idx].argmax(1) == yt[test_idx]).float().mean().item()
    return nll, acc


def main():
    y, A_true, X = make_csbm(seed=0)
    yt = torch.tensor(y, dtype=torch.long)
    Xt = torch.tensor(X, dtype=torch.float32)

    for label, anchors in CONFIGS.items():
        print(f"\n=== {label} ===")
        for env_seed in ENV_SEEDS:
            env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=env_seed, anchors_per_class=anchors)
            tr = torch.tensor(env.train_idx, dtype=torch.long)
            te = torch.tensor(env.test_idx, dtype=torch.long)
            A_obs = env.observed_graph()
            hidden = [e for e in env.true_edges if e not in env.revealed]

            base = [evaluate(A_obs, Xt, yt, tr, te, s) for s in TORCH_SEEDS]
            full = [evaluate(A_true, Xt, yt, tr, te, s) for s in TORCH_SEEDS]
            base_nll = np.array([b[0] for b in base])
            base_acc = np.mean([b[1] for b in base])
            full_acc = np.mean([f[1] for f in full])

            nll_gain = np.zeros((len(hidden), len(TORCH_SEEDS)))
            acc_gain = np.zeros((len(hidden), len(TORCH_SEEDS)))
            for k, (i, j) in enumerate(hidden):
                A_temp = A_obs.copy()
                A_temp[i, j] = A_temp[j, i] = 1.0
                for t, s in enumerate(TORCH_SEEDS):
                    nll, acc = evaluate(A_temp, Xt, yt, tr, te, s)
                    nll_gain[k, t] = base_nll[t] - nll
                    acc_gain[k, t] = acc - base[t][1]

            means = nll_gain.mean(1)
            noise_var = nll_gain.var(1, ddof=1).mean() / len(TORCH_SEEDS)
            total_var = means.var(ddof=1)
            signal_sd = np.sqrt(max(total_var - noise_var, 0.0))
            noise_sd = np.sqrt(noise_var)

            pos = np.sort(means[means > 0])[::-1]
            top10 = pos[: max(1, len(means) // 10)].sum() / pos.sum() if len(pos) else float("nan")

            struct = np.array([float(A_obs[i] @ A_obs[j]) for (i, j) in hidden])
            dist = shortest_path(A_obs, unweighted=True, indices=env.train_idx).min(axis=0)
            dist = np.where(np.isinf(dist), 99, dist)
            bridge = np.array([min(dist[i], dist[j]) <= 1 and max(dist[i], dist[j]) >= 3 for (i, j) in hidden])
            rho_struct = stats.spearmanr(means, struct).correlation if struct.std() > 0 else float("nan")

            print(f"  env_seed={env_seed}: hidden real edges={len(hidden)}, "
                  f"start acc={base_acc:.3f}, full-graph acc={full_acc:.3f}, "
                  f"unreachable-from-labels nodes={int((dist == 99).sum())}")
            print(f"    edge value (test-NLL drop): mean={means.mean():+.4f}, "
                  f"signal sd={signal_sd:.4f}, noise sd={noise_sd:.4f}, signal/noise={signal_sd / noise_sd:.2f}")
            print(f"    top-10% of edges hold {100 * top10:.0f}% of positive value; "
                  f"mean acc gain per edge={acc_gain.mean():+.4f}, max={acc_gain.mean(1).max():+.4f}")
            print(f"    bridging edges: n={int(bridge.sum())}, mean value={means[bridge].mean() if bridge.any() else float('nan'):+.4f} "
                  f"vs others={means[~bridge].mean():+.4f}; "
                  f"spearman(value, structural score)={rho_struct:+.2f}")


if __name__ == "__main__":
    main()
