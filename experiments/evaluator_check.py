"""Validate src/evaluator.py (SGC) before trusting it as the edge-value measure.

1. Determinism: identical input -> identical output.
2. Same qualitative behaviour as the GCN: graph >> features-only, similar headroom.
3. Same edge ranking as the GCN: Spearman(SGC edge value, GCN edge value averaged over
   10 torch seeds). The ceiling for that correlation is set by how reliable the 10-seed
   GCN mean itself is (split-half reliability, Spearman-Brown corrected), so we also
   report the correlation corrected for GCN noise.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from model import GCN, normalize_adj, train_model
from evaluator import sgc_eval, edge_values

OBSERVE_FRAC = 0.15
GCN_SEEDS = list(range(10))


def gcn_nll(A, Xt, yt, tr, te, seed):
    A_norm = torch.tensor(normalize_adj(A), dtype=torch.float32)
    torch.manual_seed(seed)
    m = GCN(Xt.shape[1], 16, int(yt.max()) + 1)
    train_model(m, Xt, A_norm, yt, tr, epochs=100)
    m.eval()
    with torch.no_grad():
        return F.cross_entropy(m(Xt, A_norm)[te], yt[te]).item()


def main():
    y, A_true, X = make_csbm(seed=0)
    n = len(y)

    print("=== 1. Determinism ===")
    env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=0)
    A_obs = env.observed_graph()
    a = sgc_eval(A_obs, X, y, env.train_idx, env.test_idx)
    b = sgc_eval(A_obs, X, y, env.train_idx, env.test_idx)
    hidden = [e for e in env.true_edges if e not in env.revealed]
    t0 = time.time()
    v1 = edge_values(A_obs, X, y, env.train_idx, env.test_idx, hidden)
    elapsed = time.time() - t0
    v2 = edge_values(A_obs, X, y, env.train_idx, env.test_idx, hidden)
    print(f"  sgc_eval repeat identical: {a == b}; edge_values repeat identical: {np.array_equal(v1, v2)}")
    print(f"  valuing all {len(hidden)} hidden edges took {elapsed:.1f}s")

    print("\n=== 2. Headroom (SGC accuracy, mean over env seeds 0-4) ===")
    rows = {"features only": [], "observed graph": [], "full graph": []}
    for s in range(5):
        e = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=s)
        rows["features only"].append(sgc_eval(np.zeros((n, n), dtype=np.float32), X, y, e.train_idx, e.test_idx)[1])
        rows["observed graph"].append(sgc_eval(e.observed_graph(), X, y, e.train_idx, e.test_idx)[1])
        rows["full graph"].append(sgc_eval(A_true, X, y, e.train_idx, e.test_idx)[1])
    for k, v in rows.items():
        print(f"  {k:>15s}: {np.mean(v):.3f}  {[round(x, 3) for x in v]}")

    print("\n=== 3. Agreement with GCN edge values (env seed 0, all hidden real edges) ===")
    yt = torch.tensor(y, dtype=torch.long)
    Xt = torch.tensor(X, dtype=torch.float32)
    tr = torch.tensor(env.train_idx, dtype=torch.long)
    te = torch.tensor(env.test_idx, dtype=torch.long)
    base = np.array([gcn_nll(A_obs, Xt, yt, tr, te, s) for s in GCN_SEEDS])
    gcn_vals = np.zeros((len(hidden), len(GCN_SEEDS)))
    for k, (i, j) in enumerate(hidden):
        A_temp = A_obs.copy()
        A_temp[i, j] = A_temp[j, i] = 1.0
        gcn_vals[k] = base - np.array([gcn_nll(A_temp, Xt, yt, tr, te, s) for s in GCN_SEEDS])

    half_a = gcn_vals[:, :5].mean(1)
    half_b = gcn_vals[:, 5:].mean(1)
    r_half = stats.spearmanr(half_a, half_b).correlation
    rel10 = 2 * r_half / (1 + r_half) if r_half > 0 else float("nan")
    r_sgc = stats.spearmanr(v1, gcn_vals.mean(1)).correlation
    print(f"  GCN split-half reliability (5 vs 5 seeds): {r_half:+.3f}  -> 10-seed mean reliability {rel10:.3f}")
    print(f"  Spearman(SGC value, GCN 10-seed mean value): {r_sgc:+.3f}")
    if rel10 > 0:
        print(f"  corrected for GCN noise: {r_sgc / np.sqrt(rel10):+.3f}  (1.0 = SGC ranks edges exactly as a noise-free GCN would)")


if __name__ == "__main__":
    main()
