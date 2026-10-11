"""
Weighted edge data generation with homophily-aware edge strengths.
Extends make_csbm to support weighted edges based on within/cross-class relationships.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import make_csbm


def make_weighted_csbm(n=90, k=3, p_in=0.13, p_out=0.006, w_in=0.8, w_out=0.2,
                       feat_dim=12, feat_signal=0.10, seed=0):
    """
    Generate weighted CSBM where edge weights reflect homophily.

    Args:
        n, k, p_in, p_out, feat_dim, feat_signal, seed: same as make_csbm
        w_in: weight for within-class edges (0.0 to 1.0)
        w_out: weight for cross-class edges (0.0 to 1.0)

    Returns:
        y: node labels (n,)
        A_binary: binary adjacency (n, n) -- structure only
        A_weighted: weighted adjacency (n, n) -- weights reflect homophily
        X: node features (n, feat_dim)
    """
    y, A_binary, X = make_csbm(n=n, k=k, p_in=p_in, p_out=p_out,
                               feat_dim=feat_dim, feat_signal=feat_signal, seed=seed)

    A_weighted = np.zeros_like(A_binary, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if A_binary[i, j] == 1.0:
                w = w_in if y[i] == y[j] else w_out
                A_weighted[i, j] = w
                A_weighted[j, i] = w

    return y, A_binary, A_weighted, X


def make_reliability_weighted_csbm(n=90, k=3, p_in=0.13, p_out=0.006,
                                   reliability_mean=0.7, reliability_std=0.15,
                                   feat_dim=12, feat_signal=0.10, seed=0):
    """
    Generate CSBM with per-edge reliability factors (random confidence scores).

    Args:
        Same as make_csbm, plus:
        reliability_mean: mean reliability factor for each edge
        reliability_std: std dev of reliability factors

    Returns:
        y, A_binary, A_reliability: weighted by confidence, X
    """
    y, A_binary, A_weighted, X = make_weighted_csbm(
        n=n, k=k, p_in=p_in, p_out=p_out, w_in=1.0, w_out=1.0,
        feat_dim=feat_dim, feat_signal=feat_signal, seed=seed
    )

    rng = np.random.default_rng(seed + 1)
    A_reliability = np.zeros_like(A_binary, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if A_binary[i, j] == 1.0:
                r = np.clip(rng.normal(reliability_mean, reliability_std), 0.0, 1.0)
                A_reliability[i, j] = r
                A_reliability[j, i] = r

    return y, A_binary, A_reliability, X


if __name__ == "__main__":
    # Test weighted generation
    y, A_bin, A_wgt, X = make_weighted_csbm(seed=42)
    print(f"Nodes: {len(y)}, Classes: {int(y.max()) + 1}")
    print(f"Binary edges: {(A_bin > 0).sum() // 2}")
    print(f"Weighted edges range: [{A_wgt[A_wgt > 0].min():.3f}, {A_wgt[A_wgt > 0].max():.3f}]")

    # Check within/cross-class weight distribution
    within_mask = (A_bin > 0) & (y[np.arange(len(y))[:, None]] == y[np.arange(len(y))])
    cross_mask = (A_bin > 0) & (y[np.arange(len(y))[:, None]] != y[np.arange(len(y))])

    print(f"Within-class edge weight: {A_wgt[within_mask].mean():.3f} (should be ~0.8)")
    print(f"Cross-class edge weight: {A_wgt[cross_mask].mean():.3f} (should be ~0.2)")

    # Test reliability
    y2, A_bin2, A_rel, X2 = make_reliability_weighted_csbm(seed=42)
    print(f"\nReliability edges range: [{A_rel[A_rel > 0].min():.3f}, {A_rel[A_rel > 0].max():.3f}]")
    print(f"Reliability mean: {A_rel[A_rel > 0].mean():.3f} (should be ~0.7)")
