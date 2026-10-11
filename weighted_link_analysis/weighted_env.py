"""
Graph environment supporting weighted edges for edge discovery experiments.
Extends GraphEnv to track and use edge weights in candidate scoring.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from env import GraphEnv


class WeightedGraphEnv(GraphEnv):
    """
    Like GraphEnv but tracks edge weights separately from binary structure.
    A_true_weighted stores edge strengths; oracle still reveals binary presence.
    """

    def __init__(self, y, A_true_binary, A_true_weighted, X, observe_frac=0.15,
                 train_frac=0.4, val_frac=0.2, seed=0, anchors_per_class=None):
        """
        Args:
            y: node labels (n,)
            A_true_binary: true binary adjacency (n, n) -- connectivity structure
            A_true_weighted: weighted adjacency (n, n) -- edge strengths
            X: node features (n, feat_dim)
            observe_frac, train_frac, val_frac, seed, anchors_per_class: as in GraphEnv
        """
        self.A_true_weighted = A_true_weighted.astype(np.float32)
        self.A_obs_weighted = np.zeros_like(A_true_weighted, dtype=np.float32)

        super().__init__(y, A_true_binary, X, observe_frac=observe_frac,
                         train_frac=train_frac, val_frac=val_frac,
                         seed=seed, anchors_per_class=anchors_per_class)

    def _set_edge(self, i, j):
        """Override parent to also set weights when edge is revealed."""
        super()._set_edge(i, j)
        self.A_obs_weighted[i, j] = self.A_true_weighted[i, j]
        self.A_obs_weighted[j, i] = self.A_true_weighted[j, i]

    def observed_graph(self, weighted=False):
        """
        Args:
            weighted: if True, return weighted adjacency; else binary

        Returns:
            A_obs (weighted or binary)
        """
        if weighted:
            return self.A_obs_weighted.copy()
        else:
            return self.A_obs.copy()


if __name__ == "__main__":
    from weighted_data import make_weighted_csbm

    y, A_bin, A_wgt, X = make_weighted_csbm(seed=0)
    env = WeightedGraphEnv(y, A_bin, A_wgt, X, observe_frac=0.15, seed=0)

    # Check that weights are properly tracked
    assert env.A_obs_weighted.shape == A_wgt.shape
    print(f"[ok] Weighted env initialized correctly")

    # Spot-check: revealed edges should have their true weights
    for (i, j) in list(env.revealed)[:5]:
        assert env.A_obs_weighted[i, j] == A_wgt[i, j]
    print(f"[ok] Weights correctly assigned to revealed edges")

    # Unrevealed edges should have zero weight
    assert (env.A_obs_weighted[env.A_obs == 0] == 0).all()
    print(f"[ok] Unrevealed edges have zero weight")

    print("All weighted env checks passed.")
