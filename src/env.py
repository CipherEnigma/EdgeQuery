import numpy as np


class GraphEnv:
    """Hides most true edges; reveals one at a time via an oracle, at a cost."""

    def __init__(self, y, A_true, X, observe_frac=0.1, train_frac=0.4, val_frac=0.2, seed=0):
        self.y = y
        self.A_true = A_true
        self.X = X
        self.n = A_true.shape[0]
        self.rng = np.random.default_rng(seed)

        self.train_idx, self.val_idx, self.test_idx = self._split_nodes(train_frac, val_frac)

        self.true_edges = self._all_true_edges()
        self.revealed = set()
        self.A_obs = np.zeros_like(A_true)

        self._init_observed_edges(observe_frac)

    def _split_nodes(self, train_frac, val_frac):
        idx = self.rng.permutation(self.n)
        n_train = int(self.n * train_frac)
        n_val = int(self.n * val_frac)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
        return train_idx, val_idx, test_idx

    def _all_true_edges(self):
        iu = np.triu_indices(self.n, k=1)
        mask = self.A_true[iu] == 1.0
        return list(zip(iu[0][mask].tolist(), iu[1][mask].tolist()))

    def _init_observed_edges(self, observe_frac):
        n_init = int(len(self.true_edges) * observe_frac)
        chosen = self.rng.choice(len(self.true_edges), size=n_init, replace=False)
        for k in chosen:
            i, j = self.true_edges[k]
            self._set_edge(i, j)
            self.revealed.add((i, j))

    def _set_edge(self, i, j):
        self.A_obs[i, j] = 1.0
        self.A_obs[j, i] = 1.0

    def candidates(self):
        """All unrevealed (i, j) pairs — may or may not be real edges."""
        iu = np.triu_indices(self.n, k=1)
        pairs = zip(iu[0].tolist(), iu[1].tolist())
        return [p for p in pairs if p not in self.revealed]

    def reveal(self, edge):
        """Ask the oracle about one candidate edge. Adds it to the observed graph
        only if it's real. Returns True/False for whether it was real."""
        i, j = edge
        if edge in self.revealed:
            raise ValueError(f"Edge {edge} was already revealed")
        self.revealed.add(edge)
        is_real = bool(self.A_true[i, j] == 1.0)
        if is_real:
            self._set_edge(i, j)
        return is_real

    def observed_graph(self):
        return self.A_obs.copy()


if __name__ == "__main__":
    from data import make_csbm

    y, A_true, X = make_csbm()
    env = GraphEnv(y, A_true, X, observe_frac=0.1, seed=0)

    # 1. splits are disjoint and cover every node
    all_idx = np.concatenate([env.train_idx, env.val_idx, env.test_idx])
    assert len(set(all_idx.tolist())) == env.n, "splits overlap or miss nodes"
    print("[ok] train/val/test splits disjoint and complete")

    # 2. no hidden true edge leaks into the observed graph
    for (i, j) in env.true_edges:
        if (i, j) not in env.revealed:
            assert env.A_obs[i, j] == 0.0, f"hidden edge {(i, j)} leaked into A_obs"
    print("[ok] hidden edges are not present in observed graph")

    # 3. revealing a real edge changes exactly that one entry pair
    candidates = env.candidates()
    real_candidate = next(c for c in candidates if env.A_true[c[0], c[1]] == 1.0)
    before = env.A_obs.copy()
    was_real = env.reveal(real_candidate)
    diff = np.argwhere(env.A_obs != before)
    assert was_real is True
    assert len(diff) == 2, f"reveal changed {len(diff)} entries, expected 2"
    i, j = real_candidate
    assert env.A_obs[i, j] == 1.0 and env.A_obs[j, i] == 1.0
    print("[ok] revealing a real edge adds only that edge (both symmetric entries)")

    # 4. revealing a non-edge changes nothing
    fake_candidate = next(c for c in env.candidates() if env.A_true[c[0], c[1]] == 0.0)
    before = env.A_obs.copy()
    was_real = env.reveal(fake_candidate)
    assert was_real is False
    assert np.array_equal(env.A_obs, before), "A_obs changed after revealing a non-edge"
    print("[ok] revealing a non-edge leaves the observed graph unchanged")

    # 5. can't reveal the same edge twice
    try:
        env.reveal(real_candidate)
        raised = False
    except ValueError:
        raised = True
    assert raised, "revealing an already-revealed edge should raise"
    print("[ok] re-revealing an edge raises")

    print("All leakage checks passed.")
