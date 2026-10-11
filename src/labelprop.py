"""Label propagation (Zhu et al. 2003) as a deterministic downstream classifier.

Replaces the GCN for the MVP. Why: GCN value-of-an-edge has no closed form, so scoring
requires retraining -- expensive (oracle could only sample 15/4000 candidates) and noisy
(per-edge signal/noise was 1.2-2.1). LP is deterministic and a recompute costs
milliseconds, so an edge's value is EXACT and every candidate can be scored.

No weights, no training, no features: F <- S @ F with labelled rows clamped, iterated to
convergence. Same graph always gives the same predictions.
"""
import numpy as np

# Fixed generator for unreachable-node predictions. Must not vary between candidates,
# otherwise the oracle would be comparing tie-break luck instead of edge value.
_UNREACHED_RNG_SEED = 0


def propagate(A, y, train_idx, k=None, max_iter=200, tol=1e-9):
    """Returns F (n x k) label scores. Row-normalised adjacency, labelled rows clamped."""
    n = A.shape[0]
    k = int(y.max()) + 1 if k is None else k
    Y = np.zeros((n, k))
    Y[train_idx, y[train_idx]] = 1.0

    deg = A.sum(axis=1)
    S = np.divide(A, deg[:, None], out=np.zeros((n, n)), where=deg[:, None] > 0)

    F = Y.copy()
    for _ in range(max_iter):
        F_new = S @ F
        F_new[train_idx] = Y[train_idx]
        if np.abs(F_new - F).max() < tol:
            return F_new
        F = F_new
    return F


def predict(F):
    """argmax, with unreachable (all-zero) rows given a fixed deterministic class.

    A plain argmax returns class 0 for an all-zero row, which would systematically
    favour class 0. Instead those rows get a fixed pseudo-random class -- identical on
    every call, so adding a candidate edge changes a node's prediction only by actually
    making it reachable."""
    mass = F.sum(axis=1)
    preds = F.argmax(axis=1)
    unreached = mass <= 0
    if unreached.any():
        rng = np.random.default_rng(_UNREACHED_RNG_SEED)
        preds = preds.copy()
        preds[unreached] = rng.integers(0, F.shape[1], size=int(unreached.sum()))
    return preds, unreached


def accuracy(A, y, train_idx, eval_idx, k=None):
    F = propagate(A, y, train_idx, k)
    preds, unreached = predict(F)
    return float((preds[eval_idx] == y[eval_idx]).mean()), F, unreached


def soft_margin(F, y, eval_idx):
    """Mean probability mass on the true class over eval nodes. Continuous, unlike
    accuracy, so it can break ties between candidates that don't flip a whole node."""
    mass = F.sum(axis=1, keepdims=True)
    P = np.divide(F, mass, out=np.full_like(F, 1.0 / F.shape[1]), where=mass > 0)
    return float(P[eval_idx, y[eval_idx]].mean())


def edge_values(A_obs, y, train_idx, eval_idx, candidates, k=None, tie_weight=1e-3):
    """EXACT value of adding each candidate edge: gain in eval accuracy, with the soft
    margin as a small tie-break. Accuracy on a small eval set is quantised (steps of
    1/|eval_idx|), so most candidates tie at exactly 0 without the tie-break -- the same
    quantisation trap that made the GCN oracle rank noise."""
    base_acc, base_F, _ = accuracy(A_obs, y, train_idx, eval_idx, k)
    base_margin = soft_margin(base_F, y, eval_idx)

    vals = np.zeros(len(candidates))
    for c, (i, j) in enumerate(candidates):
        A_temp = A_obs.copy()
        A_temp[i, j] = A_temp[j, i] = 1.0
        F = propagate(A_temp, y, train_idx, k)
        preds, _ = predict(F)
        acc = float((preds[eval_idx] == y[eval_idx]).mean())
        vals[c] = (acc - base_acc) + tie_weight * (soft_margin(F, y, eval_idx) - base_margin)
    return vals
