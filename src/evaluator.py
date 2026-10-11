"""Noise-free evaluator: SGC (Wu et al. 2019) -- K rounds of normalized-adjacency
propagation followed by L2-regularized multinomial logistic regression. The objective
is strictly convex, so the fitted model is unique: the same graph always gives the same
score, and a single edge's effect can be measured exactly instead of estimated through
GCN retraining noise (which was as large as the edge effect itself, see Layer 2 gate)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from model import normalize_adj


def sgc_features(A, X, K=2):
    A_norm = normalize_adj(A)
    Z = X.astype(np.float64)
    for _ in range(K):
        Z = A_norm @ Z
    return Z


def sgc_eval(A, X, y, train_idx, test_idx, K=2, C=1.0):
    """Returns (test cross-entropy, test accuracy) for SGC trained on train_idx."""
    Z = sgc_features(A, X, K)
    clf = LogisticRegression(C=C, max_iter=5000, tol=1e-10)
    clf.fit(Z[train_idx], y[train_idx])
    proba = clf.predict_proba(Z[test_idx])
    labels = np.arange(int(y.max()) + 1)
    nll = log_loss(y[test_idx], proba, labels=labels)
    acc = float((proba.argmax(1) == y[test_idx]).mean())
    return nll, acc


def edge_values(A_obs, X, y, train_idx, test_idx, edges, K=2, C=1.0):
    """Exact value of adding each edge alone to A_obs: drop in test cross-entropy."""
    base_nll, _ = sgc_eval(A_obs, X, y, train_idx, test_idx, K, C)
    values = np.zeros(len(edges))
    for k, (i, j) in enumerate(edges):
        A_temp = A_obs.copy()
        A_temp[i, j] = A_temp[j, i] = 1.0
        nll, _ = sgc_eval(A_temp, X, y, train_idx, test_idx, K, C)
        values[k] = base_nll - nll
    return values
