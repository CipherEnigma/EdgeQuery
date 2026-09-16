import numpy as np
import torch

from model import GCN, normalize_adj, train_model, accuracy, entropy_uncertainty


def random_scorer(env, candidates, model=None, X=None, A_norm=None):
    """Baseline: rate every candidate equally at random."""
    return env.rng.random(len(candidates))


def uncertainty_scorer(env, candidates, model, X, A_norm):
    """Prefer candidates whose two endpoints the model is currently most unsure about."""
    ent = entropy_uncertainty(model, X, A_norm)
    return np.array([(ent[i] + ent[j]).item() for (i, j) in candidates])


def structural_scorer(env, candidates, model=None, X=None, A_norm=None):
    """Prefer candidates whose endpoints already share many observed neighbours."""
    A_obs = env.observed_graph()
    return np.array([float(np.dot(A_obs[i], A_obs[j])) for (i, j) in candidates])


def oracle_voi_scorer(env, candidates, model, X, A_norm, n_sample=15, epochs=25, hidden_dim=16, lr=0.01, seed=0):
    """Gold-standard, expensive scorer: for a random sample of candidates, tentatively
    add the edge, retrain from scratch, and score by the resulting validation accuracy.
    Unsampled candidates get the worst possible score so they're never picked."""
    n_candidates = len(candidates)
    if n_candidates > n_sample:
        sample_pos = env.rng.choice(n_candidates, size=n_sample, replace=False)
    else:
        sample_pos = np.arange(n_candidates)

    num_classes = int(env.y.max()) + 1
    feat_dim = X.shape[1]
    y = torch.tensor(env.y, dtype=torch.long)
    train_idx = torch.tensor(env.train_idx, dtype=torch.long)
    val_idx = torch.tensor(env.val_idx, dtype=torch.long)
    A_obs = env.observed_graph()

    scores = np.full(n_candidates, -np.inf)
    for pos in sample_pos:
        i, j = candidates[pos]
        A_temp = A_obs.copy()
        A_temp[i, j] = 1.0
        A_temp[j, i] = 1.0
        A_norm_temp = torch.tensor(normalize_adj(A_temp), dtype=torch.float32)

        torch.manual_seed(seed)
        temp_model = GCN(feat_dim, hidden_dim, num_classes)
        train_model(temp_model, X, A_norm_temp, y, train_idx, epochs=epochs, lr=lr)
        scores[pos] = accuracy(temp_model, X, A_norm_temp, y, val_idx)

    return scores
