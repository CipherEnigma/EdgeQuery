import numpy as np
import torch

from model import GCN, normalize_adj, train_model, accuracy
from scorers import random_scorer


def run_loop(env, scorer_fn, budget, hidden_dim=16, epochs=100, lr=0.01, seed=0):
    num_classes = int(env.y.max()) + 1
    feat_dim = env.X.shape[1]

    y = torch.tensor(env.y, dtype=torch.long)
    X = torch.tensor(env.X, dtype=torch.float32)
    train_idx = torch.tensor(env.train_idx, dtype=torch.long)
    test_idx = torch.tensor(env.test_idx, dtype=torch.long)

    acc_curve = []
    n_revealed_curve = []
    n_real_edges_curve = []

    for step in range(budget + 1):
        A_obs = env.observed_graph()
        A_norm = torch.tensor(normalize_adj(A_obs), dtype=torch.float32)

        torch.manual_seed(seed)
        model = GCN(feat_dim, hidden_dim, num_classes)
        train_model(model, X, A_norm, y, train_idx, epochs=epochs, lr=lr)
        acc = accuracy(model, X, A_norm, y, test_idx)

        acc_curve.append(acc)
        n_revealed_curve.append(len(env.revealed))
        n_real_edges_curve.append(int(A_obs.sum() / 2))

        if step == budget:
            break

        candidates = env.candidates()
        if not candidates:
            break

        scores = scorer_fn(env, candidates, model, X, A_norm)
        best = candidates[int(np.argmax(scores))]
        env.reveal(best)

    return {"accuracy": acc_curve, "n_revealed": n_revealed_curve, "n_real_edges": n_real_edges_curve}


def full_graph_accuracy(env, A_true, hidden_dim=16, epochs=100, lr=0.01, seed=0):
    """Ceiling reference: accuracy if the model got the *entire* true graph for free,
    using the same node splits as env. Node splits don't depend on observe_frac (they're
    computed before any edges are revealed), so this is comparable across an observe_frac
    sweep for a fixed seed."""
    num_classes = int(env.y.max()) + 1
    feat_dim = env.X.shape[1]
    y_t = torch.tensor(env.y, dtype=torch.long)
    X_t = torch.tensor(env.X, dtype=torch.float32)
    train_idx = torch.tensor(env.train_idx, dtype=torch.long)
    test_idx = torch.tensor(env.test_idx, dtype=torch.long)
    A_norm = torch.tensor(normalize_adj(A_true), dtype=torch.float32)

    torch.manual_seed(seed)
    model = GCN(feat_dim, hidden_dim, num_classes)
    train_model(model, X_t, A_norm, y_t, train_idx, epochs=epochs, lr=lr)
    return accuracy(model, X_t, A_norm, y_t, test_idx)


def run_multi_seed(y, A_true, X, scorer_fn, budget, seeds, observe_frac=0.3,
                    hidden_dim=16, epochs=100, lr=0.01):
    """Run the loop once per seed (env init + model init vary, graph is fixed)
    and stack the resulting accuracy curves for averaging."""
    from env import GraphEnv

    curves = []
    for seed in seeds:
        env = GraphEnv(y, A_true, X, observe_frac=observe_frac, seed=seed)
        result = run_loop(env, scorer_fn, budget, hidden_dim=hidden_dim, epochs=epochs, lr=lr, seed=seed)
        curves.append(result["accuracy"])
    return np.array(curves)


if __name__ == "__main__":
    from data import make_csbm
    from env import GraphEnv

    y, A_true, X = make_csbm()
    env = GraphEnv(y, A_true, X, observe_frac=0.3, seed=0)

    result = run_loop(env, random_scorer, budget=10, epochs=100, seed=0)

    print("Edges revealed at each step:", result["n_revealed"])
    print("Test accuracy at each step:", [f"{a:.3f}" for a in result["accuracy"]])

    assert len(result["accuracy"]) == len(result["n_revealed"])
    assert result["n_revealed"] == sorted(result["n_revealed"]), "revealed count should never decrease"
    assert result["n_revealed"][-1] > result["n_revealed"][0], "graph should have grown"
    print("Loop ran end-to-end: accuracy recorded each step, graph grew.")
