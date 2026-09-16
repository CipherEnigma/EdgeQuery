import numpy as np
import torch

from model import GCN, normalize_adj, train_model, accuracy


def run_loop(env, scorer_fn, budget, hidden_dim=16, epochs=100, lr=0.01, seed=0):
    num_classes = int(env.y.max()) + 1
    feat_dim = env.X.shape[1]

    y = torch.tensor(env.y, dtype=torch.long)
    X = torch.tensor(env.X, dtype=torch.float32)
    train_idx = torch.tensor(env.train_idx, dtype=torch.long)
    test_idx = torch.tensor(env.test_idx, dtype=torch.long)

    acc_curve = []
    n_revealed_curve = []

    for step in range(budget + 1):
        A_norm = torch.tensor(normalize_adj(env.observed_graph()), dtype=torch.float32)

        torch.manual_seed(seed)
        model = GCN(feat_dim, hidden_dim, num_classes)
        train_model(model, X, A_norm, y, train_idx, epochs=epochs, lr=lr)
        acc = accuracy(model, X, A_norm, y, test_idx)

        acc_curve.append(acc)
        n_revealed_curve.append(len(env.revealed))

        if step == budget:
            break

        candidates = env.candidates()
        if not candidates:
            break

        scores = scorer_fn(env, candidates, model)
        best = candidates[int(np.argmax(scores))]
        env.reveal(best)

    return {"accuracy": acc_curve, "n_revealed": n_revealed_curve}


def random_scorer(env, candidates, model=None):
    return env.rng.random(len(candidates))


if __name__ == "__main__":
    from data import make_csbm
    from env import GraphEnv

    y, A_true, X = make_csbm()
    env = GraphEnv(y, A_true, X, observe_frac=0.1, seed=0)

    result = run_loop(env, random_scorer, budget=10, epochs=100, seed=0)

    print("Edges revealed at each step:", result["n_revealed"])
    print("Test accuracy at each step:", [f"{a:.3f}" for a in result["accuracy"]])

    assert len(result["accuracy"]) == len(result["n_revealed"])
    assert result["n_revealed"] == sorted(result["n_revealed"]), "revealed count should never decrease"
    assert result["n_revealed"][-1] > result["n_revealed"][0], "graph should have grown"
    print("Loop ran end-to-end: accuracy recorded each step, graph grew.")
