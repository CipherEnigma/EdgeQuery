import numpy as np
import torch

from env import GraphEnv
from model import GCN, normalize_adj, train_model


def quick_diagnostic(scorer_fn, y, A_true, X, budget=15, seeds=(0, 1, 2, 3, 4), observe_frac=0.15):
    """Cheap pre-check for a scorer: how many real edges does it acquire within a small
    budget, across a few seeds. Meant to run in seconds-to-minutes, as a mandatory gate
    before a scorer earns a spot in an expensive confirmatory experiment (e.g. run_multi_seed
    at budget=200, 10 seeds). Does NOT decide anything on its own -- it's a smell test."""
    hits_per_seed = []
    for seed in seeds:
        env = GraphEnv(y, A_true, X, observe_frac=observe_frac, seed=seed)
        yt = torch.tensor(env.y, dtype=torch.long)
        Xt = torch.tensor(env.X, dtype=torch.float32)
        train_idx = torch.tensor(env.train_idx, dtype=torch.long)

        hits = 0
        for _ in range(budget):
            A_norm = torch.tensor(normalize_adj(env.observed_graph()), dtype=torch.float32)
            torch.manual_seed(seed)
            model = GCN(Xt.shape[1], 16, int(y.max()) + 1)
            train_model(model, Xt, A_norm, yt, train_idx, epochs=100)

            candidates = env.candidates()
            scores = scorer_fn(env, candidates, model, Xt, A_norm)
            best_score = scores.max()
            tied = np.flatnonzero(scores == best_score)
            pick = candidates[int(env.rng.choice(tied))]
            hits += env.reveal(pick)

        hits_per_seed.append(hits)

    return {"hits_per_seed": hits_per_seed, "mean_hits": float(np.mean(hits_per_seed))}


def diagnostic_report(scorers, y, A_true, X, budget=15, seeds=(0, 1, 2, 3, 4), observe_frac=0.15):
    """Run quick_diagnostic on every scorer in `scorers` (dict of name -> scorer_fn, must
    include 'random'), print a one-line verdict per scorer relative to random, and return
    the raw results. A scorer whose mean_hits doesn't clearly clear random's is flagged as
    NOT earning a spot in an expensive confirmatory run without further investigation."""
    assert "random" in scorers, "diagnostic_report needs a 'random' baseline to compare against"

    results = {}
    for name, scorer_fn in scorers.items():
        results[name] = quick_diagnostic(scorer_fn, y, A_true, X, budget=budget, seeds=seeds, observe_frac=observe_frac)

    random_mean = results["random"]["mean_hits"]
    print(f"Quick diagnostic (budget={budget}, seeds={list(seeds)}, observe_frac={observe_frac}):")
    for name, r in results.items():
        gap = r["mean_hits"] - random_mean
        if name == "random":
            verdict = "(baseline)"
        elif gap > 0.5:
            verdict = "clears random -- worth a confirmatory run"
        elif gap < -0.5:
            verdict = "WORSE than random -- do not promote without fixing"
        else:
            verdict = "not clearly different from random -- do not promote yet"
        print(f"  {name:>15s}: hits={r['hits_per_seed']} mean={r['mean_hits']:.2f}  {verdict}")

    return results
