"""Experiment 2: score queries by their value to the classifier, with binary / weighted /
signed label propagation.

Value scorer (expected error reduction): score(i,j) = P(real) * E[gain in the classifier's
self-estimated accuracy over all unlabelled nodes | real]. No labels beyond train are used.
(A version scored on 18 validation nodes was tried first: it overfit them and lowered test
accuracy, since most test nodes start unreached by any label.) Here
  P(same class) comes from the classifier's current beliefs,
  P(real) = p_in * P(same) + p_out * (1 - P(same))   (assumes p_in, p_out are known),
and the edge's weight is unknown until queried, so the gain is averaged over w_in / w_out.
Exact gains are computed for a shortlist of SHORTLIST candidates ranked by P(real) x endpoint
uncertainty; everything else scores -inf.

Classifiers (how an observed edge with homophily weight w enters propagation):
  binary   -> 1          weighted -> w          signed -> 2w - 1  (cross-class edges push AGAINST)

Random and common-neighbour queries are classifier-independent, so across classifiers they
pick identical edges: comparing classifiers under them isolates the classifier; comparing
value vs random within a classifier isolates the query strategy.
Accuracy is reported on all unlabelled nodes, whose labels no scorer ever sees.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from weighted_data import make_weighted_csbm
from weighted_env import WeightedGraphEnv

N_SEEDS = 10
BUDGET = 100
CHECKPOINTS = [0, 10, 25, 50, 100]
SHORTLIST = 150
ALPHA = 0.99  # damping so the linear system is solvable on label-free components
GRAPH = dict(n=90, k=3, p_in=0.25, p_out=0.02, w_in=0.8, w_out=0.2)
LABELS = {"sparse_15": 0.15, "medium_40": 0.40}
VAL_FRAC = 0.0
CLASSIFIERS = ["binary", "weighted", "signed"]
STRATEGIES = ["random", "common_nbrs", "value"]


def edge_entry(kind, w):
    return {"binary": 1.0, "weighted": w, "signed": 2 * w - 1}[kind]


def classifier_graph(A_bin, A_w, kind):
    A_w = A_w.astype(float)
    if kind == "binary":
        return A_bin.astype(float)
    if kind == "weighted":
        return A_w
    return np.where(A_bin > 0, 2 * A_w - 1, 0.0)


def solve_lp(M, y, train_idx, k):
    """Closed-form clamped propagation: F_U = (I - a S_UU)^-1 a S_UL Y_L, S = D^-1 M, D = sum|M|."""
    n = M.shape[0]
    d = np.abs(M).sum(axis=1)
    S = np.divide(M, d[:, None], out=np.zeros_like(M), where=d[:, None] > 0)
    lab = np.zeros(n, dtype=bool)
    lab[train_idx] = True
    U = ~lab
    YL = np.zeros((int(lab.sum()), k))
    YL[np.arange(YL.shape[0]), y[lab]] = 1.0
    F = np.zeros((n, k))
    F[lab] = YL
    F[U] = np.linalg.solve(np.eye(int(U.sum())) - ALPHA * S[np.ix_(U, U)], ALPHA * S[np.ix_(U, lab)] @ YL)
    return F


def predict(F):
    """argmax; nodes no label reached keep a fixed per-node pseudo-random class, so a
    prediction changes only when the node is actually reached."""
    preds = F.argmax(axis=1)
    unreached = np.abs(F).sum(axis=1) < 1e-12
    fallback = np.random.default_rng(0).integers(0, F.shape[1], size=F.shape[0])
    return np.where(unreached, fallback, preds)


def beliefs(F):
    Q = np.clip(F, 0.0, None)
    mass = Q.sum(axis=1, keepdims=True)
    return np.divide(Q, mass, out=np.full_like(Q, 1.0 / F.shape[1]), where=mass > 0)


def self_estimated_accuracy(F, unlabelled):
    """Expected accuracy under the classifier's own beliefs (mean max-probability); uses no labels."""
    return float(beliefs(F)[unlabelled].max(axis=1).mean())


def value_scores(env, kind, cands, k):
    M = classifier_graph(env.A_obs, env.A_obs_weighted, kind)
    F = solve_lp(M, env.y, env.train_idx, k)
    q = beliefs(F)
    I, J = np.array(cands).T
    ps = (q[I] * q[J]).sum(axis=1)
    p_in_real = GRAPH["p_in"] * ps
    p_out_real = GRAPH["p_out"] * (1 - ps)
    p_real = p_in_real + p_out_real

    # Shortlist by P(real) x confidence gap: an edge carries information only from a node that
    # knows to one that doesn't. (P(real) alone picks already-confident pairs; P(real) x
    # uncertainty picks pairs of two unreached nodes -- both carry nothing.)
    conf = q.max(axis=1)
    key = p_real * np.abs(conf[I] - conf[J])
    unlabelled = np.ones(len(env.y), dtype=bool)
    unlabelled[env.train_idx] = False
    base = self_estimated_accuracy(F, unlabelled)
    shortlist = np.argsort(-(key + 1e-9 * env.rng.random(len(cands))))[:SHORTLIST]
    scores = np.full(len(cands), -np.inf)
    for c in shortlist:
        i, j = I[c], J[c]
        outcomes = [(1.0, p_real[c])] if kind == "binary" else \
                   [(GRAPH["w_in"], p_in_real[c]), (GRAPH["w_out"], p_out_real[c])]
        total = 0.0
        for w, pw in outcomes:
            M2 = M.copy()
            M2[i, j] = M2[j, i] = edge_entry(kind, w)
            total += pw * (self_estimated_accuracy(solve_lp(M2, env.y, env.train_idx, k), unlabelled) - base)
        scores[c] = total
    return scores


def run(kind, strategy, train_frac, seed):
    y, A_bin, A_w, X = make_weighted_csbm(**GRAPH, seed=seed)
    env = WeightedGraphEnv(y, A_bin, A_w, X, observe_frac=0.15, train_frac=train_frac,
                           val_frac=VAL_FRAC, seed=seed)
    k = GRAPH["k"]

    def test_acc(A_b, A_ww):
        F = solve_lp(classifier_graph(A_b, A_ww, kind), y, env.train_idx, k)
        return float((predict(F)[env.test_idx] == y[env.test_idx]).mean())

    ceiling = test_acc(A_bin, A_w)
    e0 = int(env.A_obs.sum() / 2)
    out = {0: (test_acc(env.A_obs, env.A_obs_weighted), 0)}
    for step in range(1, BUDGET + 1):
        cands = env.candidates()
        if strategy == "random":
            s = env.rng.random(len(cands))
        elif strategy == "common_nbrs":
            I, J = np.array(cands).T
            s = (env.A_obs @ env.A_obs)[I, J]
        else:
            s = value_scores(env, kind, cands, k)
        tied = np.flatnonzero(s == s.max())
        env.reveal(cands[int(env.rng.choice(tied))])
        if step in CHECKPOINTS:
            out[step] = (test_acc(env.A_obs, env.A_obs_weighted), int(env.A_obs.sum() / 2) - e0)
    return out, ceiling


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    p = stats.ttest_rel(a, b).pvalue if d.std() > 0 else 1.0
    return f"{d.mean():+.3f} +- {d.std(ddof=1):.3f}{' *' if p < 0.05 else '  '} (p={p:.3f})"


def main():
    print(f"graph={GRAPH}, seeds={N_SEEDS}, budget={BUDGET}, val_frac={VAL_FRAC}, shortlist={SHORTLIST}",
          flush=True)
    res = {}
    for scen, tf in LABELS.items():
        res[scen] = {}
        for kind in CLASSIFIERS:
            for strat in STRATEGIES:
                runs = [run(kind, strat, tf, s) for s in range(N_SEEDS)]
                key = f"{kind}/{strat}"
                res[scen][key] = {
                    "acc": {cp: [r[0][cp][0] for r in runs] for cp in CHECKPOINTS},
                    "edges": {cp: [r[0][cp][1] for r in runs] for cp in CHECKPOINTS},
                    "ceiling": [r[1] for r in runs],
                }
                a = res[scen][key]["acc"]
                print(f"  {scen:>9} {key:>20}: acc " +
                      " ".join(f"b{cp}={np.mean(a[cp]):.3f}" for cp in CHECKPOINTS) +
                      f"  edges@{BUDGET}={np.mean(res[scen][key]['edges'][BUDGET]):5.1f}"
                      f"  ceiling={np.mean(res[scen][key]['ceiling']):.3f}", flush=True)

    with open(os.path.join(HERE, "results", "value_scoring_results.json"), "w") as f:
        json.dump(res, f, indent=1)

    for scen in LABELS:
        r = res[scen]
        print(f"\n=== {scen}: paired differences in test accuracy, {N_SEEDS} seeds ===")
        for cp in [25, 50, 100]:
            print(f"-- budget {cp}")
            for kind in CLASSIFIERS:
                print(f"  {kind:>8}: value - random      {paired(r[f'{kind}/value']['acc'][cp], r[f'{kind}/random']['acc'][cp])}")
                print(f"  {kind:>8}: value - common_nbrs {paired(r[f'{kind}/value']['acc'][cp], r[f'{kind}/common_nbrs']['acc'][cp])}")
            for kind in ["weighted", "signed"]:
                for strat in ["random", "value"]:
                    print(f"  {kind:>8} - binary ({strat:>6}) {paired(r[f'{kind}/{strat}']['acc'][cp], r[f'binary/{strat}']['acc'][cp])}")
    print("\n* = paired t-test p < 0.05")


if __name__ == "__main__":
    main()
