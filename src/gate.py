"""Strict promotion gate for acquisition scorers.

Why this exists: our previous gate (diagnostics.quick_diagnostic) scored a strategy by
REAL EDGES FOUND. That metric cannot distinguish a useful edge from a useless one, and
it certified label_agreement_scorer v1, which spent 199/200 queries confirming edges
between two ALREADY-LABELLED nodes -- unbeatable on edge count, worthless for
classification, since both endpoints' classes are already given by the labels.

So the gate measures what the downstream task can actually use. With a 2-layer GCN a
test node's prediction depends on its 2-hop neighbourhood, so label information reaches
it only if some labelled node sits within 2 hops. The headline metric is therefore

    reach = #{test nodes within 2 hops of any labelled node in the observed graph}

A scorer is PROMOTED only if, against random on paired seeds:
  1. waste: <10% of its queries are on pairs whose endpoints are both already labelled
     (such a query cannot change any test node's reach), AND
  2. reach: it does not significantly REDUCE reach, AND
  3. reach: it significantly INCREASES reach (p<0.05) -- this is the actual bar.

Edge count is still reported, but it is diagnostic only and never decides promotion.
"""
import numpy as np
from scipy import stats
from scipy.sparse.csgraph import shortest_path

from env import GraphEnv


def label_reach(A_obs, train_idx, test_idx, hops=2):
    """#test nodes within `hops` of any labelled node -- the nodes a 2-layer GCN can
    actually propagate label information to."""
    d = shortest_path(A_obs, unweighted=True, indices=train_idx).min(axis=0)
    return int((d[test_idx] <= hops).sum())


def run_scorer(y, A_true, X, scorer, budget, observe_frac, seed):
    """One acquisition run. Returns (reach gained, real edges gained, waste fraction)."""
    env = GraphEnv(y, A_true, X, observe_frac=observe_frac, seed=seed)
    train_set = set(env.train_idx.tolist())
    reach_0 = label_reach(env.A_obs, env.train_idx, env.test_idx)
    edges_0 = int(env.A_obs.sum() / 2)
    wasted = 0
    for _ in range(budget):
        cands = env.candidates()
        s = scorer(env, cands, None, None, None)
        tied = np.flatnonzero(s == s.max())
        i, j = cands[int(env.rng.choice(tied))]
        if i in train_set and j in train_set:
            wasted += 1
        env.reveal((i, j))
    return (label_reach(env.A_obs, env.train_idx, env.test_idx) - reach_0,
            int(env.A_obs.sum() / 2) - edges_0,
            wasted / budget)


def gate(scorers, y, A_true, X, budget=100, seeds=range(30), observe_frac=0.15):
    """Run every scorer (must include 'random'), print a verdict, return pass/fail dict."""
    assert "random" in scorers, "gate needs a 'random' baseline"
    res = {name: np.array([run_scorer(y, A_true, X, fn, budget, observe_frac, s) for s in seeds])
           for name, fn in scorers.items()}

    print(f"Promotion gate | budget={budget}, {len(list(seeds))} seeds, observe_frac={observe_frac}")
    print("  metric that decides: REACH = test nodes within 2 hops of a label\n")
    print(f"{'scorer':>14} {'reach':>7} {'vs rand':>8} {'p':>8} {'edges':>7} {'waste':>7}  verdict")

    verdicts = {}
    for name, r in res.items():
        reach, edges, waste = r[:, 0], r[:, 1], r[:, 2].mean()
        if name == "random":
            print(f"{name:>14} {reach.mean():>7.2f} {'--':>8} {'--':>8} {edges.mean():>7.2f} {waste:>7.1%}  (baseline)")
            verdicts[name] = None
            continue
        diff = reach - res["random"][:, 0]
        t, p = stats.ttest_rel(reach, res["random"][:, 0])
        if waste >= 0.10:
            v = f"FAIL: {waste:.0%} of budget on both-labelled pairs"
        elif p < 0.05 and diff.mean() < 0:
            v = "FAIL: significantly worse reach than random"
        elif p < 0.05:
            v = "PROMOTE"
        else:
            v = "HOLD: no significant reach gain"
        verdicts[name] = v
        print(f"{name:>14} {reach.mean():>7.2f} {diff.mean():>+8.2f} {p:>8.4f} {edges.mean():>7.2f} {waste:>7.1%}  {v}")
    return verdicts
