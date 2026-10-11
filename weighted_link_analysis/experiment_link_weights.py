"""Experiment: do homophily edge weights help edge discovery, and does it depend on labels?

Three binary/weighted scorer pairs; within a pair the only difference is whether the
scorer reads edge weights. Metrics at each checkpoint, paired across seeds:
  - edges: real edges found by the queries (gain over the initial observed graph)
  - acc:   test accuracy of binary label propagation on the observed graph, i.e. whether
           the chosen edges actually carry label information to unlabelled nodes
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from labelprop import accuracy as lp_accuracy
from weighted_data import make_weighted_csbm
from weighted_env import WeightedGraphEnv
import weighted_scorers as ws

N_SEEDS = 10
BUDGET = 100
CHECKPOINTS = [10, 25, 50, 100]
GRAPH = dict(n=90, k=3, p_in=0.25, p_out=0.02, w_in=0.8, w_out=0.2)
LABEL_SCENARIOS = {"no_labels": 0.0, "sparse_15": 0.15, "medium_40": 0.40, "dense_80": 0.80}

SCORERS = {
    "random": ws.random_scorer,
    "structural": ws.structural_scorer,
    "w_structural": ws.weighted_structural_scorer,
    "adamic_adar": ws.adamic_adar_scorer,
    "w_adamic_adar": ws.weighted_adamic_adar_scorer,
    "labelprop": ws.labelprop_scorer,
    "w_labelprop": ws.weighted_labelprop_scorer,
}
PAIRS = [("structural", "w_structural"), ("adamic_adar", "w_adamic_adar"), ("labelprop", "w_labelprop")]


def run(scorer, train_frac, seed):
    y, A_bin, A_w, X = make_weighted_csbm(**GRAPH, seed=seed)
    env = WeightedGraphEnv(y, A_bin, A_w, X, observe_frac=0.15, train_frac=train_frac,
                           val_frac=0.0, seed=seed)
    has_labels = len(env.train_idx) > 0

    def snapshot():
        e = int(env.A_obs.sum() / 2)
        acc = lp_accuracy(env.A_obs, y, env.train_idx, env.test_idx)[0] if has_labels else np.nan
        return e, acc

    e0, _ = snapshot()
    out = {}
    for step in range(1, BUDGET + 1):
        cands = env.candidates()
        s = scorer(env, cands)
        tied = np.flatnonzero(s == s.max())
        env.reveal(cands[int(env.rng.choice(tied))])
        if step in CHECKPOINTS:
            e, acc = snapshot()
            out[step] = (e - e0, acc)
    return out


def main():
    print(f"graph={GRAPH}, seeds={N_SEEDS}, budget={BUDGET}", flush=True)
    res = {}
    for scen, tf in LABEL_SCENARIOS.items():
        res[scen] = {}
        for name, fn in SCORERS.items():
            runs = [run(fn, tf, s) for s in range(N_SEEDS)]
            res[scen][name] = {cp: {"edges": [r[cp][0] for r in runs], "acc": [r[cp][1] for r in runs]}
                               for cp in CHECKPOINTS}
            last = res[scen][name][BUDGET]
            print(f"  {scen:>10} {name:>14}: edges@{BUDGET}={np.mean(last['edges']):6.1f}  "
                  f"acc@{BUDGET}={np.nanmean(last['acc']) if tf else float('nan'):.3f}", flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "link_weights_results.json"), "w") as f:
        json.dump(res, f, indent=1)

    for metric in ["edges", "acc"]:
        print(f"\n=== {metric}: weighted minus binary, mean +- sd over {N_SEEDS} paired seeds ===")
        print(f"{'scenario':>10} {'pair':>12} " + " ".join(f"{'b=' + str(c):>18}" for c in CHECKPOINTS))
        for scen, tf in LABEL_SCENARIOS.items():
            if metric == "acc" and tf == 0:
                continue
            for b, w in PAIRS:
                cells = []
                for cp in CHECKPOINTS:
                    xb = np.array(res[scen][b][cp][metric], float)
                    xw = np.array(res[scen][w][cp][metric], float)
                    d = xw - xb
                    p = stats.ttest_rel(xw, xb).pvalue if d.std() > 0 else 1.0
                    cells.append(f"{d.mean():+7.2f}+-{d.std(ddof=1):5.2f}{'*' if p < 0.05 else ' '}  ")
                print(f"{scen:>10} {b:>12} " + " ".join(cells))
    print("\n* = paired t-test p < 0.05")


if __name__ == "__main__":
    main()
