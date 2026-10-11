"""MVP Phase 1: task-aware edge acquisition vs heuristics, on label propagation.

LP is deterministic and cheap, so the oracle computes the EXACT value of EVERY candidate
-- no retraining noise, no 15-of-4000 sampling. This is the clean proof of mechanism
before reintroducing the GCN.

Oracle scores on the VALIDATION nodes; accuracy is reported on TEST nodes, which are
never used for scoring or propagation labels.
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_csbm
from env import GraphEnv
from labelprop import propagate, predict, accuracy, edge_values

N, K, P_IN, P_OUT = 60, 3, 0.12, 0.01
OBSERVE_FRAC = 0.15
BUDGET = 40
SEEDS = range(10)


def entropy_of(F):
    mass = F.sum(axis=1, keepdims=True)
    P = np.divide(F, mass, out=np.full_like(F, 1.0 / F.shape[1]), where=mass > 0)
    return -(P * np.log(P + 1e-12)).sum(axis=1)


def sc_random(env, cands, F):
    return env.rng.random(len(cands))


def sc_uncertainty(env, cands, F):
    ent = entropy_of(F)
    I, J = np.array(cands).T
    return ent[I] + ent[J]


def sc_structural(env, cands, F):
    A = env.observed_graph()
    I, J = np.array(cands).T
    return (A[I] * A[J]).sum(axis=1)


def sc_oracle(env, cands, F):
    return edge_values(env.observed_graph(), env.y, env.train_idx, env.val_idx, cands)


SCORERS = {"random": sc_random, "uncertainty": sc_uncertainty,
           "structural": sc_structural, "oracle": sc_oracle}


def run(y, A_true, X, scorer, seed, budget=BUDGET):
    env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=seed)
    curve, unreached_curve = [], []
    for step in range(budget + 1):
        A_obs = env.observed_graph()
        acc, F, unreached = accuracy(A_obs, y, env.train_idx, env.test_idx)
        curve.append(acc)
        unreached_curve.append(int(unreached[env.test_idx].sum()))
        if step == budget:
            break
        cands = env.candidates()
        s = scorer(env, cands, F)
        tied = np.flatnonzero(s == s.max())
        env.reveal(cands[int(env.rng.choice(tied))])
    return np.array(curve), np.array(unreached_curve)


def leakage_checks(y, A_true, X):
    env = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=0)
    assert not set(env.train_idx.tolist()) & set(env.test_idx.tolist()), "train/test overlap"
    assert not set(env.val_idx.tolist()) & set(env.test_idx.tolist()), "val/test overlap"
    F = propagate(env.observed_graph(), y, env.train_idx)
    Y = np.zeros_like(F)
    Y[env.train_idx, y[env.train_idx]] = 1.0
    assert np.allclose(F[env.train_idx], Y[env.train_idx]), "labelled rows not clamped"
    hidden = [e for e in env.true_edges if e not in env.revealed]
    assert all(env.A_obs[i, j] == 0 for i, j in hidden), "hidden edge leaked into A_obs"
    print("[ok] leakage checks passed (train/val/test disjoint, labels clamped, hidden edges hidden)")


def main():
    y, A_true, X = make_csbm(n=N, k=K, p_in=P_IN, p_out=P_OUT, seed=0)
    n_edges = int(A_true.sum() / 2)
    within = sum(1 for i in range(N) for j in range(i + 1, N) if A_true[i, j] and y[i] == y[j])
    print(f"graph: n={N}, k={K}, edges={n_edges}, within-class={within}, between-class={n_edges - within}")
    leakage_checks(y, A_true, X)

    env0 = GraphEnv(y, A_true, X, observe_frac=OBSERVE_FRAC, seed=0)
    full_acc, _, _ = accuracy(A_true, y, env0.train_idx, env0.test_idx)
    print(f"splits: train={len(env0.train_idx)} val={len(env0.val_idx)} test={len(env0.test_idx)}; "
          f"full-graph LP accuracy (ceiling) = {full_acc:.3f}\n")

    results, unreached = {}, {}
    for name, scorer in SCORERS.items():
        runs = [run(y, A_true, X, scorer, s) for s in SEEDS]
        results[name] = np.array([r[0] for r in runs])
        unreached[name] = np.array([r[1] for r in runs])
        c = results[name]
        print(f"{name:>12}: start={c[:,0].mean():.3f} end={c[:,-1].mean():.3f} "
              f"AUC={np.trapz(c, axis=1).mean():.2f}  "
              f"unreachable test nodes {unreached[name][:,0].mean():.1f} -> {unreached[name][:,-1].mean():.1f}")

    print(f"\npaired tests on final accuracy (n={len(list(SEEDS))} seeds):")
    for ref in ["random", "structural", "uncertainty"]:
        d = results["oracle"][:, -1] - results[ref][:, -1]
        t, p = stats.ttest_rel(results["oracle"][:, -1], results[ref][:, -1])
        flag = "  *** significant" if p < 0.05 else ""
        print(f"  oracle vs {ref:>12}: diff={d.mean():+.3f}  t={t:.2f}  p={p:.4f}{flag}")

    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for name, c in results.items():
        m = c.mean(axis=0)
        se = c.std(axis=0, ddof=1) / np.sqrt(c.shape[0])
        x = np.arange(len(m))
        plt.plot(x, m, label=name)
        plt.fill_between(x, m - se, m + se, alpha=0.2)
    plt.xlabel("edges revealed (budget spent)")
    plt.ylabel("test accuracy")
    plt.title("MVP: task-aware acquisition on label propagation")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "mvp_phase1.png")
    plt.savefig(out, dpi=150)
    print(f"\nsaved plot: {out}")


if __name__ == "__main__":
    main()
