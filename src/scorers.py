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


def adamic_adar_scorer(env, candidates, model=None, X=None, A_norm=None):
    """Like structural_scorer, but each shared neighbour counts more if it has FEW other
    connections (1/log(degree)) and less if it's a high-degree hub connected to everyone.
    A hub being a shared neighbour of (i, j) is weak evidence they're related -- it's
    likely connected to i and j for unrelated reasons. A low-degree shared neighbour is
    much stronger evidence, since it's choosier about who it connects to. Model-free, same
    interface as structural_scorer, so it slots into the same cheap no-training simulation."""
    A_obs = env.observed_graph()
    degree = A_obs.sum(axis=1)
    weight = np.where(degree > 1, 1.0 / np.log(degree, where=degree > 1, out=np.ones_like(degree)), 0.0)
    scores = np.zeros(len(candidates))
    for idx, (i, j) in enumerate(candidates):
        shared = (A_obs[i] > 0) & (A_obs[j] > 0)
        scores[idx] = weight[shared].sum()
    return scores


def jaccard_scorer(env, candidates, model=None, X=None, A_norm=None):
    """Shared neighbours as a FRACTION of (i, j)'s combined neighbourhood, rather than a
    raw count. Two low-degree nodes sharing 1 neighbour is much stronger evidence than two
    high-degree nodes sharing 1 neighbour out of many -- structural_scorer can't tell these
    apart, since both score exactly 1. Model-free, same interface as structural_scorer."""
    A_obs = env.observed_graph()
    scores = np.zeros(len(candidates))
    for idx, (i, j) in enumerate(candidates):
        ni = A_obs[i] > 0
        nj = A_obs[j] > 0
        union = (ni | nj).sum()
        scores[idx] = (ni & nj).sum() / union if union > 0 else 0.0
    return scores


def label_agreement_scorer(env, candidates, model=None, X=None, A_norm=None):
    """ALPINE-style max-prob (Chen et al. 2020), with the link probability derived from
    the node-classification side instead of a network embedding. Under homophily,
    P(link) is increasing in P(same class), so ranking by P(same class) ranks by P(link)
    without needing p_in/p_out. P(same class) = sum_c q_i(c) q_j(c), where q is one-hot
    for labelled (train) nodes and uniform otherwise -- only train labels are used, never
    val/test. Ties (e.g. all unlabelled pairs at 1/k) are broken by shared-neighbour
    count, so off the labelled pairs this falls back to structural_scorer.

    v1 of this scorer (still spending nearly the whole budget confirming edges between
    two ALREADY-labelled nodes -- checked empirically: 199/200 queries at budget=200) --
    those pairs score highest (both one-hot, dot product = 1) but are close to useless:
    both endpoints' classes are already known from the labels directly, so a new edge
    between them can't inform any unlabelled node it doesn't already touch. Both-labelled
    pairs are now excluded so the budget is forced toward pairs that can actually carry
    label information to someone whose class is still in question."""
    A_obs = env.observed_graph()
    n = A_obs.shape[0]
    k = int(env.y.max()) + 1
    train_set = set(env.train_idx.tolist())
    q = np.full((n, k), 1.0 / k)
    q[env.train_idx] = 0.0
    q[env.train_idx, env.y[env.train_idx]] = 1.0
    I, J = np.array(candidates).T
    same_class = (q[I] * q[J]).sum(axis=1)
    shared = (A_obs[I] * A_obs[J]).sum(axis=1)
    both_labelled = np.array([(i in train_set) and (j in train_set) for i, j in candidates])
    same_class = np.where(both_labelled, 0.0, same_class)
    return same_class * 1000.0 + shared


def max_degree_scorer(env, candidates, model=None, X=None, A_norm=None):
    """ALPINE's max-deg (Chen et al. 2020): u(i, j) = observed degree of i + of j."""
    deg = env.observed_graph().sum(axis=1)
    I, J = np.array(candidates).T
    return deg[I] + deg[J]


def pagerank_scorer(env, candidates, model=None, X=None, A_norm=None, damping=0.85, iters=100):
    """ALPINE's page-rank: u(i, j) = PR_i + PR_j on the observed graph (unknown pairs
    treated as disconnected, as in the paper). Isolated nodes teleport uniformly."""
    A = env.observed_graph().astype(np.float64)
    n = A.shape[0]
    deg = A.sum(axis=1)
    dangling = deg == 0
    P = np.divide(A, deg[:, None], out=np.zeros_like(A), where=~dangling[:, None])
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        pr = damping * (pr @ P + pr[dangling].sum() / n) + (1 - damping) / n
    I, J = np.array(candidates).T
    return pr[I] + pr[J]


def entropy_drop_scorer(env, candidates, model, X, A_norm):
    """Cheap, smooth alternative to oracle_voi_scorer: for every candidate, do a single
    forward pass (no retraining, frozen weights) with the edge tentatively added, and
    score by how much predictive entropy drops across the edge's 2-hop neighbourhood --
    the set of nodes whose representation could possibly change in a 2-layer GCN. Needs
    no ground-truth labels at all (unlike oracle_voi_scorer, which needs val_idx), and
    is cheap enough to score every candidate rather than a small shortlist. Deliberately
    *not* weighted by structural_scorer's shared-neighbour count: that signal is zero for
    ~99% of candidates early on, so multiplying by it would zero out this scorer's own
    advantage during exactly the cold-start phase it's meant to help with."""
    model.eval()
    A_obs = env.observed_graph()

    with torch.no_grad():
        baseline_entropy = entropy_uncertainty(model, X, A_norm)

        scores = np.zeros(len(candidates))
        for idx, (i, j) in enumerate(candidates):
            neighbors_i = np.where(A_obs[i] > 0)[0]
            neighbors_j = np.where(A_obs[j] > 0)[0]
            affected = np.unique(np.concatenate([[i, j], neighbors_i, neighbors_j]))

            A_temp = A_obs.copy()
            A_temp[i, j] = 1.0
            A_temp[j, i] = 1.0
            A_norm_temp = torch.tensor(normalize_adj(A_temp), dtype=torch.float32)

            new_entropy = entropy_uncertainty(model, X, A_norm_temp)
            drop = (baseline_entropy - new_entropy)[affected].mean().item()
            scores[idx] = drop

    return scores


def oracle_voi_scorer(env, candidates, model, X, A_norm, n_sample=40, epochs=25, hidden_dim=16, lr=0.01, seed=0):
    """Gold-standard, expensive scorer: tentatively add a candidate edge, retrain from
    scratch, and score by the resulting validation accuracy. Only n_sample candidates get
    this expensive treatment each step -- a random subset would almost never contain a real
    edge (real edges are ~3.7% of ~4000 candidates), so instead we shortlist the n_sample
    candidates with the highest structural_scorer (shared-neighbour) score first, then spend
    the retrain budget on that shortlist. Un-shortlisted candidates get -inf so they're never
    picked over an evaluated one."""
    n_candidates = len(candidates)
    if n_candidates > n_sample:
        structural_scores = structural_scorer(env, candidates)
        sample_pos = np.argsort(structural_scores)[::-1][:n_sample]
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
