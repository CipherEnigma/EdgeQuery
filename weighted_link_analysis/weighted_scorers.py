"""Candidate scorers for the weighted-link experiment, vectorised over all candidates.

Each binary scorer has a weighted twin that differs ONLY in reading A_obs_weighted instead
of A_obs, so any gap between a pair is attributable to the weights alone.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from labelprop import propagate


def _ij(candidates):
    I, J = np.array(candidates).T
    return I, J


def random_scorer(env, candidates):
    return env.rng.random(len(candidates))


def _common_neighbours(A, candidates):
    I, J = _ij(candidates)
    return (A @ A)[I, J]


def _adamic_adar(A, A_bin, candidates):
    # A shared neighbour k contributes A[i,k]*A[k,j] / log(deg_k); deg from the binary graph.
    deg = A_bin.sum(axis=1)
    inv_log = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    I, J = _ij(candidates)
    return ((A * inv_log) @ A)[I, J]


def _labelprop_same_class(A, env, candidates):
    """P(i and j share a class) from label propagation over A. Both-labelled pairs are
    zeroed (their edge carries no new label information); common neighbours break ties."""
    k = int(env.y.max()) + 1
    n = A.shape[0]
    if len(env.train_idx) == 0:
        P = np.full((n, k), 1.0 / k)
    else:
        F = propagate(A, env.y, env.train_idx, k)
        mass = F.sum(axis=1, keepdims=True)
        P = np.divide(F, mass, out=np.full_like(F, 1.0 / k), where=mass > 0)
    I, J = _ij(candidates)
    same = (P[I] * P[J]).sum(axis=1)
    labelled = np.zeros(n, dtype=bool)
    labelled[env.train_idx] = True
    same[labelled[I] & labelled[J]] = 0.0
    return same + 1e-3 * _common_neighbours(env.A_obs, candidates)


def structural_scorer(env, candidates):
    return _common_neighbours(env.A_obs, candidates)


def weighted_structural_scorer(env, candidates):
    return _common_neighbours(env.A_obs_weighted, candidates)


def adamic_adar_scorer(env, candidates):
    return _adamic_adar(env.A_obs, env.A_obs, candidates)


def weighted_adamic_adar_scorer(env, candidates):
    return _adamic_adar(env.A_obs_weighted, env.A_obs, candidates)


def labelprop_scorer(env, candidates):
    return _labelprop_same_class(env.A_obs, env, candidates)


def weighted_labelprop_scorer(env, candidates):
    return _labelprop_same_class(env.A_obs_weighted, env, candidates)
