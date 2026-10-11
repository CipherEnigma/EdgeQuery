# Link weights in budgeted edge querying: first results

## Question

When we query the oracle for hidden edges, does knowing each observed edge's **weight**
help us choose better queries? And does that help depend on how many nodes are labelled?

- **H1:** scorers that read edge weights find more real edges than the same scorers on the binary graph.
- **H2:** the benefit of weights grows with label density, because weights control how much
  label information flows along each edge.

## Setup

- **Graph:** cSBM with n=90 and k=3 classes. p_in=0.25 and p_out=0.02, giving average degree ≈ 8
  (intentionally denser than the main experiments). A new graph is drawn for each seed.
- **Weights (assigned in advance, homophily):** 0.8 for edges within a class, 0.2 for edges
  across classes. The oracle still answers yes/no; the weight of an edge is only visible
  once that edge has been observed.
- **Process:** 15% of the true edges are visible at the start. Then 100 queries, one at a time, each
  going to the top-scoring unrevealed pair.
- **Labels:** 0%, 15%, 40% or 80% of nodes are labelled. The test set is all unlabelled nodes.
- **Scorer pairs.** The two scorers in a pair are identical except that one reads weights:

  | binary | weighted twin |
  |---|---|
  | common neighbours | Σ_k w_ik · w_kj |
  | Adamic–Adar | weighted Adamic–Adar |
  | label propagation P(same class) | label propagation over the weighted graph |

  A random scorer is included as a baseline.
- **Metrics at budgets 10/25/50/100:** (a) real edges found; (b) test accuracy of label
  propagation on the observed graph. Each pair is compared with a paired t-test over 10 seeds.

## Results (10 seeds)

**Real edges found after 100 queries** (out of ~57 initially visible):

| random | common nbrs | weighted | Adamic–Adar | weighted | label prop | weighted |
|---|---|---|---|---|---|---|
| 7.7 | 16.4 | 18.7 | 17.9 | 18.8 | 16.4–19.5 | identical |

**Weighted minus binary, edges found (mean ± sd, \* p<0.05):**

| pair | b=10 | b=25 | b=50 | b=100 |
|---|---|---|---|---|
| common neighbours | +0.1 ± 1.4 | +0.5 ± 1.9 | **+2.7 ± 2.2\*** | **+2.3 ± 3.2\*** |
| Adamic–Adar | **+0.8 ± 0.6\*** | **+1.9 ± 1.1\*** | **+2.8 ± 2.0\*** | +0.9 ± 2.0 |
| label propagation | 0 | 0 | 0 | 0 |

**Test accuracy:** none of the weighted−binary differences is significant at any label
density (all within ±0.02). Also, **random querying matched or beat every scorer on
accuracy** at 100 queries (e.g. 15% labels: random 0.571 vs 0.52–0.53 for the others;
80% labels: random 0.667 vs 0.628 for label propagation).

## What this means

1. **H1 is partly supported.** Weights help the neighbourhood scorers find about 15% more real
   edges (≈ +2–3 edges out of ~16–19). The effect is small and it is significant in the middle of the budget.
2. **H1 does not carry through to classification.** Those extra edges do not raise accuracy.
   Finding more edges is not the same as finding the edges that carry useful information. This matches the theme
   of the main Phase 1 results.
3. **H2 is not supported in this form.** Label propagation on the weighted graph picks *exactly*
   the same queries as on the binary graph. The cause: propagation normalises each node's
   edges to sum to 1. In this sparse observed graph most unlabelled nodes are reached by
   only one class, so their beliefs are already one-hot whatever the weights are. In addition, a 0.2 edge
   still pushes the *wrong* class, just less strongly. Weights can only rescale a signal; they cannot flip it.
4. **Caveat on the weights themselves.** The weights are derived from the true classes, so seeing a
   weight tells you whether the edge is within a class. Even with this
   generous signal, the gain is small.

## Experiment 2: score queries by value to the classifier

**Change.** Rank each candidate by P(real) × the expected gain in the classifier's
self-estimated accuracy over all unlabelled nodes (expected error reduction). No labels beyond
the training set are used. Weights enter the **classifier** in three ways: *binary* (every edge
counts 1), *weighted* (0.8 / 0.2), and *signed* (+0.6 / −0.6, so a cross-class edge pushes
*against* sharing a class). Each classifier is paired with three query strategies: random,
common neighbours, and value. 15% and 40% labels, 10 seeds, 100 queries. Accuracy is measured on all unlabelled nodes.
Script: `experiment_value_scoring.py`.

**Test accuracy after 100 queries (start → end):**

| classifier | random | common nbrs | value | full-graph ceiling |
|---|---|---|---|---|
| 15% labels: binary | 0.504 → 0.539 | 0.505 | 0.513 | 0.817 |
| 15% labels: weighted | 0.513 → 0.560 | 0.518 | 0.548 | 0.999 |
| 15% labels: signed | **0.610** → 0.690 | 0.618 | **0.704** | 1.000 |
| 40% labels: binary | 0.624 → 0.681 | 0.654 | 0.709 | 0.998 |
| 40% labels: weighted | 0.667 → 0.724 | 0.676 | 0.774 | 1.000 |
| 40% labels: signed | **0.696** → 0.756 | 0.706 | **0.802** | 1.000 |

**Findings (paired t-tests, 10 seeds):**
1. **How the classifier uses weights matters most.** Signed beats binary by +0.11 to +0.19 at 15% labels
   and +0.07 to +0.09 at 40% (all p ≤ 0.01). The gap is there before any query: 0.610 vs 0.504.
   Plain weighted is only slightly better than binary (+0.01 to +0.07, significant in about half the comparisons).
2. **Value scoring helps only once there are enough labels and the classifier uses weights.** At 40% labels,
   value beats random by +0.044 to +0.050 for weighted and signed (p < 0.05 at budgets 50 and 100).
   It is not significant for binary. At 15% labels it does not beat random for any classifier.
3. **Common neighbours finds the most edges (16.4) but gives the lowest accuracy.** Value beats it by up to
   +0.10 (p ≤ 0.005). This confirms Experiment 1: finding edges is not the same as finding information.
4. **Major caveat:** signed weights are derived from the true classes, so every observed edge reveals
   whether its endpoints share a class. Much of the signed classifier's advantage comes from that
   leak. The experiment also assumes p_in and p_out are known.

Getting the value scorer to work required three fixes, all documented in the script:
- Scoring on 18 validation nodes overfit them, so I switched to the label-free objective.
- The shortlist needed to rank by P(real) × confidence gap. Ranking by P(real) alone, or by uncertainty, only evaluated useless pairs.
- Each node not reached by any label now keeps a fixed fallback class.

## Proposed next steps

- **Noisy weights first.** Use `make_reliability_weighted_csbm` (or flip a fraction of weights)
  so a weight is no longer a perfect class indicator. This tests whether the signed classifier's
  advantage survives without the leak.
- Estimate p_in / p_out from the observed graph instead of assuming they are known.
- Make the weights **query-dependent**, so an edge matters only if it helps the target
  nodes of the current query. This is the original idea, and it is not tested yet.
- Use more seeds (20–30). Value-vs-random differences are about 0.05 with an sd of about 0.05, which is borderline at 10 seeds.

Reproduce: `python weighted_link_analysis/experiment_link_weights.py` (~3 min). Raw numbers
are in `results/link_weights_results.json` and `results/link_weights_log.txt`.
