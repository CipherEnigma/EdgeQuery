# Phase 1 — Foundation Result

## Result

On the locked dataset, `structural` acquires real edges significantly faster than
`random`. Confirmed at 50 seeds, paired t-test, significant at every budget checkpoint:

| budget | random gain | structural gain | diff | p |
|---|---|---|---|---|
| 10 | 0.28 | 0.98 | +0.70 | 0.0001 |
| 15 | 0.40 | 1.28 | +0.88 | <0.0001 |
| 25 | 0.78 | 1.94 | +1.16 | <0.0001 |
| 50 | 1.82 | 2.74 | +0.92 | 0.0036 |
| 100 | 3.68 | 5.04 | +1.36 | 0.0021 |
| 200 | 7.38 | 9.24 | +1.86 | 0.0006 |

The advantage is real but small in absolute terms (~2 extra real edges on top of ~33),
too small to move test accuracy on a 36-node test set — which is why downstream accuracy
tied (`uncertainty` also never differed from random; best p=0.42).

## How we got here

1. `phase1_confirmatory.py` (10 seeds, budget 200, full training loop): no significant
   difference on any metric (edges p=0.44, accuracy p=0.74, AUC p=0.73).
2. `diag_foundation.py` tested three explanations:
   - **Underpowered test (regression to the mean): confirmed.** The effect (+1.86 edges)
     is about half a standard deviation of the per-seed difference (sd 3.6); 10 seeds
     detect it only ~30% of the time. Earlier "strong" results were never significant
     (E2 p=0.08, E3 p=0.096); 3-seed subsets of the same data range from −5.7 to +9.3.
   - **Density retune diluted the signal: refuted.** The current dataset gives a
     stronger effect than the original (budget 15: t=4.89 vs 1.82).
   - **Large budget dilutes the effect: partly.** Absolute gap grows with budget, but
     the relative rate advantage drops from 3.2x (budget 15) to 1.25x (budget 200).

## Locked settings

- **Dataset:** `p_in=0.13`, `p_out=0.006`, `n=90`, `k=3`, `feat_signal=0.10`.
- **Environment:** random initial reveal, `observe_frac=0.15`. (Coverage-first reveal
  was tried and reverted — it destroyed the overlapping neighbourhoods structural needs.)
- **Tie-breaking:** random among tied top scores (fixed; `argmax` previously biased
  picks toward low node indices).
- **Seeds:** at least 30–50 for any claimed result. Fewer is exploration only.
- **Primary metric:** real edges acquired, headline at budget 25 (highest t). Full
  curve reported; accuracy is secondary until a scorer's edge advantage is large enough
  to move it.
- **Model-free scorers** (random, structural) are simulated without GCN training —
  the edge sequence is identical, verified by reproducing Phase 1 seeds 0–9 exactly.

## What this sets up

Structural finds edges faster, but on this dataset every real edge is worth roughly the
same, so finding a couple more changes nothing downstream. Layer 2 makes edge value
uneven (anchor labelling) so that *which* edge you find matters, not just how many.
