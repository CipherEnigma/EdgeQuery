# Next Plans — Roadmap to a Confirmed Result

This file tracks what's left to do, in priority order, to turn the current "real but
unconfirmed, modest effect" into either a defensible confirmed finding or an honest
negative result. See `experiments/E0.md` through `E3.md` for full methodology and
results behind everything referenced here.

## Priority 1 — Statistical confirmation (blocking everything else)

**What:** Rerun the E3 big-budget experiment (`experiments/e3_big_budget.py`) at more
seeds (8-10 instead of 3), with a paired significance test on top.

**Why it's first:** every result so far — structural's edge-acquisition advantage, its
lower variance, uncertainty's negative gap — is based on only 3 seeds. None of it is
confirmed. This is the cheapest, highest-value thing to do next because it doesn't
require building anything new, just running longer.

**How:** the code for this was already written once (10 seeds + `scipy.stats.ttest_rel`,
paired per-seed on `final_edges`, `final_acc`, and `auc`, comparing each strategy against
`random`) and then intentionally reverted before completion, to bring 3-seed results to
a professor check-in first. Re-apply that change and let it run to completion (~20
minutes at 10 seeds, 3 scorers, budget=200).

**Why paired, specifically:** each seed gives every strategy the identical starting
graph slice and identical model initialization — so `structural`'s result at seed 3 and
`random`'s result at seed 3 aren't independent draws, they're a matched pair differing
only in which edges got queried. A paired t-test (or Wilcoxon signed-rank, if the
differences don't look normal) is more statistically powerful than comparing unpaired
means, and is the honest way to use this structure.

**Success criterion:** structural's advantage over random on final real-edge count
and/or AUC reaches p < 0.05. If it doesn't even at 10 seeds, that itself is worth
reporting — it would mean the effect, while directionally consistent, isn't yet
distinguishable from noise at this scale.

## Priority 2 — Give oracle_voi a fair shot

**What:** run `oracle_voi_scorer` at a budget and seed count actually comparable to what
`structural` and `random` got, instead of the compute-starved settings it's been stuck
with since E1 (`n_sample=15`, `epochs=25`, only ever tested at budget=15-20).

**Why:** oracle_voi is supposed to be the theoretical best-case strategy. Every result so
far shows it barely beating random, but every one of those results was measured under a
compute budget that was explicitly acknowledged as too small to be a fair test (see
`E1.md`'s "Known limitations" and `E2.md`'s discrepancy note: the *diagnosed* problem used
`n_sample=40`, but the actual formal script only ever used `n_sample=15`). This has never
actually been resolved — only theorized about.

**How:** because each oracle_voi query requires a full model retrain per candidate, it
cannot share a 200-query, 10-seed run with the cheap strategies without becoming
prohibitively slow. Scope it down deliberately: e.g. a smaller budget (30-50 queries) but
with a properly-sized candidate shortlist (`n_sample` large enough to contain most of the
real structural signal — the E2 diagnostic found ~40-100 candidates have nonzero
structural score once the graph has some density) and enough epochs per probe (closer to
the main loop's 100, not 25) to reduce measurement noise. Expect this single run to take
significantly longer than the cheap-strategy runs; budget the time accordingly or run
overnight/in background.

## Priority 3 — Fix the tie-breaking bug

**What:** `run_loop` (in `src/loop.py`) picks the best-scoring candidate via
`candidates[int(np.argmax(scores))]`. When multiple candidates tie for the top score
(common, since `structural_scorer` returns many exact-zero ties, and `oracle_voi`'s
validation-accuracy scores tie often too), `np.argmax` deterministically returns the
*first* one in array order — which systematically favors low node-index pairs. This was
identified during E1's diagnosis (candidates like `(0, 33)`, `(0, 56)` kept getting
picked, not because they were meaningfully chosen, but because of array-order artifacts)
and never fixed.

**Why it matters:** this isn't just a cosmetic issue — it means some of what looked like
"strategy behavior" in earlier diagnostics may have partly been this artifact instead.
It should be fixed and results re-checked before trusting any fine-grained claims about
*which* candidates a strategy tends to pick.

**How:** replace `int(np.argmax(scores))` with a random tiebreak — e.g. find all indices
achieving the max score, then pick one uniformly at random via the environment's own
seeded `rng` (for reproducibility).

## Priority 4 — Diagnose uncertainty's specific failure

**What:** E3 found `uncertainty_scorer` acquires exactly as many real edges as `random`
(no quantity disadvantage) but ends up with meaningfully worse accuracy (-4.25 AUC gap).
Same amount of information, worse outcome — that's a specific, answerable question, not
just "it didn't work."

**Why:** if uncertainty is systematically picking real edges that are *less useful* than
random ones (e.g. edges connecting nodes that were already easy to classify, or edges
between already-confident-but-wrong regions), that's an interesting, explainable result
worth reporting rather than a shrug. It could also reveal a subtler bug.

**How:** for a handful of seeds, log every real edge uncertainty acquires alongside every
real edge random acquires, and compare: are uncertainty's edges disproportionately
connecting nodes that are already correctly classified (redundant information)? Are they
disproportionately involving the small number of structurally well-connected "hub" nodes
in a way that doesn't help the sparser regions of the graph? This requires new
instrumentation, not just a parameter change.

## Priority 5 — Try better link-prediction heuristics

**What:** replace or supplement `structural_scorer`'s raw shared-neighbor *count* with
established link-prediction scores that are known to behave better in sparse graphs:
- **Adamic-Adar index** — weights shared neighbors by how rare/exclusive they are
  (a shared neighbor who has very few other connections counts more than one who's
  connected to everyone), which is designed specifically to be more informative than a
  raw count in sparse regimes.
- **Jaccard similarity** — shared neighbors as a *fraction* of each node's total
  neighbors, rather than a raw count, which could behave differently as the graph
  densifies over a long budget.

**Why:** E3 found structural's raw-count approach has a "diminishing returns" problem —
it exploits a small pool of obviously-good candidates early, then has nothing left to
say about the rest. A more nuanced scoring function might extract useful signal from a
wider set of candidates for longer, sustaining an advantage across a bigger budget instead
of front-loading it.

**How:** implement `adamic_adar_scorer` and `jaccard_scorer` in `src/scorers.py`
following the exact same interface as `structural_scorer`, and re-run the E3-style
big-budget comparison with all of them included.

## Priority 6 — Hybrid scorer (only after Priority 5 is tried)

**What:** a scorer that uses `structural` (or a Priority-5 alternative) while it still has
real signal, then falls back to something else — `uncertainty` or `random` — once the
shortlist of structurally-promising candidates runs dry.

**Why:** this directly targets the mechanism E3 identified for why structural's advantage
shrinks over a long budget, rather than hoping a better heuristic alone fixes it.

**How:** track, at each step, how many candidates currently have nonzero structural
score. Once that count drops below some threshold (needs tuning — maybe track this as a
diagnostic first before deciding a cutoff), switch the scoring function for the rest of
the budget. Only worth building once Priority 5 has been tried and, if it still isn't
enough on its own, this becomes the natural next lever.

## Priority 7 — Map budget x observe_frac jointly

**What:** E2 swept `observe_frac` at a fixed small budget (15) and found it didn't cleanly
explain the effect. E3 raised the budget at a fixed `observe_frac` (0.1) and found the
advantage shrinks with budget. These are two views of the same underlying 2D surface
(advantage as a function of both starting density *and* budget) that has never been
mapped jointly.

**Why:** the "real" operating advantage of structural querying might peak at some
combination of the two that neither single-axis sweep would reveal — e.g. maybe a
moderate `observe_frac` (~0.15) with a moderate budget (~75) is actually the best
regime, and neither of the two experiments run so far would have found it.

**How:** only worth doing after Priorities 1-2 give a statistically confirmed baseline —
otherwise this is expensive compute spent mapping noise. Once justified, a modest grid
(e.g. 3 observe_frac values x 3 budget values x enough seeds for significance) using the
`normalized_advantage` metric from E2, plotted as a heatmap.

## Housekeeping (not experiments, but blocking a clean writeup)

- **`notes.md`'s Part 12** still reflects the very first, since-corrected E1 result at
  `observe_frac=0.1` with the buggy oracle_voi — needs to be rewritten or replaced with
  pointers to `E0.md`-`E3.md` once the story stabilizes further.
- **Nothing has been committed since Steps 1-4** (`scorers.py`, `eval.py`, all four
  experiment scripts, the `loop.py`/`env.py` changes, and all of `E0.md`-`E3.md` /
  `next_plans.md` are still uncommitted). Worth committing in logical chunks (e.g. "add
  scorers + eval + E0/E1 experiments", then "diagnose and document cold-start/ceiling
  effects", then "add E3 big-budget experiment") rather than one giant commit, given how
  much investigation happened between them.

## Open questions worth raising with the professor now, given where things stand

- Is a paired significance test (Priority 1) the right bar, or does a project like this
  usually want something else (e.g. a permutation test, given the small sample size)?
- Given oracle_voi hasn't had a fair run yet, is it reasonable to present structural's
  result as "best strategy so far" with that caveat explicit, or should the writeup wait
  until oracle_voi's fair run (Priority 2) is done first?
- Is the diminishing-advantage-with-budget finding (Priority 5's motivation) itself an
  interesting result worth reporting on its own, independent of whether a fix is found?
