# weighted_link_analysis

Self-contained experiment: do homophily edge weights (0.8 within a class, 0.2 across classes) help budgeted
edge querying, and does that depend on label density? Reuses `src/` (`data`, `env`, `labelprop`).

- `weighted_data.py`: `make_weighted_csbm` (homophily weights), `make_reliability_weighted_csbm` (noisy weights, not used yet)
- `weighted_env.py`: `WeightedGraphEnv`, which tracks `A_obs_weighted` alongside `A_obs`
- `weighted_scorers.py`: binary/weighted scorer pairs (common neighbours, Adamic–Adar, label propagation)
- `experiment_link_weights.py`: Exp 1, scorers that read weights (4 label densities × 7 scorers × 10 seeds, about 3 min)
- `experiment_value_scoring.py`: Exp 2, value-of-query scoring × binary/weighted/signed classifiers (about 10 min)
- `results/`: JSON + console log
- `PROFESSOR_EXPLANATION.md`: setup, results, interpretation
