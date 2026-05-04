# Model-Based Scheduler Profile Report

This profile set connects the trained sklearn latency model to `scheduler.py` by generating the existing `core_splits` JSON schema.

## Important Limitation

The current trained model was fit from fixed-Q measurements. This script therefore uses the model for core-set/load effects and applies linear scaling for layer work and SVD rate. It is a useful scheduler integration, but not yet the final Q-aware Algorithmv2 5.2 profile.

## Generated Profiles

- profiles: 4
- n_layers: 28
- rates: 0,0.3,0.5,0.8
- timeout_budget_ms: 8.0
- idle reference deadline base: 32.035800 ms
- loss source: heuristic placeholder

## Scheduler Sanity

| scenario | mode | p | clipped | max rate | loss | total ms | full local ms | est. speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| load_0 | baseline_no_svd | 0 | 0 | 0.000 | 0.000000 | 32.035800 | 32.035800 | 1.000x |
| load_50 | baseline_no_svd | 0 | 0 | 0.000 | 0.000000 | 31.880400 | 31.880400 | 1.000x |
| load_80 | baseline_no_svd | 0 | 0 | 0.000 | 0.000000 | 32.474400 | 32.474400 | 1.000x |
| load_100 | local | 7 | 14 | 0.500 | 4.015000 | 32.208125 | 37.062400 | 1.151x |
