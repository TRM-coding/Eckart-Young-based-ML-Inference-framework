# Model-Based Scheduler Profile Report

This profile set connects the trained sklearn latency model to `scheduler.py` by generating the existing `core_splits` JSON schema.

## Important Limitation

The current trained model was fit from fixed-Q measurements. This script therefore uses the model for core-set/load effects and applies linear scaling for layer work and SVD rate. It is a useful scheduler integration, but not yet the final Q-aware Algorithmv2 5.2 profile.

## Generated Profiles

- profiles: 1
- n_layers: 28
- rates: 0,0.5,0.8
- timeout_budget_ms: 8.0
- idle reference deadline base: 32.035800 ms
- loss source: measured spectral residual norm

## Scheduler Sanity

| scenario | mode | p | clipped | max rate | loss | total ms | full local ms | est. speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| load_50 | local | 8 | 0 | 0.000 | 0.000000 | 31.880394 | 31.880400 | 1.000x |
