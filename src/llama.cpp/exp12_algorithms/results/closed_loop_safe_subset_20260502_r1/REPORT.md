# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.050x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `2c_balanced_mid` | 2 | `trunc_even_0.8` | 11.5044 | 11.1746 | 0.971x |
| `2c_asym_low_high` | 2 | `trunc_even_0.8` | 11.6182 | 12.9253 | 1.113x |
| `2c_idle_saturated` | 2 | `trunc_even_0.8` | 11.8492 | 13.1785 | 1.112x |
| `2c_high` | 2 | `trunc_even_0.85` | 11.7394 | 13.3147 | 1.134x |
| `4c_gradient` | 4 | `trunc_even_0.8` | 16.5927 | 21.6125 | 1.303x |
| `4c_balanced_high` | 4 | `trunc_even_0.85` | 20.576 | 23.6089 | 1.147x |
| `4c_front_hot` | 4 | `trunc_even_0.8` | 19.65 | 22.8659 | 1.164x |
| `4c_high` | 4 | `trunc_even_0.8` | 19.9995 | 22.0429 | 1.102x |
| `8c_mixed` | 8 | `trunc_even_0.9` | 0.275754 | 4.90434 | 17.785x |
| `8c_front_hot` | 8 | `baseline_no_svd` | 27.1387 | 27.1387 | 1.000x |
| `8c_high` | 8 | `trunc_even_0.8` | 30.0372 | 33.3246 | 1.109x |
