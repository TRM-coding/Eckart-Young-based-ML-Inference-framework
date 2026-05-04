# Runtime Optimized Scheduler

- calibration repeats: `2`
- validation repeats: `5`
- candidate policies: `baseline_no_svd,trunc_even_0.5,trunc_even_0.6,trunc_even_0.7,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.050x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `2c_balanced_mid` | 2 | `trunc_even_0.8` | 11.9518 | 13.0128 | 1.089x |
| `4c_gradient` | 4 | `trunc_even_0.6` | 18.3049 | 21.0806 | 1.152x |
