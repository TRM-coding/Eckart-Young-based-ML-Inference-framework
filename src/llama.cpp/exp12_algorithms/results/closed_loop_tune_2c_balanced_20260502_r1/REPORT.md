# Runtime Optimized Scheduler

- calibration repeats: `2`
- validation repeats: `7`
- candidate policies: `baseline_no_svd,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.050x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `2c_balanced_mid` | 2 | `trunc_even_0.8` | 11.4893 | 12.2154 | 1.063x |
