# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `trunc_even_0.9` | 7.09915 | 10.0269 | 1.412x |
| `4c_mixed` | 4 | `trunc_even_0.85` | 7.00646 | 9.52788 | 1.360x |
| `4c_front_hot` | 4 | `trunc_even_0.9` | 7.11168 | 10.2162 | 1.437x |
| `4c_high` | 4 | `trunc_even_0.9` | 6.90328 | 10.1734 | 1.474x |
| `6c_ramp_light` | 6 | `trunc_even_0.9` | 8.10909 | 13.2671 | 1.636x |
| `6c_mixed` | 6 | `trunc_even_0.9` | 0.484998 | 11.8957 | 24.527x |
| `6c_front_hot` | 6 | `trunc_even_0.8` | 8.34625 | 11.829 | 1.417x |
| `6c_high` | 6 | `trunc_even_0.9` | 8.97646 | 13.6248 | 1.518x |
| `8c_ramp_light` | 8 | `trunc_even_0.9` | 10.2935 | 14.2382 | 1.383x |
| `8c_mixed` | 8 | `trunc_even_0.9` | 9.25494 | 13.52 | 1.461x |
| `8c_front_hot` | 8 | `trunc_even_0.9` | 9.28262 | 13.4633 | 1.450x |
| `8c_high` | 8 | `trunc_even_0.85` | 9.55915 | 13.9039 | 1.455x |
