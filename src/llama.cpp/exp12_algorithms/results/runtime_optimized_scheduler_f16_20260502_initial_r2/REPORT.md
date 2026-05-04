# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `3`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `trunc_even_0.8` | 7.07387 | 9.47808 | 1.340x |
| `4c_mixed` | 4 | `trunc_even_0.8` | 7.16876 | 9.7947 | 1.366x |
| `4c_front_hot` | 4 | `edge_end` | n/a | n/a | n/a |
| `4c_high` | 4 | `trunc_even_0.8` | 6.99559 | 9.63813 | 1.378x |
| `6c_ramp_light` | 6 | `trunc_even_0.85` | 3.41512 | 12.9045 | 3.779x |
| `6c_mixed` | 6 | `edge_end` | n/a | n/a | n/a |
| `6c_front_hot` | 6 | `edge_end` | n/a | n/a | n/a |
| `6c_high` | 6 | `trunc_even_0.8` | 8.90009 | 12.2892 | 1.381x |
| `8c_ramp_light` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_mixed` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_front_hot` | 8 | `edge_end` | n/a | n/a | n/a |
| `8c_high` | 8 | `trunc_even_0.85` | 9.43409 | 13.0068 | 1.379x |
