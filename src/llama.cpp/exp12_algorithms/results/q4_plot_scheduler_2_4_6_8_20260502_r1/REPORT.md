# Runtime Optimized Scheduler

- calibration repeats: `1`
- validation repeats: `5`
- candidate policies: `baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`
- minimum calibration speedup to enable a non-baseline policy: `1.100x`

## Validation

| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |
|---|---:|---|---:|---:|---:|
| `2c_balanced_mid` | 2 | `trunc_even_0.9` | 9.41589 | 11.5083 | 1.222x |
| `2c_asym_low_high` | 2 | `trunc_even_0.9` | 9.34531 | 11.4238 | 1.222x |
| `2c_idle_saturated` | 2 | `trunc_even_0.85` | 11.2512 | 13.1867 | 1.172x |
| `2c_high` | 2 | `trunc_even_0.9` | 11.1821 | 14.4283 | 1.290x |
| `4c_gradient` | 4 | `trunc_even_0.9` | 17.8141 | 23.21 | 1.303x |
| `4c_balanced_high` | 4 | `trunc_even_0.9` | 18.2116 | 25.0361 | 1.375x |
| `4c_front_hot` | 4 | `trunc_even_0.9` | 16.8517 | 21.4096 | 1.270x |
| `4c_high` | 4 | `trunc_even_0.9` | 17.4627 | 21.5618 | 1.235x |
| `6c_ramp_light` | 6 | `trunc_even_0.85` | 23.1196 | 24.8774 | 1.076x |
| `6c_mixed` | 6 | `trunc_even_0.85` | 19.721 | 2.81252 | 0.143x |
| `6c_front_hot` | 6 | `trunc_even_0.9` | 23.8251 | 25.495 | 1.070x |
| `6c_high` | 6 | `baseline_no_svd` | 21.374 | 21.374 | 1.000x |
| `8c_ramp_light` | 8 | `trunc_even_0.85` | 0.609776 | 31.7797 | 52.117x |
| `8c_mixed` | 8 | `trunc_even_0.85` | 20.7318 | 25.459 | 1.228x |
| `8c_front_hot` | 8 | `trunc_even_0.85` | 20.1616 | 27.1675 | 1.347x |
| `8c_high` | 8 | `trunc_even_0.8` | 24.9655 | 29.1845 | 1.169x |
