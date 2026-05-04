# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `3`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload rows are selected but not executed because this experiment does not use adb

## Result

| scenario | cores | loads | mode | M | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---:|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | 2 | `20,80` | `edge_end_no_svd` | 20 | `` | `` | 0 | 10.569 | n/a | n/a |
| `2c_balanced_mid` | 2 | `50,50` | `edge_end_no_svd` | 20 | `` | `` | 0 | 9.51928 | n/a | n/a |
| `2c_high` | 2 | `80,100` | `edge_end_no_svd` | 2 | `` | `` | 0 | 10.4069 | n/a | n/a |
| `2c_idle_saturated` | 2 | `0,100` | `edge_end_no_svd` | 20 | `` | `` | 0 | 11.7703 | n/a | n/a |
| `4c_balanced_high` | 4 | `20,40,60,80` | `local` | 28 | `60,61,62,63` | `` | 0 | 19.8716 | 19.8716 | 1x |
| `4c_front_hot` | 4 | `90,70,20,0` | `local` | 28 | `61,62,63` | `60` | 14 | 19.8042 | 20.071 | 1.01x |
| `4c_gradient` | 4 | `0,20,60,100` | `local` | 28 | `60,61,62` | `63` | 14 | 20.1218 | 16.1594 | 0.803x |
| `4c_high` | 4 | `70,80,90,100` | `local` | 28 | `61,62,63` | `60` | 14 | 21.3379 | 17.1631 | 0.804x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | 22 | `62,63,65` | `60,61,64` | 11 | 23.1136 | n/a | n/a |
| `6c_high` | 6 | `50,60,70,80,90,100` | `local` | 28 | `60,61,62,63,64` | `65` | 14 | 26.6986 | 24.667 | 0.924x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `local` | 28 | `60,61,62,63` | `64,65` | 14 | 23.4937 | 21.278 | 0.906x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `local` | 28 | `60,61,62,63,65` | `64` | 14 | 27.2588 | 4.55138 | 0.167x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | 1 | `61,62,63,66` | `60,64,65,67` | 1 | 21.4633 | n/a | n/a |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `local` | 28 | `61,62,63,64,65,66` | `60,67` | 14 | 30.2904 | 23.9706 | 0.791x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `local` | 28 | `60,61,62,63,66,67` | `64,65` | 14 | 2.61829 | 19.3651 | 7.4x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `local` | 28 | `60,61,62,63,64,65,66` | `67` | 14 | 2.42606 | 21.8657 | 9.01x |

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
