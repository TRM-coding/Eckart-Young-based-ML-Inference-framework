# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `3`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload rows are selected but not executed because this experiment does not use adb

## Result

| scenario | cores | loads | mode | M | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---:|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | 2 | `20,80` | `edge_end_no_svd` | 20 | `` | `` | 0 | 11.3859 | n/a | n/a |
| `2c_balanced_mid` | 2 | `50,50` | `edge_end_no_svd` | 20 | `` | `` | 0 | 11.7473 | n/a | n/a |
| `2c_high` | 2 | `80,100` | `edge_end_no_svd` | 2 | `` | `` | 0 | 11.2776 | n/a | n/a |
| `2c_idle_saturated` | 2 | `0,100` | `edge_end_no_svd` | 20 | `` | `` | 0 | 11.7691 | n/a | n/a |
| `4c_balanced_high` | 4 | `20,40,60,80` | `local` | 28 | `60,61,62,63` | `` | 0 | 19.3818 | 19.3818 | 1x |
| `4c_front_hot` | 4 | `90,70,20,0` | `local` | 28 | `61,62,63` | `60` | 14 | 21.1915 | 20.4099 | 0.963x |
| `4c_gradient` | 4 | `0,20,60,100` | `local` | 28 | `60,61,62` | `63` | 14 | 15.7415 | 19.1488 | 1.22x |
| `4c_high` | 4 | `70,80,90,100` | `local` | 28 | `61,62,63` | `60` | 14 | 20.6824 | 16.7563 | 0.81x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | 22 | `62,63,65` | `60,61,64` | 11 | 1.7803 | n/a | n/a |
| `6c_high` | 6 | `50,60,70,80,90,100` | `local` | 28 | `60,61,62,63,64` | `65` | 14 | 24.9193 | 21.2965 | 0.855x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `local` | 28 | `60,61,62,63` | `64,65` | 14 | 25.0751 | 21.4105 | 0.854x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `local` | 28 | `60,61,62,63,65` | `64` | 14 | 24.3045 | 22.8669 | 0.941x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | 1 | `61,62,63,66` | `60,64,65,67` | 1 | 0.393366 | n/a | n/a |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `local` | 28 | `61,62,63,64,65,66` | `60,67` | 14 | 29.0605 | 23.8411 | 0.82x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `local` | 28 | `60,61,62,63,66,67` | `64,65` | 14 | 0.202711 | 2.32555 | 11.5x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `local` | 28 | `60,61,62,63,64,65,66` | `67` | 14 | 4.52056 | 3.92944 | 0.869x |

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
