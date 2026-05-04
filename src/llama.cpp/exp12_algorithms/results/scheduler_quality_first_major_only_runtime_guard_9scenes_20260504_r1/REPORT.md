# Heterogeneous Load Scheduler Effectiveness

- cgroup root: `/sys/fs/cgroup/ce_ada_llama_6079`
- repeats: `3`
- tokens per decode run: `1`
- baseline: no SVD, same core set and same heterogeneous background load
- model_schedule: model profile + DP scheduler; edge/end offload rows are selected but not executed because this experiment does not use adb

## Result

| scenario | cores | loads | mode | M | major | minor | clipped | baseline tok/s | schedule tok/s | speedup |
|---|---:|---|---|---:|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | 2 | `20,80` | `baseline_no_svd` | 28 | `` | `` | 0 | 9.25275 | 9.25275 | 1x |
| `2c_balanced_mid` | 2 | `50,50` | `baseline_no_svd` | 28 | `` | `` | 0 | 11.3584 | 11.3584 | 1x |
| `2c_high` | 2 | `80,100` | `baseline_no_svd` | 28 | `` | `` | 0 | 11.4954 | 11.4954 | 1x |
| `2c_idle_saturated` | 2 | `0,100` | `baseline_no_svd` | 28 | `` | `` | 0 | 11.7055 | 11.7055 | 1x |
| `4c_balanced_high` | 4 | `20,40,60,80` | `baseline_no_svd` | 28 | `` | `` | 0 | 16.4392 | 16.4392 | 1x |
| `4c_front_hot` | 4 | `90,70,20,0` | `major_only_no_svd` | 28 | `61,62,63` | `60` | 0 | 6.22102 | 16.1694 | 2.6x |
| `4c_gradient` | 4 | `0,20,60,100` | `baseline_no_svd` | 28 | `` | `` | 0 | 15.9487 | 15.9487 | 1x |
| `4c_high` | 4 | `70,80,90,100` | `baseline_no_svd` | 28 | `` | `` | 0 | 16.8858 | 16.8858 | 1x |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `major_only_no_svd` | 28 | `60,61,62,63,64,65,66` | `67` | 0 | 20.2239 | 25.2289 | 1.25x |

## Files

- `summary.csv`: aggregated throughput and speedup.
- `schedule_summary.csv`: scheduler choices and estimated latencies.
- `raw.csv`: every decode run with status and output tail.
