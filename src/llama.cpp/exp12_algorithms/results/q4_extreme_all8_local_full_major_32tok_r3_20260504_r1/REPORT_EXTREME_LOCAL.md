# Q4 Extreme-load Full Major Enumeration Report

## 口径

- 所有场景均允许 8 个 PC 核心 `60-67` 参与计算。
- 背景负载为极端负载，用于观察本地 `major_only_no_svd` 是否会替代 baseline。
- 模型：原始 Q4 `qwen.q4_0.gguf`。
- tokens：`32`；repeats：`3`。
- `major_only_no_svd` 已完整枚举 `p=1..7`，不是抽样。
- 当前仅包含本地 no-SVD。adb 当前不稳定/offline，未写入 offloading 实测。
- PPL：最终策略均 no-SVD，无裁剪，PPL 与 Q4 baseline 相同，ctx=128 参考 `15.1424 +/- 4.47405`。

## Summary

| 场景 | 负载 | 最终策略 | 参数 | baseline mean tok/s | selected mean tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `6c_extreme_2idle` | `90,90,90,100,100,100,0,0` | `major_only_no_svd` | `p=7;major=66,67,60,61,62,63,64` | 1.4454 | 20.2824 | 14.032x | 15.1424 |
| `7c_extreme_1idle` | `80,90,90,100,100,100,100,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 28.6368 | 28.6368 | 1.000x | 15.1424 |
| `8c_all_100` | `100,100,100,100,100,100,100,100` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 28.6577 | 28.6577 | 1.000x | 15.1424 |
| `8c_extreme_all_hot` | `80,90,90,90,100,100,100,100` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 28.4305 | 28.4305 | 1.000x | 15.1424 |

## 解释

- `6c_extreme_2idle` 中，baseline 被高负载核心严重拖慢，完整枚举后 `p=7` 的 major_only 最优。
- `7c_extreme_1idle`、`8c_extreme_all_hot`、`8c_all_100` 中，baseline 反而最优；这说明在当前 stress-ng 负载口径下，全核 baseline 仍能获得较高吞吐，不能仅凭“负载很高”推断 offload/major_only 一定胜出。
- 若要验证 offloading 是否触发，需要 adb shell 能稳定执行，并用同一场景补测 `edge_end_no_svd`。
