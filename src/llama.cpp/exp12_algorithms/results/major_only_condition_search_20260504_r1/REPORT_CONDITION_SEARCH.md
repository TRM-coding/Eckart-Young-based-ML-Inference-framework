# Major-only 胜出条件搜索报告

## 目标

寻找什么实验条件下，本地 `major_only_no_svd` 能高于 all-core `baseline_no_svd`。

## 已验证有效条件

- 8 个 PC 核心 `60-67` 都允许参与计算。
- 热核心：`60-63`。
- 冷核心：`64-67`。
- 每个热核心启动 1 个 `stress-ng --cpu 1 --cpu-load 100 --cpu-method matrixprod` worker。
- 模型：原始 Q4 `qwen.q4_0.gguf`。
- decode：32 tokens。
- 当前已完成重复：2 次。
- major-only 完整枚举 p=1..7。

## 结果

| 场景 | 热核心 | worker/热核 | 方法 | baseline mean tok/s | 最优策略 | selected mean tok/s | speedup |
|---|---|---:|---|---:|---|---:|---:|
| `4hot_matrix_1w` | `60,61,62,63` | 1 | `matrixprod` | 7.7877 | `p=6;major=64,65,66,67,60,61` | 20.3524 | 2.613x |

## 策略明细

| 策略 | 参数 | runs | mean tok/s | speedup vs baseline |
|---|---|---:|---:|---:|
| `major_only_no_svd` | `p=6;major=64,65,66,67,60,61` | 2 | 20.3524 | 2.613x |
| `major_only_no_svd` | `p=4;major=64,65,66,67` | 2 | 20.2865 | 2.605x |
| `major_only_no_svd` | `p=3;major=64,65,66` | 2 | 16.6424 | 2.137x |
| `major_only_no_svd` | `p=2;major=64,65` | 2 | 12.2229 | 1.570x |
| `baseline_all_8c` | `60,61,62,63,64,65,66,67` | 2 | 7.7877 | 1.000x |
| `major_only_no_svd` | `p=1;major=64` | 2 | 6.3492 | 0.815x |
| `major_only_no_svd` | `p=5;major=64,65,66,67,60` | 2 | 5.2253 | 0.671x |
| `major_only_no_svd` | `p=7;major=64,65,66,67,60,61,62` | 2 | 0.7942 | 0.102x |

## 结论

这个条件下 `major_only_no_svd` 明确高于 baseline。最优为 `p=6; major=64,65,66,67,60,61`，平均吞吐 `20.35 tok/s`，baseline 平均 `7.79 tok/s`，提速约 `2.61x`。
这说明：当部分高负载核心显著拖慢 all-core baseline 时，调度器需要完整枚举 `p=1..7`，并选择跳过/减少这些热核心参与的 major-only 路线。

## 注意

本次搜索为节省时间，在找到有效条件后停止，因此只完整记录了第一个有效条件。后续可围绕该条件做更多 repeats 或扩展其他 stress 方法。
