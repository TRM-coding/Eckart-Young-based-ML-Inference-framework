# Q4 Corrected All-8 Local-only Max-performance Report

## 口径

- 正确语义：所有场景均允许 8 个 PC 核心 `60-67` 参与计算；`2c/4c/8c` 只表示有多少核心被加背景负载。
- 模型：原始 Q4 `qwen.q4_0.gguf`。
- tokens：`32`。
- repeats：`3`。
- 本报告按用户要求使用“最高性能”：同一场景同一策略取重复中的最大 tok/s；最终策略选择最大 tok/s 最高者。
- 当前报告只包含本地 no-SVD：`baseline_no_svd` 和 `major_only_no_svd`。协同卸载需 adb 稳定后另补。
- PPL：最终策略均 no-SVD，无裁剪，PPL 与 Q4 baseline 相同，ctx=128 参考 `15.1424 +/- 4.47405`。

## 结果

| 场景 | 负载 | 最终策略 | 参数 | baseline max tok/s | scheduler max tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `20,80,0,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 32.8447 | 32.8447 | 1.000x | 15.1424 |
| `2c_balanced_mid` | `50,50,0,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 33.0200 | 33.0200 | 1.000x | 15.1424 |
| `2c_high` | `80,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=7;major=62,63,64,65,66,67,60` | 28.6291 | 32.6299 | 1.140x | 15.1424 |
| `2c_idle_saturated` | `0,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=7;major=60,62,63,64,65,66,67` | 34.4643 | 37.4369 | 1.086x | 15.1424 |
| `4c_balanced_high` | `20,40,60,80,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 29.8555 | 29.8555 | 1.000x | 15.1424 |
| `4c_front_hot` | `90,70,20,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 35.4072 | 35.4072 | 1.000x | 15.1424 |
| `4c_gradient` | `0,20,60,100,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 32.7348 | 32.7348 | 1.000x | 15.1424 |
| `4c_high` | `70,80,90,100,0,0,0,0` | `major_only_no_svd` | `p=7;major=64,65,66,67,60,61,62` | 8.9605 | 31.8271 | 3.552x | 15.1424 |
| `8c_high` | `30,40,50,60,70,80,90,100` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 28.6190 | 28.6190 | 1.000x | 15.1424 |

## 汇总

- 非 baseline 场景数：`3/9`。
- max-performance speedup 范围：`1.000x` 到 `3.552x`。
- max-performance speedup 中位数：`1.000x`。

## 文件

- `summary_max.csv`: `src/llama.cpp/exp12_algorithms/results/q4_corrected_all8_localonly_32tok_r3_20260504_r1_max/summary_max.csv`
- `strategy_max.csv`: `src/llama.cpp/exp12_algorithms/results/q4_corrected_all8_localonly_32tok_r3_20260504_r1_max/strategy_max.csv`
