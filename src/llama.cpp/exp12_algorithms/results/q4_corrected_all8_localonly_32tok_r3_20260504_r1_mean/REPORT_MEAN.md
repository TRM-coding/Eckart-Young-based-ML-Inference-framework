# Q4 Corrected All-8 Local-only Mean-performance Report

## 口径

- 正确语义：所有场景均允许 8 个 PC 核心 `60-67` 参与计算；`2c/4c/8c` 只表示有多少核心被加背景负载。
- 模型：原始 Q4 `qwen.q4_0.gguf`。
- tokens：`32`。
- repeats：`3`。
- 本报告使用“平均性能”：同一场景同一策略取 repeats 的平均 tok/s；最终策略选择平均 tok/s 最高者。
- 当前报告只包含本地 no-SVD：`baseline_no_svd` 和 `major_only_no_svd`。协同卸载需 adb 稳定后另补。
- PPL：最终策略均 no-SVD，无裁剪，PPL 与 Q4 baseline 相同，ctx=128 参考 `15.1424 +/- 4.47405`。

## 结果

| 场景 | 负载 | 最终策略 | 参数 | baseline mean tok/s | scheduler mean tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `20,80,0,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 31.8784 | 31.8784 | 1.000x | 15.1424 |
| `2c_balanced_mid` | `50,50,0,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 32.2512 | 32.2512 | 1.000x | 15.1424 |
| `2c_high` | `80,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=7;major=62,63,64,65,66,67,60` | 16.1587 | 31.7169 | 1.963x | 15.1424 |
| `2c_idle_saturated` | `0,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=7;major=60,62,63,64,65,66,67` | 25.7767 | 35.3780 | 1.372x | 15.1424 |
| `4c_balanced_high` | `20,40,60,80,0,0,0,0` | `major_only_no_svd` | `p=6;major=64,65,66,67,60,61` | 14.6674 | 28.3030 | 1.930x | 15.1424 |
| `4c_front_hot` | `90,70,20,0,0,0,0,0` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 33.1322 | 33.1322 | 1.000x | 15.1424 |
| `4c_gradient` | `0,20,60,100,0,0,0,0` | `major_only_no_svd` | `p=7;major=60,64,65,66,67,61,62` | 22.6894 | 30.6988 | 1.353x | 15.1424 |
| `4c_high` | `70,80,90,100,0,0,0,0` | `major_only_no_svd` | `p=7;major=64,65,66,67,60,61,62` | 4.3239 | 21.6383 | 5.004x | 15.1424 |
| `8c_high` | `30,40,50,60,70,80,90,100` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 27.0780 | 27.0780 | 1.000x | 15.1424 |

## 汇总

- 非 baseline 场景数：`5/9`。
- mean-performance speedup 范围：`1.000x` 到 `5.004x`。
- mean-performance speedup 中位数：`1.353x`。
- mean-performance speedup 平均值：`1.736x`。

## 文件

- `summary_mean.csv`: `src/llama.cpp/exp12_algorithms/results/q4_corrected_all8_localonly_32tok_r3_20260504_r1_mean/summary_mean.csv`
- `strategy_mean.csv`: `src/llama.cpp/exp12_algorithms/results/q4_corrected_all8_localonly_32tok_r3_20260504_r1_mean/strategy_mean.csv`
