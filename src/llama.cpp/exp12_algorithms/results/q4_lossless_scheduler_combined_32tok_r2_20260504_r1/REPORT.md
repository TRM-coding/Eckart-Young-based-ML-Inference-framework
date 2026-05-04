# Q4 完整无损调度 9 场景性能报告

## 实验口径

- baseline / major_only / 协同卸载全部使用原始 Q4 模型：`src/llama.cpp/gguf_models/qwen.q4_0.gguf`。
- 只有需要 SVD 裁剪时才应切换到 SVD-capable GGUF；本轮最终选择均为 no-SVD 路线，因此未启用 SVD。
- decode tokens: `32`。
- repeats: `2`。
- 协同卸载：PC 计算 `[0,M)`，手机计算 `[M,28)`，手机端 `LAYER_COOP_KEEP_HOT=1`，coop threads=`8`。
- PPL：所有最终策略均为 no-SVD，无 rank 裁剪，无 timeout drop，因此 PPL 与原始 Q4 baseline 相同。ctx=128 sanity PPL = `15.1424 +/- 4.47405`。

## 结果

| 场景 | 负载 | 最终策略 | 参数 | baseline tok/s | scheduler tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `20,80` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 12.4440 | 15.5711 | 1.251x | 15.1424 |
| `2c_balanced_mid` | `50,50` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 12.6473 | 17.3932 | 1.375x | 15.1424 |
| `2c_high` | `80,100` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 12.5518 | 15.1871 | 1.210x | 15.1424 |
| `2c_idle_saturated` | `0,100` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 12.7975 | 15.1362 | 1.183x | 15.1424 |
| `4c_balanced_high` | `20,40,60,80` | `baseline_no_svd` | `60,61,62,63` | 18.6621 | 18.6621 | 1.000x | 15.1424 |
| `4c_front_hot` | `90,70,20,0` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 10.7003 | 15.8666 | 1.483x | 15.1424 |
| `4c_gradient` | `0,20,60,100` | `edge_end_no_svd` | `M=2;PC=[0,2);Phone=[2,28)` | 11.9725 | 17.1524 | 1.433x | 15.1424 |
| `4c_high` | `70,80,90,100` | `baseline_no_svd` | `60,61,62,63` | 19.8483 | 19.8483 | 1.000x | 15.1424 |
| `8c_high` | `30,40,50,60,70,80,90,100` | `baseline_no_svd` | `60,61,62,63,64,65,66,67` | 28.8488 | 28.8488 | 1.000x | 15.1424 |

## 汇总

- 非 baseline 策略场景数：`6/9`。
- speedup 范围：`1.000x` 到 `1.483x`。
- speedup 中位数：`1.210x`。
- 当前有效提升主要来自 `edge_end_no_svd` 协同卸载；`major_only_no_svd` 在原始 Q4 32-token 口径下没有成为最终最优。
- `8c_high` 中原始 Q4 all-core baseline 已经很快，major_only 不快；协同 M=2 在补测中 timeout，因此该场景保留 baseline 是正确选择。

## 文件

- `raw.csv`: `src/llama.cpp/exp12_algorithms/results/q4_lossless_scheduler_combined_32tok_r2_20260504_r1/raw.csv`
- `summary.csv`: `src/llama.cpp/exp12_algorithms/results/q4_lossless_scheduler_combined_32tok_r2_20260504_r1/summary.csv`
