# Exp12 调度策略 F16 全链路性能报告 日期：2026-05-02

本文档总结在 F16 模型上重新执行的 exp12 调度策略全链路验证。实验核心设置保持与上一版性能报告一致：仍然使用 `60-79` 隔离 CPU 集合、不使用 adb、使用异构 per-core 背景负载。

## 结论摘要

F16 模型上，scheduler 的本地 runtime 选择结果比 Q4 更偏向 SVD 截断路径。重新从 SVD-capable F16 GGUF 的 runtime 测量开始做 calibration 后，12 个场景全部选择本地 `trunc_even_*` 策略，没有选择 `edge_end`。

主要结论如下：

- 本轮 F16 full-chain 测量共覆盖 12 个异构负载场景。
- 所有 validation 运行均为 `ok`，没有 `decode_svd_test` abort，也没有 `edge_end` 占位行。
- 如果 baseline 定义为同一个 SVD-capable GGUF 上关闭 SVD 的 no-SVD runtime，那么 12 个场景 scheduler 均快于 baseline。
- 如果 baseline 定义为原始不带 SVD 信息的 `qwen.gguf`，scheduler 大多数场景仍然更慢，不能声明相对原始模型加速。
- 除去 `6c_mixed` 中 no-SVD baseline 的异常低速点后，scheduler 的提速范围为 `1.360x` 到 `1.636x`，中位数为 `1.450x`。
- 若不排除异常点，`6c_mixed` 因 baseline validation 中位数掉到 `0.485 tok/s`，会得到 `24.527x` 的异常放大 speedup；该数值不应作为稳定收益解读。

因此，在 F16 模型上，当前 local SVD truncation 调度策略显著快于“同一 SVD GGUF 的 no-SVD runtime baseline”；但它尚未证明快于“原始不带 SVD 信息的 F16 GGUF baseline”。如果论文最终 baseline 是原始模型，那么当前 F16 结论必须修正为：SVD GGUF 转换/执行路径本身引入了明显 overhead，scheduler 只能部分弥补该 overhead。

## 实验设置

- 机器端隔离 CPU 集合：`60-79`。
- 实际 decode 实验使用 `60-67` 的不同子集构造 4 核、6 核和 8 核场景。
- 背景负载使用 `stress-ng --cpu-load` 按核心单独生成。
- Decode 程序：`build-release-current/decode_svd_test`。
- F16 模型文件：`src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`。
- 模型大小：`6,198,508,608 bytes`，约 `5.77 GiB`。
- 模型类型：F16。
- 每次测量 decode token 数：`1`。
- Calibration repeats：`1`。
- Validation repeats：`3`。
- Candidate policies：`baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9`。
- 启用非 baseline 策略的 calibration speedup 阈值：`1.10x`。
- 本轮实验不使用 adb，也不执行端侧 runtime。

本报告现在同时给出两种 baseline：

- `svd_gguf_no_svd_runtime`：使用同一个 SVD-capable F16 GGUF 文件 `qwen.gguf.sort_svd.compact.gguf`，但 runtime 中关闭 SVD rate 文件和调度策略。
- `original_qwen_no_svd`：使用原始不带 SVD 信息的 F16 GGUF 文件 `qwen.gguf`，不使用 SVD、不使用调度策略。

需要特别说明：本报告里的 baseline 不是使用原始 `qwen.gguf` 跑出来的，而是使用同一个 SVD-capable F16 模型文件 `qwen.gguf.sort_svd.compact.gguf`，只是在 runtime 中关闭 SVD rate 文件和调度策略。因此它应被理解为“同一 SVD GGUF 下的 no-SVD runtime baseline”，而不是“原始未转换 GGUF baseline”。这样做的目的是保证 baseline 和 scheduler 只差调度/SVD runtime 策略，避免模型文件、tensor layout、加载路径不同带来的干扰。

用户追问后，本文补充了原始 `qwen.gguf` 的 baseline 测量。该测量显示，原始 GGUF baseline 明显快于 SVD-capable GGUF 的 no-SVD runtime baseline。

## Full-Chain Scheduler 结果

本表为 SVD-capable F16 GGUF 重新 calibration 后得到的最终 validation 结果。吞吐均为 3 次 validation 重复实验的 decode-only tok/s 中位数。该表的 speedup 分母是 `svd_gguf_no_svd_runtime`，不是原始 `qwen.gguf`。

| 场景 | 核心数 | 每核负载 | F16 选择策略 | Baseline tok/s | Scheduler tok/s | 提速 |
|---|---:|---|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | `trunc_even_0.9` | 7.0992 | 10.0269 | 1.412x |
| `4c_mixed` | 4 | `0,30,60,90` | `trunc_even_0.85` | 7.0065 | 9.5279 | 1.360x |
| `4c_front_hot` | 4 | `90,70,20,0` | `trunc_even_0.9` | 7.1117 | 10.2162 | 1.437x |
| `4c_high` | 4 | `70,80,90,100` | `trunc_even_0.9` | 6.9033 | 10.1734 | 1.474x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `trunc_even_0.9` | 8.1091 | 13.2671 | 1.636x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `trunc_even_0.9` | 0.4850 | 11.8957 | 24.527x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `trunc_even_0.8` | 8.3463 | 11.8290 | 1.417x |
| `6c_high` | 6 | `50,60,70,80,90,100` | `trunc_even_0.9` | 8.9765 | 13.6248 | 1.518x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `trunc_even_0.9` | 10.2935 | 14.2382 | 1.383x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `trunc_even_0.9` | 9.2549 | 13.5200 | 1.461x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `trunc_even_0.9` | 9.2826 | 13.4633 | 1.450x |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `trunc_even_0.85` | 9.5592 | 13.9039 | 1.455x |

## 原始 F16 GGUF Baseline 对比

用户追问 baseline 后，额外使用原始不带 SVD 信息的模型文件 `src/llama.cpp/gguf_models/qwen.gguf` 跑了同一组场景的 no-SVD baseline。该模型可以被当前 `decode_svd_test` 正常加载，并且 SVD profile 输出均为 0，说明没有启用 SVD 路径。

下表对比三组结果：

- `Original baseline`：原始 `qwen.gguf`。
- `SVD GGUF no-SVD`：`qwen.gguf.sort_svd.compact.gguf`，关闭 SVD 和调度。
- `Scheduler`：`qwen.gguf.sort_svd.compact.gguf`，使用 F16 calibration 选择的 `trunc_even_*` 策略。

| 场景 | 核心数 | 每核负载 | Original baseline tok/s | SVD GGUF no-SVD tok/s | Scheduler tok/s | Scheduler / Original | Scheduler / SVD no-SVD |
|---|---:|---|---:|---:|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | 13.4796 | 7.0991 | 10.0269 | 0.744x | 1.412x |
| `4c_mixed` | 4 | `0,30,60,90` | 13.3970 | 7.0065 | 9.5279 | 0.711x | 1.360x |
| `4c_front_hot` | 4 | `90,70,20,0` | 12.8982 | 7.1117 | 10.2162 | 0.792x | 1.437x |
| `4c_high` | 4 | `70,80,90,100` | 13.0021 | 6.9033 | 10.1734 | 0.782x | 1.474x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | 16.6595 | 8.1091 | 13.2671 | 0.796x | 1.636x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | 15.4372 | 0.4850 | 11.8957 | 0.771x | 24.527x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | 15.1649 | 8.3462 | 11.8290 | 0.780x | 1.417x |
| `6c_high` | 6 | `50,60,70,80,90,100` | 15.6141 | 8.9765 | 13.6248 | 0.873x | 1.518x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | 14.2282 | 10.2935 | 14.2382 | 1.001x | 1.383x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | 14.7553 | 9.2549 | 13.5200 | 0.916x | 1.461x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | 15.6752 | 9.2826 | 13.4633 | 0.859x | 1.450x |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | 15.8871 | 9.5592 | 13.9039 | 0.875x | 1.455x |

相对原始 `qwen.gguf` baseline 的汇总如下：

| 指标 | Scheduler / Original |
|---|---:|
| 最小值 | 0.711x |
| 中位数 | 0.794x |
| 平均值 | 0.825x |
| 最大值 | 1.001x |
| 超过原始 baseline 的场景数 | 1 / 12 |

这说明：当前 F16 scheduler 的确能提升 SVD-capable GGUF 的 runtime 表现，但还没有追上原始 F16 GGUF 的 no-SVD baseline。真正的问题不在调度器本身是否能相对 SVD no-SVD baseline 加速，而在 SVD-capable GGUF 的 no-SVD 执行路径相对原始 GGUF 已经有较大 overhead。

## 汇总指标

包含全部 12 个场景时：

| 指标 | 数值 |
|---|---:|
| 场景数 | 12 |
| validation raw rows | 72 |
| calibration raw rows | 60 |
| 实际运行状态 | 全部 `ok` |
| 最小提速 | 1.360x |
| 提速中位数 | 1.452x |
| 几何平均提速 | 1.839x |
| 算术平均提速 | 3.377x |
| 最大提速 | 24.527x |
| 低于 baseline 的场景数 | 0 |
| 低于 1.10x 的场景数 | 0 |

由于 `6c_mixed` 的 no-SVD baseline 在 validation 中出现明显异常低速，去掉该场景后再看更稳健的本地收益：

| 指标 | 数值 |
|---|---:|
| 场景数 | 11 |
| 最小提速 | 1.360x |
| 提速中位数 | 1.450x |
| 几何平均提速 | 1.453x |
| 算术平均提速 | 1.455x |
| 最大提速 | 1.636x |
| 低于 baseline 的场景数 | 0 |
| 低于 1.10x 的场景数 | 0 |

## 异常点说明

`6c_mixed` 的 validation 明细如下：

| repeat | Baseline tok/s | Scheduler tok/s |
|---:|---:|---:|
| 0 | 0.4848 | 12.8443 |
| 1 | 2.1599 | 3.9488 |
| 2 | 0.4850 | 11.8957 |

这说明该场景下 no-SVD baseline 在 F16 模型上出现了重复低速，而不是 scheduler 单方面获得了稳定 24 倍收益。因此报告中的主结论不使用 `24.527x` 作为代表性加速，而使用去掉该异常场景后的中位数和几何平均值描述整体效果。

另外，少数场景的单次 scheduler run 也存在低速波动，例如 `4c_mixed` 的一次 scheduler validation 为 `1.5508 tok/s`，`8c_front_hot` 的一次 scheduler validation 为 `2.9891 tok/s`。使用中位数可以减弱单次系统调度抖动的影响，但后续若要写论文图表，建议将 repeats 提升到 5 或 7。

## 与“沿用 Q4 策略”的对照

为了区分“F16 重新测量后自决策”和“沿用 Q4 final policy map”，本轮额外跑了一组对照实验：

```text
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_initial_r2/
```

该对照使用 Q4 报告中的 `policy_overrides_final.json`。由于其中 6 个场景被指定为 `edge_end`，而本轮不使用 adb，所以只有 6 个本地场景可验证。

| 指标 | 数值 |
|---|---:|
| 本地可验证场景数 | 6 |
| `edge_end` 占位场景数 | 6 |
| 最小本地提速 | 1.340x |
| 本地提速中位数 | 1.378x |
| 本地平均提速 | 1.770x |
| 最大本地提速 | 3.779x |
| 低于 baseline 的本地场景数 | 0 |

结论是：即使沿用 Q4 的最终策略，F16 本地可验证场景也没有低于 baseline；但如果允许 F16 从自己的 runtime calibration 重新选择策略，则 12 个场景全部可以在本地完成验证，且整体收益更稳定。

## F16 Runtime Policy Model

基于本轮 F16 full-chain raw runtime 数据，重新训练了一版 runtime policy model：

```text
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_f16_20260502_r1/
```

模型训练摘要如下：

| 项目 | 数值 |
|---|---:|
| 训练样本数 | 132 |
| 最优模型 | `extra_trees` |
| 目标变量 | `log(speedup_vs_no_svd)` |
| speedup 训练裁剪上限 | 3.0x |
| Grouped-CV speedup MAE | 0.2182 |

该模型只基于本次 12 个场景和 5 个 candidate policies 的 runtime 测量，因此可以作为 F16 调度策略的轻量预测器，但还不应被理解为覆盖所有模型、所有 prompt 长度、所有核心组合的通用模型。

## 文件位置

F16 full-chain 主结果：

```text
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_fullchain/REPORT.md
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_fullchain/decisions.csv
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_fullchain/raw.csv
```

F16 runtime policy model：

```text
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_f16_20260502_r1/REPORT.md
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_f16_20260502_r1/metrics.csv
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_f16_20260502_r1/runtime_policy_model.pkl
```

F16 沿用 Q4 策略对照：

```text
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_initial_r2/REPORT.md
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_initial_r2/decisions.csv
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_f16_20260502_initial_r2/raw.csv
```

## 最终结论

在 SVD-capable F16 GGUF 内部，全链路重新测量后的 scheduler 是有效的：

- 所有 12 个异构负载场景均完成本地 validation。
- 所有场景的 scheduler 中位数吞吐均高于同一 SVD GGUF 的 no-SVD runtime baseline。
- 排除一个 no-SVD baseline 明显异常低速场景后，整体提速中位数为 `1.450x`，几何平均为 `1.453x`。
- F16 模型上最优策略更倾向较高截断率，主要选择 `trunc_even_0.9`，少数场景选择 `trunc_even_0.85` 或 `trunc_even_0.8`。

但是，相对原始不带 SVD 信息的 `qwen.gguf` baseline，当前 scheduler 大多数场景仍然更慢。这个结果说明，后续优化重点应该转向 SVD-capable GGUF 的 no-SVD 执行开销和模型转换 layout 开销：只有先把 SVD GGUF 的 no-SVD 路径拉近原始 GGUF，scheduler 的加速才可能成为论文中相对原始模型也成立的端到端加速。
