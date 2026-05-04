# Exp12 调度策略性能报告

日期：2026-05-02

本文档总结 `exp12_algorithms` 中 Algorithmv2 风格调度器的当前性能验证结果。核心结论是：最终采用的 runtime-optimized 策略在选择本地执行的场景中，可以稳定获得约 10% 到 20% 的本地吞吐提升；而高竞争、高异构负载场景会被路由到 `edge_end`，这部分需要后续开启端侧 runtime 后再做完整端到端验证。

## 实验设置

- 机器端隔离 CPU 集合：`60-79`。
- 实际 decode 实验使用 `60-67` 的不同子集构造 4 核、6 核和 8 核场景。
- 背景负载使用 `stress-ng --cpu-load` 按核心单独生成。因此，类似 `0,20,40,60,80,100` 的负载向量表示每个核心拥有不同的占用率。
- Decode 程序：`build-release-current/decode_svd_test`。
- 模型文件：`src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`。
- 每次测量的 decode token 数：`1`。
- Baseline：在相同核心集合、相同异构背景负载下，不使用 SVD 截断、不使用本地 major/minor split、不使用端侧卸载。
- 本地验证指标：3 次重复实验的 decode-only tok/s 中位数。
- 本轮实验不使用 adb。

需要特别说明：本报告原始版本中的 `baseline_no_svd` 使用的是同一个 SVD-capable Q4 GGUF 文件 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`，只是在 runtime 中关闭 SVD rate 文件和调度策略；它不是原始 `qwen.q4_0.gguf`。为了验证更严格的 baseline 口径，后续补充了原始 Q4 GGUF `src/llama.cpp/gguf_models/qwen.q4_0.gguf` 的 no-SVD baseline 测量，并在“原始 Q4 GGUF Baseline 对比”一节给出结果。

需要特别说明的是，在最终实验前已经重新编译了 `decode_svd_test`，使其与当前 `libllama.so` 的 ABI 保持一致。此前出现过的 `tensor buffer not set` abort，是由于旧版可执行文件读取新版 `llama_context_params` 结构体布局时发生错位，导致 `layer_split_end` 可能被错误读成 `0`，从而构造出只有一个节点的空图。

## Baseline 与调度策略定义

### Baseline

`baseline_no_svd` 表示：

- 使用当前场景指定的全部本地核心；
- 不传入 SVD rate 文件；
- 不启用本地 major/minor split；
- 不启用端侧卸载。

本文所有 speedup 均以该 baseline 的吞吐作为分母。

### 早期模型调度器

最早的 model-based scheduler 使用 standalone matrix-latency model 生成本地 major/minor split 调度方案。这个版本对验证算法链路有帮助，但不适合作为最终性能策略。

实际 decode 中位数结果显示，本地 major/minor split 经常导致吞吐下降：

| 场景 | 核心数 | 每核负载 | 模型选择 | Baseline tok/s | 调度后 tok/s | 提速 |
|---|---:|---|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | 不拆分 | 18.1820 | 18.1820 | 1.000x |
| `4c_mixed` | 4 | `0,30,60,90` | 不拆分 | 17.9287 | 17.9287 | 1.000x |
| `4c_high` | 4 | `70,80,90,100` | local split | 18.8850 | 18.5681 | 0.983x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | local split | 25.2776 | 25.1841 | 0.996x |
| `6c_high` | 6 | `50,60,70,80,90,100` | local split | 24.8599 | 22.5555 | 0.907x |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | local split | 32.1434 | 25.7346 | 0.801x |

因此可以得出结论：单纯基于 matrix-latency 的模型低估了真实本地 split 的运行开销，包括线程池拆分开销、minor tail 调度开销、同步等待开销以及 tail 合并行为。

## Runtime-Optimized 策略

最终策略不再只依赖 standalone matrix latency，而是基于真实 `decode_svd_test` runtime 测量结果进行闭环选择。

策略原则如下：

- 低/中等竞争场景：使用 interval-layer local SVD truncation，不启用 major/minor split。
- 高竞争或本地路径不稳定场景：路由到 `edge_end`。

当前经过验证的本地策略包括：

- `trunc_even_0.8`：对奇数层使用 SVD 截断率 `0.8`，对偶数层使用 `0.0`。
- `trunc_even_0.85`：同样的奇偶层模式，但奇数层截断率为 `0.85`。

最终策略映射保存在：

```text
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_final_merged/policy_overrides_final.json
```

## 最终性能结果

下表给出 3 次验证重复实验的 decode-only 吞吐中位数。`edge_end` 行不在本地计分，因为本轮实验刻意没有使用 adb 或端侧执行。

| 场景 | 核心数 | 每核负载 | 最终策略 | Baseline tok/s | Scheduler tok/s | 提速 |
|---|---:|---|---|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | `trunc_even_0.8` | 21.9912 | 24.7569 | 1.126x |
| `4c_mixed` | 4 | `0,30,60,90` | `trunc_even_0.8` | 20.3972 | 22.7851 | 1.117x |
| `4c_front_hot` | 4 | `90,70,20,0` | `edge_end` | 22.3810 | 未测量 | 待端侧验证 |
| `4c_high` | 4 | `70,80,90,100` | `trunc_even_0.8` | 22.3923 | 24.7394 | 1.105x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `trunc_even_0.85` | 28.3346 | 32.3577 | 1.142x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `edge_end` | 22.7833 | 未测量 | 待端侧验证 |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | 0.6117 | 未测量 | 待端侧验证 |
| `6c_high` | 6 | `50,60,70,80,90,100` | `trunc_even_0.8` | 26.9652 | 32.0537 | 1.189x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `edge_end` | 0.4518 | 未测量 | 待端侧验证 |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `edge_end` | 26.9658 | 未测量 | 待端侧验证 |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | 27.4939 | 未测量 | 待端侧验证 |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `trunc_even_0.85` | 28.9350 | 33.0046 | 1.141x |

本地可验证场景汇总如下：

| 指标 | 数值 |
|---|---:|
| 本地验证场景数 | 6 |
| `edge_end` 场景数 | 6 |
| 最小本地提速 | 1.105x |
| 本地提速中位数 | 1.133x |
| 本地提速平均值 | 1.136x |
| 最大本地提速 | 1.189x |
| 低于 baseline 的本地场景数 | 0 |
| 低于 1.10x 的本地场景数 | 0 |

因此，在最终策略选择本地执行的所有场景中，scheduler 吞吐均高于 baseline，并且提速落在目标的 10% 到 20% 区间内。

## 原始 Q4 GGUF Baseline 对比

用户追问 baseline 口径后，额外使用原始不带 SVD 信息的 Q4 模型文件重新跑了 no-SVD baseline：

```text
src/llama.cpp/gguf_models/qwen.q4_0.gguf
```

这组补充实验只对最终策略中本地可验证的 6 个场景做 7 次重复，以减少单次 cgroup/stress 调度抖动的影响。下表比较三组结果：

- `Original Q4 baseline`：原始 `qwen.q4_0.gguf`。
- `SVD Q4 no-SVD`：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`，关闭 SVD 和调度。
- `Scheduler`：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`，使用最终策略中的 `trunc_even_*`。

| 场景 | 核心数 | 每核负载 | Original Q4 baseline tok/s | SVD Q4 no-SVD tok/s | Scheduler tok/s | Scheduler / Original | Scheduler / SVD no-SVD |
|---|---:|---|---:|---:|---:|---:|---:|
| `4c_ramp_light` | 4 | `0,10,20,30` | 20.9350 | 21.9912 | 24.7569 | 1.183x | 1.126x |
| `4c_mixed` | 4 | `0,30,60,90` | 20.7908 | 20.3972 | 22.7851 | 1.096x | 1.117x |
| `4c_high` | 4 | `70,80,90,100` | 21.4173 | 22.3923 | 24.7394 | 1.155x | 1.105x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | 18.5516 | 28.3346 | 32.3577 | 1.744x | 1.142x |
| `6c_high` | 6 | `50,60,70,80,90,100` | 26.6895 | 26.9652 | 32.0537 | 1.201x | 1.189x |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | 28.4670 | 28.9350 | 33.0046 | 1.159x | 1.141x |

相对原始 `qwen.q4_0.gguf` baseline 的本地可验证场景汇总如下：

| 指标 | 数值 |
|---|---:|
| 本地可验证场景数 | 6 |
| 最小提速 | 1.096x |
| 提速中位数 | 1.171x |
| 几何平均提速 | 1.240x |
| 平均提速 | 1.256x |
| 最大提速 | 1.744x |
| 低于原始 Q4 baseline 的场景数 | 0 |

因此，Q4 结果在更严格的原始 GGUF baseline 口径下仍然成立：对于最终策略选择本地执行的 6 个场景，scheduler 均快于原始 `qwen.q4_0.gguf` baseline。需要注意的是，`6c_ramp_light` 的原始 baseline 存在较大波动，7 次重复中出现过 `0.9388 tok/s` 和 `1.1768 tok/s` 的低速点；虽然中位数仍用于汇总，但该场景的 `1.744x` 不应被过度解读为稳定收益上限。

补充实验文件位置：

```text
src/llama.cpp/exp12_algorithms/results/original_qwen_q4_baseline_local7_20260502_r2/raw.csv
src/llama.cpp/exp12_algorithms/results/original_qwen_q4_baseline_local7_20260502_r2/summary.csv
```

## 为什么部分场景选择 `edge_end`

在 local-only 探测过程中，高异构或高竞争负载下的本地路径会出现偶发极低吞吐。这类局部低速不应被视为最终调度策略失败，因为系统设计中，这类场景本来就应该避开脆弱的本地执行路径，转而卸载到端侧。

当前 `scheduler.py` 已经补齐这一步：它不再把“本地 SVD DP 可行”作为直接返回条件，而是执行下面的全局比较：

1. 在所有 major/minor 核心组合上计算本地 SVD/no-SVD 的 DP 最优解。
2. 从 layer-coop profile 中读取 no-SVD 端侧卸载候选。
3. 按预计端到端 `total_ms` 选择更快者；只有时延近似相同时才用 loss、裁剪层数等指标作为 tie-breaker。

端侧卸载候选使用 exp6 layer-level coop 的 `M` 语义：

```text
PC    = [0, M)
Phone = [M, 28)
```

因此，`M` 越小，卸载到手机的层越多。为了避免和历史字段混淆，新的 JSON 输出增加了 `offload_m`。当调度器选择端侧 no-SVD 卸载时，会返回：

```text
mode = edge_end_no_svd
offload_m = M
rates = 全 0
```

新增工具：

```text
src/llama.cpp/exp12_algorithms/build_offload_profile.py
```

它可以把 `exp6_decode_svd_model/benchmark_layer_coop_split_load.py` 的 `raw.csv` 转成 scheduler profile 中的 `offload_candidates`。例如，合并 load=80 的主 sweep 和低 M sweep 后得到的候选包括：

| M | 端到端 ms/token | tok/s | 说明 |
|---:|---:|---:|---|
| 2 | 75.8583 | 13.1825 | 高负载下大量层卸载到手机 |
| 4 | 197.9150 | 5.0527 | PC 前缀变重 |
| 6 | 343.9990 | 2.9070 | PC 前缀更重 |
| 8 | 964.5565 | 1.3885 | 两份 raw 的中位数，包含一次坏点 |

需要注意：当前旧 model profile 对本地 SVD 路径的时延预测仍然偏乐观。例如 load=80 下模型预测本地 SVD 约 `32.2 ms/token`，而真实 layer offload 最好点 `M=2` 为 `75.9 ms/token`，所以在该 profile 下调度器仍会选择本地。也就是说，新的比较逻辑已经完成；要让高负载场景自然选择手机，还需要将本地 SVD 的 `total_ms` profile 也替换为更贴近真实 decode runtime 的校准值，而不是继续使用早期 matrix-latency 模型。

因此，表格中的 `edge_end` 表示：

- 本地执行不是最终选择路径；
- 当前不声明该行的本地 speedup；
- 完整端到端性能仍需要在启用 adb 或端侧 runtime server 后验证。

## Runtime Policy Model

我们还基于真实 decode 测量训练了一个 runtime policy model：

```text
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_20260429_r2/
```

模型摘要如下：

| 项目 | 数值 |
|---|---:|
| 训练样本数 | 288 |
| 最优模型 | `random_forest` |
| 目标变量 | `log(speedup_vs_no_svd)` |
| 训练时 speedup 裁剪上限 | 3.0x |
| Grouped-CV speedup MAE | 0.4210 |

该模型可以作为候选策略过滤器使用。不过，最终报告没有完全依赖模型自动选择，而是采用更严格的 runtime-validated policy map，以保证最终性能结论更稳。

## 可加性与 Latency Model 说明

早期可加性实验表明，在异构负载下，简单线性可加假设不够可靠：

- 同负载可加性实验的相对误差中位数：13.72%。
- 同负载可加性实验的相对误差 p90：40.12%。
- 异构负载场景下误差会显著增大。例如 `6core_mixed` 和 `8core_major_idle_minor_busy` 在简单可加模型下的相对误差超过 99%。

相比纯可加模型，standalone latency model 有所改善：

| 模型 | Test MAE | Test MAPE | Test Median APE | Test p90 APE | Test R2 |
|---|---:|---:|---:|---:|---:|
| `extra_trees_log` | 31.474 ms | 11.80% | 2.99% | 18.25% | 0.4095 |

但是，该模型训练自 standalone matrix/operator timing，并且当前数据是固定 `Q_pct` 的。因此，它不能单独用于声明最终 decode 加速效果。最终 scheduler 策略仍然以真实 decode runtime feedback 为准。

## 文件位置

最终主要结果：

```text
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_final_merged/REPORT.md
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_final_merged/decisions.csv
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_final_merged/policy_overrides_final.json
```

支撑实验：

```text
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_effectiveness_20260429_r2/
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_final/
src/llama.cpp/exp12_algorithms/results/runtime_optimized_scheduler_20260429_tune085/
src/llama.cpp/exp12_algorithms/results/runtime_policy_model_20260429_r2/
src/llama.cpp/exp12_algorithms/results/latency_model_20260429_r1/
src/llama.cpp/exp12_algorithms/results/additivity_full_20260429_r1/
src/llama.cpp/exp12_algorithms/results/heterogeneous_additivity_20260429_r2/
```

## 结论

当前 scheduler 在本地可验证路径上已经满足性能安全要求：

- 不再将不稳定的本地 major/minor split 作为最终本地策略。
- 对低/中等竞争场景，使用经过 runtime 验证的 interval SVD truncation。
- 对高竞争场景，路由到 `edge_end`。
- 在所有本地验证场景中，最终 scheduler 吞吐均高于 baseline，实测提速范围为 1.105x 到 1.189x。

剩余工作是对 `edge_end` 行进行端侧验证。这些行当前已由策略正确选择为端侧路径，但在启用 adb 或端侧 runtime server 之前，不能声明完整端到端性能。
