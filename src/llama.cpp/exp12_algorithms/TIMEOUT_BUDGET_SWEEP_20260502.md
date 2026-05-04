# Timeout Budget Sweep 重跑报告

日期：2026-05-02

## 目的

上一轮加入 layer offload 候选后，scheduler 仍然在部分本地 local SVD 场景中低于 baseline。这里固定其它实验条件不变，只调整 scheduler 的总 timeout budget，观察更大的 timeout 是否能改善本地 SVD 调度表现。

注意：这里的 timeout budget 是全局总预算，不是每层预算。`scheduler.py` 会把它按各层 `weight` 分配为每层 `timeout_ms`。

## 实验设置

- timeout budget：`8 ms`, `20 ms`, `40 ms`, `80 ms`
- 场景：`scenarios_q4_plot_2_4_6_8.json`，共 16 个异构负载场景
- 每个场景 repeats：3
- 每次 decode token 数：1
- Baseline：同一个 SVD-capable Q4 模型关闭 SVD，即 `baseline_no_svd`
- Scheduler 模型：`qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- 本轮不启动 adb，因此 `edge_end` / `edge_end_no_svd` 只记录选择，不执行真实端到端 decode

输出目录：

```text
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502/
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_timeout20_20260502/
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_timeout40_20260502/
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_timeout80_20260502/
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_timeout_sweep_20260502/
```

## 聚合结果

下表只统计实际执行了本地 scheduler 的场景。端侧 offload 场景本轮没有 adb，所以不计入本地 speedup。

| 总 timeout budget | 选择 offload 场景数 | 本地执行场景数 | 本地低于 baseline 数 | 本地最小 speedup | 本地中位 speedup | 本地平均 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 8 ms | 6 | 10 | 7 | 0.009x | 0.943x | 0.858x |
| 20 ms | 6 | 10 | 6 | 0.167x | 0.915x | 2.282x |
| 40 ms | 6 | 10 | 7 | 0.810x | 0.905x | 1.980x |
| 80 ms | 6 | 10 | 8 | 0.024x | 0.899x | 1.659x |

这里的平均 speedup 不应作为主结论，因为若 baseline 某次出现极端低速，speedup 会被异常放大。例如：

- `8c_mixed` 在 40 ms 下 baseline 中位只有 `0.2027 tok/s`，导致 speedup 被放大到 `11.47x`。
- `4c_gradient` 在 80 ms 下 baseline 中位只有 `1.9103 tok/s`，导致 speedup 被放大到 `9.45x`。
- `8c_ramp_light` 在 20 ms 下 baseline 中位只有 `2.4261 tok/s`，导致 speedup 被放大到 `9.01x`。

因此，本轮更可信的主指标是：

- 本地中位 speedup。
- 本地低于 baseline 的场景数。
- 每个场景是否稳定改善。

## 逐场景结果

| 场景 | 8 ms | 20 ms | 40 ms | 80 ms |
|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` |
| `2c_balanced_mid` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` |
| `2c_high` | `edge_end_no_svd M=2` | `edge_end_no_svd M=2` | `edge_end_no_svd M=2` | `edge_end_no_svd M=2` |
| `2c_idle_saturated` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` | `edge_end_no_svd M=20` |
| `4c_balanced_high` | 1.000x | 1.000x | 1.000x | 1.000x |
| `4c_front_hot` | 0.988x | 1.013x | 0.963x | 0.963x |
| `4c_gradient` | 1.099x | 0.803x | 1.216x | 9.454x |
| `4c_high` | 0.858x | 0.804x | 0.810x | 0.963x |
| `6c_front_hot` | `edge_end M=22` | `edge_end M=22` | `edge_end M=22` | `edge_end M=22` |
| `6c_high` | 0.756x | 0.924x | 0.855x | 0.859x |
| `6c_mixed` | 0.899x | 0.906x | 0.854x | 0.773x |
| `6c_ramp_light` | 0.989x | 0.167x | 0.941x | 0.940x |
| `8c_front_hot` | `edge_end M=1` | `edge_end M=1` | `edge_end M=1` | `edge_end M=1` |
| `8c_high` | 0.826x | 0.791x | 0.820x | 0.810x |
| `8c_mixed` | 1.159x | 7.396x | 11.472x | 0.804x |
| `8c_ramp_light` | 0.009x | 9.013x | 0.869x | 0.024x |

## 结论

放大 timeout budget 到 `20/40/80 ms` 并没有解决当前本地 scheduler 低于 baseline 的问题。

主要观察：

1. offload 选择数量没有变化。
   - 四组 timeout 下，均有 6 个场景选择 offload。
   - 说明 timeout budget 主要影响本地 SVD minor 等待，不影响当前 offload 候选的 ranking。

2. 本地中位 speedup 均低于 1。
   - 8 ms: `0.943x`
   - 20 ms: `0.915x`
   - 40 ms: `0.905x`
   - 80 ms: `0.899x`

3. 更大的 timeout 不等于更快。
   - timeout 增大后，runtime 可能等待 minor tail 更久。
   - 如果本地 split 本身被误选，放大 timeout 可能只是在扩大等待成本。

4. 高 speedup 个例主要来自 baseline 长尾低速，不应视为稳定提升。
   - 例如 20/40/80 ms 下若 baseline 偶发掉到极低 tok/s，speedup 会被异常放大。

因此，当前问题不应该继续靠调大 timeout budget 解决。更合理的下一步是：

- 对本地 SVD 路径加入 runtime guard：只有短校准显示本地 SVD 稳定快于 baseline 时才选择 local SVD。
- 或者重建本地 SVD `total_ms` profile，把真实 decode 的同步、线程池、tail 等待和长尾风险纳入模型。
- 在高竞争或高不确定场景中，优先 fallback 到 no-SVD baseline 或 layer offload，而不是继续依赖 matrix-latency profile。

## 数据文件

```text
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_timeout_sweep_20260502/timeout_sweep_summary.csv
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_timeout_sweep_20260502/timeout_sweep_wide.csv
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_timeout_sweep_20260502/timeout_sweep_aggregate.csv
```
