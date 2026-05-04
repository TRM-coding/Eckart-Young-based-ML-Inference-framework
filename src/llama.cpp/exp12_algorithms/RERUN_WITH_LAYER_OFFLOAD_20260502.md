# 加入 Layer Offload 候选后的重跑报告

日期：2026-05-02

## 实验目的

本轮重跑用于验证新补齐的调度逻辑：

1. 先计算本地 SVD/no-SVD 的 DP 最优解。
2. 再读取 layer-level no-SVD 端侧卸载候选。
3. 最后比较两者预计端到端时延，选择更快者。

这里的 layer offload 使用 exp6 协同推理语义：

```text
PC    = [0, M)
Phone = [M, 28)
```

因此 `M` 越小，卸载到手机的层越多。

## 实验设置

- 模型：`src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`
- Decode 程序：`build-release-current/decode_svd_test`
- CPU 范围：隔离核心 `60-79` 中的 `60-67`
- 场景：`scenarios_q4_plot_2_4_6_8.json`，共 16 个异构负载场景
- 每个场景 repeats：3
- 每次 decode token 数：1
- 本轮不启动 adb，因此端侧路径只记录 scheduler 选择，不执行真实端到端 decode

加入 scheduler 的 layer-coop raw 数据：

```text
src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_load_20260502_r2/raw.csv
src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_M0_8_load80_20260502_r1/raw.csv
```

运行命令：

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
  src/llama.cpp/exp12_algorithms/benchmark_heterogeneous_schedule_cgroup.py \
  --scenarios-json src/llama.cpp/exp12_algorithms/scenarios_q4_plot_2_4_6_8.json \
  --repeats 3 \
  --tokens 1 \
  --timeout-s 240 \
  --layer-coop-raw src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_load_20260502_r2/raw.csv \
  --layer-coop-raw src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_M0_8_load80_20260502_r1/raw.csv \
  --out-dir src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502
```

## Scheduler 选择结果

下表使用 3 次重复实验的 decode-only tok/s 中位数。端侧路径因为本轮不使用 adb，所以 `Scheduler tok/s` 为空。

| 场景 | 核心数 | 每核负载 | Scheduler 模式 | M | Baseline tok/s | Scheduler tok/s | 本地 speedup |
|---|---:|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | 2 | `20,80` | `edge_end_no_svd` | 20 | 9.4757 | 未执行 | - |
| `2c_balanced_mid` | 2 | `50,50` | `edge_end_no_svd` | 20 | 11.2818 | 未执行 | - |
| `2c_high` | 2 | `80,100` | `edge_end_no_svd` | 2 | 11.3364 | 未执行 | - |
| `2c_idle_saturated` | 2 | `0,100` | `edge_end_no_svd` | 20 | 11.8377 | 未执行 | - |
| `4c_balanced_high` | 4 | `20,40,60,80` | `local` | 28 | 19.1139 | 19.1139 | 1.000x |
| `4c_front_hot` | 4 | `90,70,20,0` | `local` | 28 | 20.5645 | 20.3104 | 0.988x |
| `4c_gradient` | 4 | `0,20,60,100` | `local` | 28 | 16.4476 | 18.0706 | 1.099x |
| `4c_high` | 4 | `70,80,90,100` | `local` | 28 | 20.5410 | 17.6188 | 0.858x |
| `6c_front_hot` | 6 | `100,80,60,30,10,0` | `edge_end` | 22 | 4.9633 | 未执行 | - |
| `6c_high` | 6 | `50,60,70,80,90,100` | `local` | 28 | 27.3176 | 20.6636 | 0.756x |
| `6c_mixed` | 6 | `0,20,40,60,80,100` | `local` | 28 | 22.7898 | 20.4897 | 0.899x |
| `6c_ramp_light` | 6 | `0,10,20,30,40,50` | `local` | 28 | 24.5063 | 24.2307 | 0.989x |
| `8c_front_hot` | 8 | `100,90,80,70,30,20,10,0` | `edge_end` | 1 | 26.2005 | 未执行 | - |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `local` | 28 | 29.4303 | 24.3080 | 0.826x |
| `8c_mixed` | 8 | `0,20,40,60,80,100,30,50` | `local` | 28 | 21.3624 | 24.7582 | 1.159x |
| `8c_ramp_light` | 8 | `0,10,20,30,40,50,60,70` | `local` | 28 | 28.2094 | 0.2582 | 0.009x |

## 汇总

| 指标 | 数值 |
|---|---:|
| 总场景数 | 16 |
| 选择端侧路径的场景数 | 6 |
| 选择 `edge_end_no_svd` 的场景数 | 4 |
| 选择 legacy `edge_end` 的场景数 | 2 |
| 本地实际执行场景数 | 10 |
| 本地场景 speedup 中位数 | 0.943x |
| 本地场景最低 speedup | 0.009x |
| 本地场景低于 baseline 数 | 7 |

## 结论

本轮重跑证明，新加入的 offload 候选已经真正进入 scheduler 决策：2 核场景全部被路由到 `edge_end_no_svd`，并且高负载 `2c_high` 选择了低 M 的 `M=2`，符合“高负载时更多层卸载到手机”的方向。

但这轮结果也说明，当前自动 profile 仍不能作为最终策略：旧 matrix/latency model 对本地 SVD 路径过于乐观，导致多个场景仍被误判为 `local`，真实 decode 后出现负收益，甚至 `8c_ramp_light` 出现长尾低速。

因此当前状态应理解为：

- 调度算法结构已经补齐：可以在本地 SVD 和 no-SVD 端侧卸载之间做全局选择。
- offload profile 接口已经可用：`offload_candidates` 会被 `scheduler.py` 正确读取。
- 最终要让 scheduler 稳定“不低于 baseline”，还需要继续把本地 SVD 的 `total_ms` profile 替换为真实 runtime-calibrated profile，或者在 runtime 层加入更强的 fallback guard。

## 输出文件

```text
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502/raw.csv
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502/summary.csv
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502/schedule_summary.csv
src/llama.cpp/exp12_algorithms/results/heterogeneous_schedule_with_offload_rerun_20260502/REPORT.md
```
