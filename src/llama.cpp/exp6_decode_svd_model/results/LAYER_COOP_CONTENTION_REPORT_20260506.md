# 真实协同推理竞争实验报告

日期：2026-05-06

## 结论先行

这次重新按 layer-level 协同推理工具链测量后，可以确认：之前把 `edge_end` 跑到很高 tok/s 的数据不是协同速度，而是手机端 full decode，不能用于证明 PC + 手机协同卸载。

本轮使用真正的协同链路：

```text
PC:    layer_coop_client，计算 [0, M)
Phone: layer_mobile_server，计算 [M, 28)
```

在 8 个 PC 核心 `72-79` 都受到强竞争时，真实协同可以明显超过 PC baseline，也可以超过 major-only：

| 场景 | PC full baseline | 最优 major-only | 最优真实协同 | 相对 baseline | 相对 major-only |
|---|---:|---:|---:|---:|---:|
| 空载，1 worker/core 参数但 load=0 | 51.49 tok/s | 44.51 tok/s | 26.24 tok/s, M=4 | 0.51x | 0.59x |
| 80% load, 1 worker/core | 0.54 tok/s | 6.66 tok/s | 3.10 tok/s, M=2 | 5.76x | 0.47x |
| 100% load, 2 workers/core | 0.37 tok/s | 7.96 tok/s | 12.93 tok/s, M=2 | 35.36x | 1.62x |

因此调度器的策略边界应该是：

- 空载/轻载：不协同，PC full baseline 最快。
- 高负载但还存在少量可用 PC 算力：major-only 可能优于协同。
- 极端负载、所有 PC 核都被强竞争时：触发真实协同 `M=2`，可以超过 baseline 和 major-only。

## 为什么之前的竞争实验不可信

之前使用 `decode_svd_test` 测 PC baseline/SVD 时，负载没有真正把 baseline 打下来，根本原因是这个程序自身把优先级拉得很高：

- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp:34) 里实现了 `set_highest_process_priority()`，Linux 下调用 `setpriority(..., -20)`。
- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp:192) 启动时直接调用这个函数。
- [decode_svd_model.cpp](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/decode_svd_model.cpp:317) 还把 ggml threadpool 设置成 `GGML_SCHED_PRIO_REALTIME`。

也就是说，普通 `stress-ng` 和它放在同一组核心上时，未必能公平竞争；这解释了为什么之前有些“8 核全高负载”下 baseline 还看起来很高。

本轮改用 `layer_tail_local_bench` 和 `layer_coop_client/layer_mobile_server` 做同口径测量。它们默认是普通优先级，相关逻辑在 [layer_threadpool.h](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/layer_threadpool.h:102)，默认返回 `GGML_SCHED_PRIO_NORMAL`。因此同核 stress-ng 能真正抢到 CPU。

## 实验设置

- PC 核心：`72-79`，共 8 个核心。
- PC 模型：`src/llama.cpp/gguf_models/qwen.q4_0.gguf`。
- 手机模型：`/data/local/tmp/CE_Ada/qwen.q4_0.gguf`。
- adb：`adb -P 5038 -s 10.20.0.3:5555`。
- 手机服务：`LAYER_COOP_KEEP_HOT=1 ./layer_mobile_server`。
- 负载生成：每个 PC 核单独启动 `stress-ng --cpu 1 --cpu-load <load> --cpu-method matrixprod`，并绑定到同一个核心。
- PC baseline：`layer_tail_local_bench ... full_token`。
- major-only：只在 PC 子集核心上跑 `full_token`，例如 `72-73`、`72-75`、`72-77`。
- 真实协同：PC `layer_coop_client` + 手机 `layer_mobile_server`，枚举 split M。

## 原始数据

完整 raw 数据：

- [80% 竞争与空载对照 raw.csv](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/layer_coop_contention_focused_20260506_r1/raw.csv)
- [极端竞争 raw.csv](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/layer_coop_contention_extreme_20260506_r1/raw.csv)
- [汇总 CSV](/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/exp6_decode_svd_model/results/layer_coop_contention_summary_20260506.csv)

## 关键结果解释

### 1. 空载时不应该卸载

空载时：

```text
PC full baseline: 51.49 tok/s
best split:       26.24 tok/s, M=4
```

这说明空载场景下协同会引入网络/同步开销，不应该触发卸载。调度器在这种场景应选择 no-offloading baseline。

### 2. 80% 同核竞争时，baseline 会被打到几乎不可用

80% load、每核 1 个 worker：

```text
PC full baseline: 0.54 tok/s
best major-only:  6.66 tok/s
best split:       3.10 tok/s, M=2
```

这个结果说明两个问题：

- baseline 确实被真实竞争打下来了；
- 但此时 major-only 仍优于协同，所以调度器不应该盲目卸载，而应该比较 major-only 与 split。

### 3. 极端竞争下协同超过 baseline 和 major-only

100% load、每核 2 个 worker：

```text
PC full baseline: 0.37 tok/s
best major-only:  7.96 tok/s
best split:       12.93 tok/s, M=2
```

真实协同 `M=2` 的详细开销：

```text
steady_ms_per_token: 77.36 ms/token
PC prefix total:     305.82 ms / 16 tokens
Phone tail total:    607.23 ms / 16 tokens
Network/sync total:  324.62 ms / 16 tokens
```

这个场景满足你要求的三点：

- scheduler 可以触发协同卸载；
- PC 负载足够大，baseline 几乎不可用；
- 协同速度 `12.93 tok/s` 大于 baseline `0.37 tok/s`，也大于 major-only 最优 `7.96 tok/s`。

## 对调度器的建议

调度器不应该把 `edge_end` 理解成手机 full decode；论文里要展示的是 layer-level 协同，应使用 split coop 的预测/实测值。

调度决策应至少比较三类候选：

```text
1. no-offloading PC full baseline
2. major-only no-SVD：只用低负载 PC 核心
3. layer-level coop：PC [0,M)，Phone [M,28)
```

当前实验证明，第三类策略只在极端 PC 竞争下有优势；这正好可以作为论文中的“极端负载触发协同卸载”证据。
