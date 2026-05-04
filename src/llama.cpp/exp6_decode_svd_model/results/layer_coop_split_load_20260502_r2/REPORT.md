# Layer Coop Split vs PC Load Sweep

日期：2026-05-02

本实验在 PC 侧不同 CPU 占用率下，枚举 layer split 点 `M`，测试 Qwen2.5-1.5B Q4 模型的 PC + Android layer-level cooperative decode 效果。

当前协议中：

```text
PC 计算 [0,M)
Phone 计算 [M,28)
```

本轮实验使用真实 TCP 通路，手机端开启：

```bash
LAYER_COOP_KEEP_HOT=1
```

## 实验设置

- 模型：`qwen.q4_0.gguf`
- decode tokens：`64`
- PC 线程数：`8`
- Phone 线程数：`8`
- PC cgroup/cpus：`71-79`
- Phone adb：`adb -P 5038`
- Phone server：`/data/local/tmp/CE_Ada/layer_mobile_server`
- PC client：`build-release-current/layer_coop_client`
- 负载生成：`stress-ng --cpu 9 --cpu-load <load> --cpu-method matrixprod`
- 负载和 client 均限制在 `71-79` 这组核心上

原始数据：

```text
src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_load_20260502_r2/raw.csv
```

补充高 M 数据：

```text
src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_load_highM_20260502_r1/raw.csv
```

注意：高负载下单次 run 抖动很大，本报告将 `50%/80%` 负载结果视作趋势探测，而不是稳定最终数值。后续需要对候选 split 做多次重复并取中位数。

## 主 Sweep 结果

| PC load | split M | tok/s | ms/token | PC prefix ms/token | Phone tail ms/token | Net/sync ms/token |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 19.64 | 50.92 | 12.36 | 24.29 | 14.25 |
| 0 | 12 | 21.30 | 46.95 | 14.86 | 17.54 | 14.54 |
| 0 | 16 | 16.53 | 60.49 | 19.96 | 24.22 | 16.31 |
| 0 | 20 | 18.67 | 53.56 | 23.28 | 15.94 | 14.34 |
| 50 | 8 | 0.68 | 1479.44 | 1210.60 | 45.77 | 223.05 |
| 50 | 12 | 0.50 | 2004.65 | 1739.83 | 56.52 | 208.30 |
| 50 | 16 | 0.69 | 1447.03 | 1260.20 | 34.16 | 152.66 |
| 50 | 20 | 16.12 | 62.03 | 23.20 | 22.85 | 15.98 |
| 80 | 8 | 0.69 | 1450.06 | 1117.63 | 46.38 | 286.04 |

## 补充高 M 结果

补充实验将 decode tokens 降为 `32`，只看高负载下继续增大 M 是否有帮助。结果显示该条件下单次波动极强，因此只作为 instability 证据。

| PC load | split M | tokens | tok/s | ms/token | PC prefix ms/token | Phone tail ms/token | Net/sync ms/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 20 | 32 | 0.30 | 3363.27 | 3048.27 | 35.47 | 279.52 |
| 50 | 24 | 32 | 0.21 | 4828.30 | 4508.09 | 41.97 | 278.24 |

该补充实验和主 sweep 中 `load=50, M=20` 的结果冲突：

```text
主 sweep:     load=50, M=20, tokens=64 -> 16.12 tok/s
补充 sweep:   load=50, M=20, tokens=32 -> 0.30 tok/s
```

这说明在 PC stress-ng 与 client 共享 `71-79` 核时，系统调度抖动足以让单次结果失真。高负载下必须使用多次重复和中位数，而不能依赖单次测量。

## 初步观察

### 空载最优 split

空载下，当前测试点中最优为：

```text
M = 12
throughput = 21.30 tok/s
```

空载下各项开销：

```text
PC prefix:   14.86 ms/token
Phone tail:  17.54 ms/token
Net/sync:    14.54 ms/token
Total:       46.95 ms/token
```

与此前 `M=14` 的 128-token 结果 `21.27 tok/s` 接近，说明空载情况下最优 split 大概在 `M=12-14` 附近，而不一定正好是 `14`。

### 中等 PC 负载下，split 应明显右移

`50%` 负载下，`M=8/12/16` 单次结果都出现灾难性低速，主要是 PC prefix 被压垮：

```text
M=8:  PC prefix 1210.60 ms/token
M=12: PC prefix 1739.83 ms/token
M=16: PC prefix 1260.20 ms/token
```

但同一轮中：

```text
M=20: 16.12 tok/s
```

这说明在 PC 核心受到明显干扰时，应把更多层分给手机，即增大 `M`，减少 PC 每 token 的 prefix 计算量。

### 高负载下结果极不稳定

`80%` 负载下仅完成了 `M=8`：

```text
M=8: 0.69 tok/s
```

PC prefix 达到：

```text
1117.63 ms/token
```

结合补充高 M 实验的强烈波动，可以判断：

- 在 `71-79` 上直接用 stress-ng 压同一组核心，会引入非常强的调度抖动；
- 单次 run 不能代表真实性能；
- 高负载实验需要更严格的重复测量、独立 load cgroup 监控，以及可能更短/固定 warmup 后再计时。

## 当前建议

### 1. 空载或轻负载

优先测试：

```text
M = 12, 14
```

当前单次结果：

```text
M=12: 21.30 tok/s
M=14: 之前 128-token 实验约 21.27 tok/s
```

### 2. 中等 PC 负载

优先测试：

```text
M = 18, 20, 22
```

本轮 `M=20` 有一次可用结果：

```text
load=50, M=20: 16.12 tok/s
```

但重复实验出现过极慢点，因此需要至少 `3-5` 次重复取中位数。

### 3. 高 PC 负载

不建议继续大网格单次 sweep。建议先固定候选：

```text
M = 20, 22, 24, 26
```

每个点跑：

```text
tokens = 32 或 64
repeats >= 5
```

并报告：

```text
median tok/s
p25/p75
min/max
```

否则单次调度抖动会掩盖 split 的真实影响。

## 方法学问题与脚本修正

本轮新增脚本：

```text
src/llama.cpp/exp6_decode_svd_model/benchmark_layer_coop_split_load.py
```

脚本功能：

- 创建 run/load cgroup；
- 启动 PC stress-ng；
- 启动 PC `layer_coop_client`；
- 启动 Android `layer_mobile_server`；
- 解析 `[layer-coop-steady]` 和 `[layer-coop-profile]`；
- 输出 `raw.csv` 和 `REPORT.md`。

实验中发现初版脚本沿用了 exp12 的全局：

```text
pkill -x stress-ng
```

这可能影响机器上的其他实验。脚本已经修正为只清理本脚本持有的进程组和 cgroup，不再全局 kill `stress-ng`。

## 初步结论

1. PC 空载时，当前最优 split 在 `M=12-14` 附近，协同速度约 `21 tok/s`。
2. 当 PC 侧核心出现明显负载时，最优 split 应向更大的 `M` 移动，即让手机承担更多层。
3. `load=50, M=20` 在一次 64-token sweep 中达到 `16.12 tok/s`，明显优于同负载下 `M=8/12/16` 的灾难性低速点。
4. 高负载下单次结果极不稳定，必须改用重复实验中位数；当前数据不能直接用于最终论文/报告结论。
5. 下一步应围绕候选 split 做窄范围、多重复验证：

```text
load=0:   M=10,12,14,16
load=50:  M=18,20,22
load=80:  M=20,22,24,26
```

