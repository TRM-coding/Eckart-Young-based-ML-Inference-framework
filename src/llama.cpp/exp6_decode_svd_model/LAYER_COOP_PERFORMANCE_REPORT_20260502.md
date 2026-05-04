# Layer-Level PC-Android Cooperative Inference Performance Report

日期：2026-05-02

本文档总结本轮 Qwen2.5-1.5B Q4 模型在 PC + Android 真机上的 layer-level cooperative inference 测速与排查结果。实验目标是验证按层切分后，PC 计算前半层、手机计算后半层时，实际 decode 吞吐是否符合理论预期，并定位当前协同路径低于理论值的原因。

## 结论摘要

当前 layer-level 协同路径的 layer split 本身是生效的：

- PC prefix-only `[0,14)` 单独连续 decode：`13.75 ms/token`，约 `72.74 tok/s`。
- Phone tail-only `[14,28)` 单独连续 decode：`14.08 ms/token`，约 `71.04 tok/s`。
- 因此，半模型单独运行确实接近用户预期，不存在“半模型实际仍跑完整模型”的问题。

但是，真实 TCP 协同路径是逐 token 严格串行：

```text
PC prefix decode -> TCP send hidden -> phone tail decode -> TCP return token -> PC next token
```

因此真实每 token 延迟不是 `max(prefix, tail)`，而是近似：

```text
PC prefix time + phone tail time + TCP / synchronization wait
```

本轮真实 TCP 协同 Q4/Q4、decode 128 tokens 的主要结果如下：

| 配置 | steady throughput | steady ms/token | PC prefix | Phone tail | Network / sync |
|---|---:|---:|---:|---:|---:|
| baseline TCP coop | `17.61 tok/s` | `56.79 ms` | `20.88 ms` | `18.56 ms` | `17.34 ms` |
| phone server keep-hot | `21.27 tok/s` | `47.01 ms` | `17.31 ms` | `15.58 ms` | `14.10 ms` |

其中 `phone server keep-hot` 指在手机端 server 等待下一次网络输入时启用轻量 keep-hot 线程，命令环境变量为：

```bash
LAYER_COOP_KEEP_HOT=1
```

该配置将手机 tail decode 从 `18.56 ms/token` 降到 `15.58 ms/token`，接近 tail-only microbench 的 `14.08 ms/token`。这证明协同路径中 PC/手机半模型变慢的主要原因不是 split 错误，而是每 token 一次 TCP 同步等待导致线程池、CPU 调度状态和执行热度被打断。

## 实验环境

### 代码路径

协同推理相关代码位于：

```text
src/llama.cpp/exp6_decode_svd_model/
```

核心文件：

```text
layer_coop_client.cpp
layer_mobile_server.cpp
layer_tail_local_bench.cpp
layer_coop_net.h
layer_threadpool.h
```

layer split 实现在 llama.cpp 的 Qwen2 graph builder 中：

```text
src/llama.cpp/3dparty/llamacpp/src/llama-model.cpp
```

相关参数来自：

```text
src/llama.cpp/3dparty/llamacpp/include/llama.h
```

关键字段：

```cpp
int32_t layer_split_start;
int32_t layer_split_end;
```

### 模型

本轮报告主测 Q4 模型：

```text
PC:    src/llama.cpp/gguf_models/qwen.q4_0.gguf
Phone: /data/local/tmp/CE_Ada/qwen.q4_0.gguf
```

模型校验：

```text
qwen.q4_0.gguf  md5 = 859bd3b74247cf351816e86c80f68d80
qwen.gguf       md5 = 5de3ccf1b9b365bccd542f6ac51382d1
```

手机端日志确认 Q4 AArch64 repack 路径生效：

```text
CPU_AARCH64 model buffer
repack tensor ... with q4_0_4x8
```

因此，本轮 Q4 慢速问题不是由误用 F16 模型、推错模型或未启用 AArch64 Q4 repack 导致。

### 设备与连接

手机通过 adb server port `5038` 连接：

```bash
adb -P 5038 devices
```

设备：

```text
10.20.0.3:5555 device
```

协同 TCP 通路：

```text
Phone -> PC 10.126.59.25:<port>
```

PC 侧测试时使用 `71-79` 核所在 cgroup，避免干扰其他实验：

```bash
sudo bash -lc 'CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; ...'
```

手机侧使用 8 线程，并绑定到 `0xff`：

```bash
LAYER_COOP_CPU_MASK=0xff taskset -a ff ...
```

本轮没有继续调整手机频率。频率相关排查在用户要求后停止。

## 已实现/调整的功能

### 1. steady decode 计时窗口

`layer_coop_client.cpp` 中增加了 steady decode 统计，排除以下开销：

- 模型加载；
- repack；
- full prefill；
- prefix prefill；
- server accept/connect 等待；
- 文本 detokenize 输出。

新增输出：

```text
[layer-coop-steady] generated=... steady_decode_ms=... steady_throughput=... steady_ms_per_token=... prefix_decode_ms=... server_decode_ms=... network_wait_ms=...
```

该行是当前报告中判断协同 decode 吞吐的主指标。

### 2. 手机端 token selection

手机端 server 已经在 tail decode 后本地完成 greedy token selection，只回传被选中的 token，不再回传完整 logits。

响应体只包含：

```text
selected_token
server_decode_ms
status
```

这将返回数据量从完整 vocab logits 降到每 token 约几十字节。

### 3. server 常驻与重连

手机端 `layer_mobile_server` 支持模型加载后常驻，并在连接断开后继续重连 PC endpoint。这样重复测试时无需每次重新加载模型。

### 4. phone server keep-hot

手机端支持：

```bash
LAYER_COOP_KEEP_HOT=1
```

在等待下一次网络输入时启动轻量 keep-hot 循环，减少每 token TCP 等待导致的手机侧线程/CPU 状态变冷。

该选项实测有效，是当前建议保留的优化。

### 5. client keep-hot 开关

PC client 侧新增了独立开关：

```bash
LAYER_COOP_CLIENT_KEEP_HOT=1
```

该开关只在 PC 等待手机返回 token 时启用 keep-hot。实测在本轮 8 线程、71-79 cgroup 环境下会与 PC prefix 计算抢资源，默认不建议开启。

注意：client keep-hot 与手机端 `LAYER_COOP_KEEP_HOT` 已经解耦。只设置手机端 `LAYER_COOP_KEEP_HOT=1` 不会触发 PC client keep-hot。

## 命令记录

### PC full Q4 baseline

```bash
sudo bash -lc 'set -e; CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; cd /home/tianruiming/CE_ADA_LLAMA; ./build-release-current/layer_tail_local_bench ./src/llama.cpp/gguf_models/qwen.q4_0.gguf 128 8 full_token 2>/tmp/pc_q4_full128.err'
```

结果：

```text
[tail-local-bench] mode=full_token tokens=128 threads=8 prefill_ms=44.6006 decode_ms=4089.6 avg_decode_ms=31.95 throughput=31.2989 tok/s
```

### PC prefix-only `[0,14)` microbench

```bash
sudo bash -lc 'set -e; CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; cd /home/tianruiming/CE_ADA_LLAMA; ./build-release-current/layer_tail_local_bench ./src/llama.cpp/gguf_models/qwen.q4_0.gguf 128 8 prefix_token 14 2>/tmp/pc_q4_prefix128.err'
```

结果：

```text
[tail-local-bench] mode=prefix_token tokens=128 threads=8 split_m=14 prefix_layers=[0,14) n_layer=28 prefill_ms=19.9504 decode_ms=1759.77 avg_decode_ms=13.7482 throughput=72.7368 tok/s
```

### Phone tail-only `[14,28)` microbench

```bash
adb -P 5038 shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff taskset -a ff ./layer_tail_local_bench ./qwen.q4_0.gguf 128 8 tail_embd 14 4 2>/tmp/phone_q4_tail128.err'
```

结果：

```text
[tail-local-bench] mode=tail_embd tokens=128 threads=8 split_m=14 pos_offset=4 tail_layers=[14,28) n_embd=1536 decode_ms=1801.71 avg_decode_ms=14.0759 throughput=71.0435 tok/s
```

### Baseline TCP coop, decode 128

PC client：

```bash
sudo bash -lc 'set -e; CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; cd /home/tianruiming/CE_ADA_LLAMA; ./build-release-current/layer_coop_client ./src/llama.cpp/gguf_models/qwen.q4_0.gguf 128 8 listen:18200 14 0'
```

Phone server：

```bash
adb -P 5038 shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff taskset -a ff ./layer_mobile_server ./qwen.q4_0.gguf 0 14 8 10.126.59.25:18200'
```

PC 输出：

```text
[layer-coop-client] generated=128 pc_prefill_ms=56.2789 prefix_decode_ms=2673.02 server_decode_ms=2375.53 total_ms=13845.4 throughput=9.24496 tok/s
[layer-coop-steady] generated=128 steady_decode_ms=7269.06 steady_throughput=17.6089 tok/s steady_ms_per_token=56.7895 prefix_decode_ms=2673.02 server_decode_ms=2375.53 network_wait_ms=2219.45
[layer-coop-profile] requests=128 send_bytes=790016 recv_bytes=4096 send_ms=13.6123 recv_header_ms=4581.37 roundtrip_ms=4594.98 client_wait_other_ms=2.72848e-12
```

Phone 输出：

```text
[layer-coop-server] requests=128 avg_header_wait_ms=35.0418 avg_payload_recv_ms=2.76758 avg_decode_ms=18.5588 avg_send_ms=0.0897848 recv_bytes=790016 send_bytes=4096
[layer-coop-server-final] requests=128 header_wait_ms=4485.35 payload_recv_ms=354.25 decode_ms=2375.53 send_ms=11.4925 recv_bytes=790016 send_bytes=4096
```

### TCP coop + phone server keep-hot, decode 128

PC client：

```bash
sudo bash -lc 'set -e; CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; cd /home/tianruiming/CE_ADA_LLAMA; ./build-release-current/layer_coop_client ./src/llama.cpp/gguf_models/qwen.q4_0.gguf 128 8 listen:18201 14 0'
```

Phone server：

```bash
adb -P 5038 shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff LAYER_COOP_KEEP_HOT=1 taskset -a ff ./layer_mobile_server ./qwen.q4_0.gguf 0 14 8 10.126.59.25:18201'
```

PC 输出：

```text
[layer-coop-client] generated=128 pc_prefill_ms=57.1322 prefix_decode_ms=2215.98 server_decode_ms=1994.5 total_ms=13868.5 throughput=9.22952 tok/s
[layer-coop-steady] generated=128 steady_decode_ms=6016.87 steady_throughput=21.2735 tok/s steady_ms_per_token=47.0068 prefix_decode_ms=2215.98 server_decode_ms=1994.5 network_wait_ms=1805.43
[layer-coop-profile] requests=128 send_bytes=790016 recv_bytes=4096 send_ms=11.3993 recv_header_ms=3788.53 roundtrip_ms=3799.93 client_wait_other_ms=1.36424e-12
```

Phone 输出：

```text
[layer-coop-server] requests=128 avg_header_wait_ms=29.4131 avg_payload_recv_ms=1.60494 avg_decode_ms=15.582 avg_send_ms=0.0913359 recv_bytes=790016 send_bytes=4096
[layer-coop-server-final] requests=128 header_wait_ms=3764.88 payload_recv_ms=205.432 decode_ms=1994.5 send_ms=11.691 recv_bytes=790016 send_bytes=4096
```

## 结果分析

### 1. layer split 没有跑错

Qwen2 graph builder 中 split 边界如下：

```cpp
const int layer_begin = std::max<int>(0, layer_split_start);
const int layer_end = layer_split_end < 0
    ? (int) n_layer
    : std::min<int>((int) n_layer, layer_split_end);

for (int il = layer_begin; il < layer_end; ++il) {
    ...
}
```

prefix context 设置：

```cpp
prefix_cp.embeddings = true;
prefix_cp.layer_split_start = 0;
prefix_cp.layer_split_end = split_m;
```

server tail context 设置：

```cpp
cp.layer_split_start = split_m;
cp.layer_split_end = -1;
```

microbench 也验证了实际速度：

| 路径 | 层范围 | avg ms/token | tok/s |
|---|---|---:|---:|
| PC full | `[0,28)` | `31.95` | `31.30` |
| PC prefix-only | `[0,14)` | `13.75` | `72.74` |
| Phone tail-only | `[14,28)` | `14.08` | `71.04` |

所以，协同中 PC prefix 和 phone tail 变慢不是因为 split 没生效。

### 2. 真实 TCP 协同是串行而非并行

当前协议每生成一个 token，需要完整完成一次：

```text
PC prefix decode
PC send hidden vector
Phone receive hidden vector
Phone tail decode + token selection
Phone send selected token
PC receive selected token
```

由于下一个 token 的 PC prefix 输入依赖手机返回的 token，所以 PC 无法在同一 token 级别与手机并行工作。

因此，当前协议的理论下界更接近：

```text
T_token >= T_pc_prefix + T_phone_tail + T_tcp_sync
```

而不是：

```text
T_token ~= max(T_pc_prefix, T_phone_tail)
```

也不是 PC 和手机 tok/s 简单相加。

### 3. 每 token 等待会让半模型 decode 变冷

baseline TCP coop 中：

```text
PC prefix:   20.88 ms/token
Phone tail:  18.56 ms/token
```

而连续 microbench 中：

```text
PC prefix-only:   13.75 ms/token
Phone tail-only:  14.08 ms/token
```

这说明 token 间的网络等待/同步等待会让两边的执行状态变冷。手机端 keep-hot 实验进一步验证了这一点：

```text
Phone tail baseline TCP coop:    18.56 ms/token
Phone tail with server keep-hot: 15.58 ms/token
Phone tail-only microbench:      14.08 ms/token
```

因此，keep-hot 不是在修改模型计算，而是在减少每 token 网络等待造成的调度/线程热度损失。

### 4. 网络/同步仍然是显著开销

即使 phone server keep-hot 后，`network_wait_ms` 仍为：

```text
1805.43 ms / 128 tokens = 14.10 ms/token
```

这里的 `network_wait_ms` 定义为：

```text
client roundtrip_ms - server_decode_ms
```

它包含：

- PC send hidden 的时间；
- 手机接收 header/payload 的时间；
- 手机发送 response 的时间；
- TCP 往返延迟；
- 两端同步等待与调度空档。

hidden payload 大小：

```text
n_embd = 1536
float32 hidden = 1536 * 4 = 6144 bytes/token
128 tokens send_bytes = 790016 bytes
```

返回 token 很小：

```text
128 tokens recv_bytes = 4096 bytes
```

因此返回 logits 已经不是主要问题；主要问题变成每 token 一次往返和同步等待。

## 已验证但不采用的优化

### 1. PC client keep-hot

尝试开启：

```bash
LAYER_COOP_CLIENT_KEEP_HOT=1
```

同时手机端也开启 `LAYER_COOP_KEEP_HOT=1`。

结果：

```text
steady_throughput = 17.6499 tok/s
steady_ms_per_token = 56.6576 ms/token
prefix_decode_ms = 2357.78 ms
server_decode_ms = 2797.69 ms
network_wait_ms = 2095.82 ms
```

该结果比只开手机 server keep-hot 的 `21.27 tok/s` 更差。原因推测是 PC keep-hot 线程与 prefix decode 共享 PC cgroup 核心，反而抢占资源。该选项保留为显式实验开关，但默认不使用。

### 2. server batch 复用 / 直接接收到 batch buffer

曾尝试在 `layer_mobile_server.cpp` 中复用 `llama_batch`，并将 hidden payload 直接接收到 `batch.embd`，避免每 token 分配 `std::vector` 和 `memcpy`。

真实 TCP 实测反而更差：

```text
steady_throughput = 15.7429 tok/s
steady_ms_per_token = 63.5205 ms/token
prefix_decode_ms = 2578.22 ms
server_decode_ms = 2250.14 ms
network_wait_ms = 3301.48 ms
```

server 侧 payload 接收时间明显变坏：

```text
avg_payload_recv_ms = 6.67187 ms
```

而原始路径 baseline TCP coop 中：

```text
avg_payload_recv_ms = 2.76758 ms
```

phone server keep-hot 中：

```text
avg_payload_recv_ms = 1.60494 ms
```

因此该优化已回滚，不纳入当前建议路径。

## 当前建议运行方式

### Q4/Q4 协同 decode 128 推荐命令

PC：

```bash
sudo bash -lc 'set -e; CG=/sys/fs/cgroup/tianruiming-exclusive/codex_7179_layer_profile; [ -d "$CG" ] && echo $$ > "$CG/cgroup.procs"; cd /home/tianruiming/CE_ADA_LLAMA; ./build-release-current/layer_coop_client ./src/llama.cpp/gguf_models/qwen.q4_0.gguf 128 8 listen:18201 14 0'
```

Phone：

```bash
adb -P 5038 shell 'cd /data/local/tmp/CE_Ada && LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff LAYER_COOP_KEEP_HOT=1 taskset -a ff ./layer_mobile_server ./qwen.q4_0.gguf 0 14 8 10.126.59.25:18201'
```

预期 steady 结果量级：

```text
steady_throughput ~= 21 tok/s
steady_ms_per_token ~= 47 ms/token
```

## 后续优化方向

### 1. 减少每 token TCP 往返

当前主要瓶颈已经不是 logits 回传，而是每 token 一次同步 round-trip。要接近 `40 tok/s`，必须减少或隐藏这部分同步时间。

可能方向：

- speculative decode；
- multi-token block protocol；
- pipeline 形式提前计算；
- 将 token selection / draft 逻辑放到其中一端，减少强同步。

### 2. 真正的异步流水

当前下一个 token 依赖手机返回 token，因此 PC 无法并行计算下一步 prefix。若要并行，需要改变生成策略，例如：

- 手机生成 draft token；
- PC 负责校验或修正；
- 或者以 block 为单位传输中间状态。

否则 layer split 的实际 token 延迟必然是各段串行相加。

### 3. 搜索更优 split M

当前测试使用：

```text
split_m = 14
prefix_layers = [0,14)
tail_layers = [14,28)
```

因为 tail 侧还包含：

- 后半 transformer layers；
- output norm；
- lm_head；
- greedy token selection。

所以理论最优 M 不一定是 14。可以对 `M = 8, 10, 12, 14, 16, 18, 20` 做 sweep，目标不是让 PC/phone 层数相等，而是让：

```text
T_pc_prefix(M) + T_phone_tail(M) + T_sync(M)
```

最小。

### 4. 优化 hidden 传输格式

当前 hidden 使用 float32：

```text
1536 * 4 = 6144 bytes/token
```

可以尝试：

- FP16 hidden；
- int8/quantized hidden；
- socket buffer / writev 减少分段发送；
- 预分配网络 buffer。

但从本轮结果看，单纯减少 payload 大小不一定足够，因为 `network_wait_ms` 中包含大量同步等待和调度空档。

### 5. 分离 benchmark 窗口

后续所有协同测速建议优先使用 `[layer-coop-steady]`，不要使用 `[layer-coop-client] throughput`，因为后者包含模型加载、prefill、连接等待等非 decode loop 开销。

## 最终结论

本轮实验确认：

1. layer-level split 实现正确，PC 和手机确实只运行各自层段。
2. 半模型连续运行速度符合预期，PC prefix-only 和 phone tail-only 都约 `14 ms/token`。
3. 真实 TCP 协同路径只有 `17.61 tok/s`，不是因为半模型算错，而是因为每 token 严格串行同步导致：
   - PC prefix decode 变慢；
   - phone tail decode 变慢；
   - 额外产生约 `17 ms/token` 的网络/同步等待。
4. 手机端 server keep-hot 可以将协同 steady throughput 从 `17.61 tok/s` 提升到 `21.27 tok/s`，是当前实测有效的优化。
5. 若目标是接近 `40 tok/s`，需要改变协议结构，减少或隐藏每 token round-trip，而不是继续只优化单次半模型 compute。

