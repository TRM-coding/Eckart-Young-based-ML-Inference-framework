# 8c Extreme-high Load 诊断报告

## 问题

用户质疑：`8c_extreme_all_hot` / `8c_all_100` 在所有 8 个 PC 核心都有高负载时仍然能达到约 `28 tok/s`，是否是实验操作错误。

## 诊断方法

手动启动 8 个独立 `stress-ng` worker，每个 worker 通过 `taskset` 绑定到 `60-67` 中的一个核心，并放入 cgroup：

```text
/sys/fs/cgroup/ce_ada_llama_6079/diag_8c_all100_load
```

负载配置：

```text
CPU 60-67: 100,100,100,100,100,100,100,100
```

然后用 `mpstat -P 60,61,62,63,64,65,66,67 1 3` 检查核心利用率，再在同一负载窗口中运行：

```text
decode_svd_test qwen.q4_0.gguf 32 tokens, 8 threads, cpus=60-67
```

## 关键观测

`ps` 显示 `stress-ng-cpu` worker 确实绑定在 `60-67`：

```text
stress-ng-cpu [run] on PSR 60
stress-ng-cpu [run] on PSR 61
stress-ng-cpu [run] on PSR 62
stress-ng-cpu [run] on PSR 63
stress-ng-cpu [run] on PSR 64
stress-ng-cpu [run] on PSR 65
stress-ng-cpu [run] on PSR 66
stress-ng-cpu [run] on PSR 67
```

`mpstat` 在 decode 前显示 60-67 均为约 100% busy：

```text
CPU 60: 100% usr, 0% idle
CPU 61: 100% usr, 0% idle
CPU 62: 100% usr, 0% idle
CPU 63: 100% usr, 0% idle
CPU 64: 100% usr, 0% idle
CPU 65: ~99.7-100% busy
CPU 66: 100% usr, 0% idle
CPU 67: 100% usr, 0% idle
```

在该负载下，Q4 baseline decode 实测：

```text
status = ok
decode_tok_s = 27.8354 tok/s
generation_decode_ms = 1149.62 ms for 32 tokens
```

## 结论

`8c extreme-high` 下仍然有约 `28 tok/s`，不是因为 stress 负载没有启动，也不是因为核心绑定完全失效。独立诊断确认 60-67 核心确实被 `stress-ng` 打到 100% busy。

更可能的解释是：当前 `stress-ng --cpu-method matrixprod --cpu-load 100` 与 `decode_svd_test` 在 Linux 调度器下共享 CPU 时，decode 仍然能获得足够运行时间；也就是说这个负载模型并不能模拟“decode 被完全挤出 CPU”的情况。

因此：

1. 之前 `8c_all_100` 约 `28 tok/s` 的现象是真实可复现的。
2. 这个现象不代表“没有负载”，而是说明当前 stress-ng 负载模型对 decode 的干扰没有想象中那么强。
3. 如果目标是构造能迫使调度器选择 offloading 的 PC 端极端场景，需要更强的干扰方式，例如更高优先级/实时优先级负载、更多同核竞争 worker、内存带宽型负载，或直接限制 decode cgroup 的 CPU quota。
