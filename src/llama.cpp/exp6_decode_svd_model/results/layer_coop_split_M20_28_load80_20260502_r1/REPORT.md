# Layer Coop High-Load M=20..28 Sweep

日期：2026-05-02

## 说明

当前代码中 `split_m = M` 的含义是：

```text
PC:    [0, M)
Phone: [M, 28)
```

因此 `M` 越大，PC 计算层数越多，手机计算层数越少。若目标是“高负载时全量/大部分卸载到手机”，在当前定义下应关注小 M，例如 `M=2/4/6/8`，而不是大 M。

本报告仍按用户要求补充 `M=20,22,24,26,28`，PC load 为 `80%`，decode tokens 为 `32`，手机端开启 `LAYER_COOP_KEEP_HOT=1`。

## M=20..28 结果

| PC load | split M | PC layers | Phone layers | status | tok/s | ms/token | PC prefix ms/tok | Phone tail ms/tok | Net/sync ms/tok |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 80 | 20 | 20 | 8 | ok | 0.273 | 3665.99 | 3213.91 | 42.81 | 409.28 |
| 80 | 22 | 22 | 6 | ok | 0.245 | 4089.37 | 3810.50 | 54.27 | 224.59 |
| 80 | 24 | 24 | 4 | ok | 0.228 | 4386.49 | 4103.66 | 45.58 | 237.26 |
| 80 | 26 | 26 | 2 | ok | 0.212 | 4712.92 | 4395.94 | 26.27 | 290.72 |
| 80 | 28 | 28 | 0 | timeout > 360s |  |  |  |  |  |

结果显示，`M=20..26` 都非常慢，吞吐只有 `0.21-0.27 tok/s`；`M=28` 在 360 秒内未完成。原因是高 PC 负载下，大 M 让 PC 承担 20-28 层，PC prefix 成为绝对瓶颈。

## 低 M 对照：真正的大部分卸载到手机

同样 `load=80, tokens=32` 下，补测了 `M=0,2,4,6,8`。这里小 M 表示手机承担大部分层。

| PC load | split M | PC layers | Phone layers | status | tok/s | ms/token | PC prefix ms/tok | Phone tail ms/tok | Net/sync ms/tok |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 80 | 0 | 0 | 28 | client_rc_-6 |  |  |  |  |  |
| 80 | 2 | 2 | 26 | ok | 13.182 | 75.86 | 20.02 | 29.11 | 26.73 |
| 80 | 4 | 4 | 24 | ok | 5.053 | 197.91 | 86.54 | 32.26 | 79.10 |
| 80 | 6 | 6 | 22 | ok | 2.907 | 344.00 | 165.24 | 55.42 | 123.34 |
| 80 | 8 | 8 | 20 | ok | 2.087 | 479.05 | 204.78 | 65.56 | 208.71 |

`M=0` 当前会触发 prefix 空图问题：`graph nodes = 1`，随后 `tensor buffer not set` abort。因此当前 layer-coop 程序不支持 `M=0` 全量手机协同路径；全量手机应该单独用手机本地 full decode benchmark 测。

有效低 M 点中，`M=2` 最好：

```text
load=80, M=2, PC layers=2, Phone layers=26
throughput = 13.18 tok/s
PC prefix = 20.02 ms/token
Phone tail = 29.11 ms/token
Net/sync = 26.73 ms/token
```

这与 `M=20..28` 的低速形成鲜明对比，说明高 PC 负载时如果要继续协同，应该把 M 往小调，让手机承担大部分层。

## 结论

1. 按当前代码定义，`M=20..28` 不是“大部分卸载到手机”，而是“大部分留在 PC”。
2. 在 `80%` PC 负载下，`M=20..26` 都只有约 `0.21-0.27 tok/s`，`M=28` 超过 360 秒未完成。
3. 真正的大部分手机计算对应小 M；`M=2` 在本轮达到 `13.18 tok/s`，明显优于高 M。
4. 当前 `M=0` 不支持，会触发 prefix 空图崩溃；需要单独实现 bypass 或直接使用手机 full decode baseline 代表全量手机执行。
5. 下一步建议在 `load=80` 下对 `M=2,4,6` 做 repeats >= 5 的中位数验证。
