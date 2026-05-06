# ResNet50 极端负载下真实协同推理实验报告

日期：2026-05-06

## 实验目标

复现 LLaMA layer-level coop 图的同类实验，但换成 ResNet50：

- PC 在极端负载下运行 ResNet50。
- 比较 PC baseline、PC 侧 No-Offloading 选核策略。
- 实现并测试真正的 ResNet50 split 协同推理，而不是把手机 full inference 冒充成协同。

## 实现内容

新增 ResNet50 协同推理实现：

- `src/llama.cpp/exp8_resnet50/resnet_coop_client.cpp`
- `src/llama.cpp/exp8_resnet50/resnet_coop_server.cpp`
- `src/llama.cpp/exp8_resnet50/resnet_coop_net.h`

同时修改：

- `src/llama.cpp/exp8_resnet50/run_resnet50.cpp`
- `src/llama.cpp/CMakeLists.txt`

协同语义：

```text
PC:
  stem + maxpool + 前 M 个 bottleneck blocks
  将中间 feature map 发送给手机

Phone:
  从第 M 个 bottleneck block 继续执行到 classifier
  返回 top-1 class id
```

因此图中的 `M=12/14/16` 是真实 ResNet50 block split，不是手机 full inference。

## 构建与部署

桌面端已编译：

```text
build-release-current/resnet_coop_client
build-release-current/resnet_coop_server
build-release-current/run_resnet50
```

Android 端已编译并推送：

```text
/data/local/tmp/CE_Ada/resnet/resnet_coop_server
/data/local/tmp/CE_Ada/resnet/resnet50-f32.gguf
/data/local/tmp/CE_Ada/resnet/test.JPEG
```

ADB 设备：

```text
adb -P 5038 devices
10.20.0.3:5555  device
model: PLK110
abi: arm64-v8a
```

## 实验设置

- PC 核心：`60-67`
- PC 背景负载：每个核心 2 个 `stress-ng --cpu-load 100 --cpu-method matrixprod`
- 测试图像：ImageNet val 子集第一张，`ILSVRC2012_val_00000001.JPEG`
- ResNet50 模型：`resnet50-f32.gguf`
- 手机端线程：8
- PC 协同 prefix 线程：4
- 协同 split：`M=12,14,16`
- 每个策略测 1 次 forward

## 实验数据

原始数据：

- `raw.csv`

| 策略 | 吞吐量 |
|---|---:|
| PC基线 | 0.382 img/s |
| No-Offloading 2核 | 0.919 img/s |
| No-Offloading 4核 | 0.568 img/s |
| No-Offloading 6核 | 0.408 img/s |
| 协同推理 M=12 | 0.202 img/s |
| 协同推理 M=14 | 0.198 img/s |
| 协同推理 M=16 | 0.327 img/s |

协同内部耗时：

| Split | prefix_ms | send_ms | wait_ms | server_ms |
|---:|---:|---:|---:|---:|
| M=12 | 1929.94 | 717.43 | 2308.52 | 1078.73 |
| M=14 | 2343.12 | 484.74 | 2213.25 | 512.75 |
| M=16 | 2551.52 | 255.87 | 249.11 | 3.97 |

## 图

- `figures/resnet_extreme_coop_vs_no_offloading.png`
- `figures/resnet_extreme_coop_vs_no_offloading.pdf`

## 结论

ResNet50 的真实 split 协同推理已经实现并跑通，且返回的 top-1 正确：

```text
top1 = 65, sea snake
```

但是当前真实数据不能支持“协同推理优于 No-Offloading”的结论：

```text
最佳 No-Offloading: 0.919 img/s
最佳协同推理:      0.327 img/s
```

也就是说，ResNet50 当前实现下：

```text
协同推理 M=16 / PC基线 = 0.856x
协同推理 M=16 / 最优 No-Offloading = 0.356x
```

主要原因：

1. PC prefix 在极端负载下仍然很慢，`M=12/14/16` 的 prefix 都在秒级。
2. Android 端 ResNet50 server 当前走 Android native/ggml 路径，没有 oneDNN；它和 PC 端优化后的 oneDNN ResNet50 不是同级实现。
3. ResNet50 feature map 传输比 LLaMA 单 token hidden/state 更重，早期 split 尤其不划算。
4. 对 ResNet50 来说，No-Offloading 选择低负载核心子集更直接有效。

因此，这张图不能被处理成和 LLaMA 那张图相同的“协同推理显著优于 No-Offloading”的结论。若论文需要 ResNet50 上也展示协同优势，需要进一步优化 Android 端 ResNet tail，例如：

- 给 Android 端实现更快的 ResNet50 tail kernel；
- 避免发送大 feature map；
- 选择更合适的 split 粒度；
- 或者实现端侧专用 oneDNN/NNAPI/GPU 后端。

当前可以可信报告的结论是：

```text
ResNet50 协同 split 已实现且功能正确；
但在当前 Android CPU 实现下，极端负载时最佳策略仍是 No-Offloading 选核，而不是协同卸载。
```
