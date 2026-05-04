# SVD 矩阵 2-范数 Loss 测量报告

日期：2026-05-02

本文档说明 exp12 中 scheduler 使用的 `loss` 已经从手写启发式公式替换为真实 SVD 矩阵残差 2-范数测量。

## 原来的问题

此前 profile builder 中的 `loss` 是 placeholder：

```python
loss = (rate * rate) * (1.0 + 0.015 * layer)
```

它只表达“裁剪率越大、层越靠后，惩罚越大”这个人工偏好，不是精度损失测量，也不是矩阵范数测量。

涉及文件包括：

```text
src/llama.cpp/exp12_algorithms/run_exp12_local.py
src/llama.cpp/exp12_algorithms/build_model_profile.py
src/llama.cpp/exp12_algorithms/build_additive_profile.py
```

## 新的 Loss 定义

按照当前要求，loss 定义为：

```text
经过 SVD 裁剪丢弃后的残差矩阵与原始矩阵之差的 2-范数
```

即对矩阵 `W` 的 SVD：

```text
W = U diag(sigma) V^T
```

当 rate 表示丢弃尾部 rank 比例时：

```text
k_trunc = ceil(rate * rank)
k_keep = rank - k_trunc
```

裁剪后：

```text
W_keep = U[:, :k_keep] diag(sigma[:k_keep]) V[:k_keep, :]
```

残差：

```text
R = W - W_keep
```

矩阵 2-范数：

```text
loss = ||R||_2
```

因为 SVD 奇异值已经按降序排列，所以：

```text
||R||_2 = sigma[k_keep]
```

如果 `rate = 0`，没有丢弃 rank，loss 为 `0`。

## 测量方法

新增脚本：

```text
src/llama.cpp/exp12_algorithms/measure_svd_matrix_loss.py
```

测量命令：

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
  src/llama.cpp/exp12_algorithms/measure_svd_matrix_loss.py \
  --model src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf \
  --rates 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9 \
  --layer-reduction sum \
  --out-dir src/llama.cpp/exp12_algorithms/results/svd_matrix_loss_qwen_f16_20260502_r1
```

这里使用的是 F16 compact SVD GGUF：

```text
src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf
```

没有使用 Q4 量化后的 SVD 因子来测 loss，因为该 loss 的目标是描述 SVD 裁剪本身的矩阵残差；Q4 因子会额外引入量化误差，不适合作为纯 SVD truncation loss。

## SVD 因子说明

`exp7_generate_sort_svd/generate_sort_svd.py` 生成 SVD GGUF 时，将原始矩阵分解为：

```python
u, s, vh = torch.linalg.svd(weight, full_matrices=False)
u_factor = u * sqrt(s)
v_factor = sqrt(s) * vh
```

因此脚本不需要重新对大矩阵做 SVD，而是从因子范数恢复奇异值：

```text
sigma_i = ||u_factor[:, i]||_2^2
sigma_i = ||v_factor[i, :]||_2^2
```

实际实现中对两边估计取平均，以减小 F16 存储误差。

## 输出文件

测量结果目录：

```text
src/llama.cpp/exp12_algorithms/results/svd_matrix_loss_qwen_f16_20260502_r1/
```

核心文件：

| 文件 | 内容 |
|---|---|
| `svd_matrix_loss.csv` | 每层、每个 FFN 矩阵、每个 rate 的残差 2-范数 |
| `svd_layer_loss.csv` | 每层对 `up/gate/down` 三个矩阵 loss 做聚合后的结果 |
| `svd_loss_for_scheduler.csv` | 给 profile builder 使用的 layer-level loss 表 |
| `svd_singular_summary.csv` | 每个矩阵的奇异值概要 |
| `SVD_MATRIX_LOSS_REPORT.md` | 测量摘要 |

本轮测量规模：

| 项目 | 数量 |
|---|---:|
| 层数 | 28 |
| 每层 FFN 矩阵 | 3 |
| 矩阵总数 | 84 |
| rate 数 | 11 |
| 矩阵级记录 | 924 |
| 层级记录 | 308 |

## 数值示例

layer 0 的 layer-level loss，也就是 `up + gate + down` 三个矩阵残差 2-范数求和：

| rate | loss |
|---:|---:|
| 0.0 | 0.0000 |
| 0.1 | 3.2003 |
| 0.2 | 4.6577 |
| 0.3 | 5.6062 |
| 0.4 | 6.4087 |
| 0.5 | 7.2166 |
| 0.6 | 8.0132 |
| 0.7 | 8.8498 |
| 0.8 | 9.7897 |
| 0.85 | 10.3440 |
| 0.9 | 10.9908 |

全层统计：

| rate | layer loss 中位数 | layer loss 最小值 | layer loss 最大值 |
|---:|---:|---:|---:|
| 0.1 | 4.7444 | 3.2003 | 5.3791 |
| 0.5 | 7.5311 | 6.9564 | 7.9159 |
| 0.8 | 9.5523 | 9.1003 | 10.2220 |
| 0.85 | 9.9910 | 9.5448 | 10.7034 |
| 0.9 | 10.6113 | 10.0830 | 11.2608 |

## 接入方式

新增公共工具：

```text
src/llama.cpp/exp12_algorithms/loss_table.py
```

以下 profile builders 已支持 `--loss-table`：

```text
src/llama.cpp/exp12_algorithms/build_model_profile.py
src/llama.cpp/exp12_algorithms/build_additive_profile.py
src/llama.cpp/exp12_algorithms/run_exp12_local.py
```

示例：

```bash
/home/tianruiming/miniconda3/envs/pytorch/bin/python \
  src/llama.cpp/exp12_algorithms/build_model_profile.py \
  --loss-table src/llama.cpp/exp12_algorithms/results/svd_matrix_loss_qwen_f16_20260502_r1/svd_loss_for_scheduler.csv \
  --loss-reduction sum \
  --rates 0,0.5,0.8 \
  --loads 50 \
  --run-scheduler \
  --out-dir src/llama.cpp/exp12_algorithms/results/model_profile_measured_loss_smoke_20260502
```

验证结果显示，新 profile 中：

```text
loss_source = measured spectral residual norm
```

并且 layer 0、rate 0.5 的 loss 已经从旧启发式的 `0.25` 变为实测值 `7.21660329`。

## 当前限制

1. 当前 loss 是矩阵级残差 2-范数，不是 PPL、KL 或最终 logits loss。
2. layer-level loss 默认使用 `up/gate/down` 三个矩阵的 sum 聚合。
3. 该 loss 不依赖输入激活分布，因此它描述的是权重矩阵裁剪误差，而不是数据分布下的输出误差。
4. 该 loss 使用 F16 SVD 因子测量；如果后续要研究 Q4 量化误差，需要单独测量 quantized SVD factor loss。

因此，当前 DP 的 loss 已经从“人工公式”升级为“真实 SVD 矩阵残差 2-范数”。如果论文需要进一步说明任务级质量损失，还需要额外做 PPL 或 logit KL 验证。
