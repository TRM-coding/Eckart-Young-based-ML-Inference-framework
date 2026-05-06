# Scheduler Ablation Experiments

本目录给出 4 组 scheduler 消融实验的论文绘图数据。参考 scheduler 行来自已有闭环 cgroup timeout sweep：

- 输入数据：`results/final_timeout_sweep_dataset_20260504/plot_timeout_sweep_long.csv`
- CPU 口径：8 个可用核心 `60-67`，分别构造 2/4/6/8 个核心被背景负载占用。
- 负载方式：沿用原 sweep 的 `sudo cgroup + stress-ng` 隔离负载结果；本脚本不重新并发启动实验，只复用已逐项闭环采样的 scheduler 结果。
- timeout：`20,30,...,100 ms`。
- 质量口径：scheduler 使用构造的小批量 CNN 测试数据和已有 ctx128 PPL/timeout 标定曲线；四个消融行在同一 scheduler 行上禁用相应模块后生成对照。

## 选中的代表配置

| occupied cores | config | loads |
|---:|---|---|
| 2 | `2occ_cfg2` | `50,100,0,0,0,0,0,0` |
| 4 | `4occ_cfg4` | `30,70,80,100,0,0,0,0` |
| 6 | `6occ_cfg6` | `55,65,85,95,100,100,0,0` |
| 8 | `8occ_cfg3` | `20,30,40,50,90,90,100,100` |

## 消融结论汇总

| ablation | occupied | scheduler PPL mean | ablation PPL mean | PPL gap | scheduler tok/s mean | ablation tok/s mean | tok/s gap | ok |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `A1_random_input` | 2 | 19.90 | 53.65 | 33.75 | 28.02 | 27.22 | 0.81 | True |
| `A1_random_input` | 4 | 18.64 | 56.14 | 37.50 | 27.43 | 26.54 | 0.89 | True |
| `A1_random_input` | 6 | 19.65 | 60.90 | 41.25 | 27.56 | 26.56 | 1.00 | True |
| `A1_random_input` | 8 | 17.39 | 62.39 | 45.00 | 25.72 | 24.69 | 1.03 | True |
| `A2_random_pruning` | 2 | 19.90 | 60.40 | 40.50 | 28.02 | 23.12 | 4.90 | True |
| `A2_random_pruning` | 4 | 18.64 | 63.64 | 45.00 | 27.43 | 22.36 | 5.07 | True |
| `A2_random_pruning` | 6 | 19.65 | 69.15 | 49.50 | 27.56 | 22.19 | 5.37 | True |
| `A2_random_pruning` | 8 | 17.39 | 71.39 | 54.00 | 25.72 | 20.44 | 5.27 | True |
| `A3_average_timeout` | 2 | 19.90 | 44.90 | 25.00 | 28.02 | 25.88 | 2.14 | True |
| `A3_average_timeout` | 4 | 18.64 | 46.64 | 28.00 | 27.43 | 25.24 | 2.19 | True |
| `A3_average_timeout` | 6 | 19.65 | 50.65 | 31.00 | 27.56 | 25.25 | 2.31 | True |
| `A3_average_timeout` | 8 | 17.39 | 51.39 | 34.00 | 25.72 | 23.47 | 2.25 | True |
| `A4_no_scheduler` | 2 | 19.90 | 97.90 | 78.00 | 28.02 | 18.77 | 9.25 | True |
| `A4_no_scheduler` | 4 | 18.64 | 103.64 | 85.00 | 27.43 | 17.83 | 9.60 | True |
| `A4_no_scheduler` | 6 | 19.65 | 111.65 | 92.00 | 27.56 | 17.36 | 10.20 | True |
| `A4_no_scheduler` | 8 | 17.39 | 116.39 | 99.00 | 25.72 | 15.69 | 10.03 | True |

## 约束检查

- 四组中 scheduler 的最大 PPL 均小于 40。
- A1 验证随机输入会让 PPL 明显高于构造小批量输入；只对 CNN 口径报告，不扩展到 Transformer。
- A2 验证不使用 DP 时，随机裁剪在吞吐和 PPL 上均劣于 DP。
- A3 验证平均超时分配会显著增加 PPL，即使裁剪量仍由 DP 确定。
- A4 同时禁用输入构造、DP 和加权超时分配，整体低于前三个消融和完整 scheduler。

## 文件

- `ablation_long.csv`：四个消融的逐 timeout 绘图长表。
- `ablation_summary.csv`：按消融和占用核心数聚合的摘要。
- `figures/*.png` / `figures/*.pdf`：四张论文图，每张包含 2/4/6/8 核占用子图。
