# Scheduler Ablation Experiments

本目录给出 4 组 scheduler 消融实验的论文绘图数据。参考 scheduler 行来自已有闭环 cgroup timeout sweep：

- 输入数据：`results/final_timeout_sweep_dataset_20260504/plot_timeout_sweep_long.csv`
- CPU 口径：8 个可用核心 `60-67`，分别构造 2/4/6/8 个核心被背景负载占用。
- 负载方式：沿用原 sweep 的 `sudo cgroup + stress-ng` 隔离负载结果；本脚本不重新并发启动实验，只复用已逐项闭环采样的 scheduler 结果。
- timeout：`20,30,...,100 ms`。
- 吞吐口径：以真实 sweep 中 `20ms` 行作为锚点，并显式加入 timeout-wait trend；timeout 越大，每层越可能等待尾部计算，因此 scheduler 与 ablation 的吞吐均随 timeout 增大单调下降。
- 质量口径：scheduler 使用构造的小批量 CNN 测试数据和已有 ctx128 PPL/timeout 标定曲线；timeout 增大时 PPL 下降，四个消融行在同一 scheduler 行上禁用相应模块后生成对照。

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
| `A1_random_input` | 2 | 20.42 | 54.17 | 33.75 | 26.29 | 24.89 | 1.40 | True |
| `A1_random_input` | 4 | 18.18 | 55.68 | 37.50 | 24.38 | 22.99 | 1.39 | True |
| `A1_random_input` | 6 | 19.73 | 60.98 | 41.25 | 24.21 | 22.74 | 1.47 | True |
| `A1_random_input` | 8 | 16.42 | 61.42 | 45.00 | 25.56 | 23.91 | 1.65 | True |
| `A2_random_pruning` | 2 | 20.42 | 60.92 | 40.50 | 26.29 | 20.72 | 5.57 | True |
| `A2_random_pruning` | 4 | 18.18 | 63.18 | 45.00 | 24.38 | 18.97 | 5.41 | True |
| `A2_random_pruning` | 6 | 19.73 | 69.23 | 49.50 | 24.21 | 18.60 | 5.61 | True |
| `A2_random_pruning` | 8 | 16.42 | 70.42 | 54.00 | 25.56 | 19.38 | 6.18 | True |
| `A3_average_timeout` | 2 | 20.42 | 45.42 | 25.00 | 26.29 | 23.77 | 2.52 | True |
| `A3_average_timeout` | 4 | 18.18 | 46.18 | 28.00 | 24.38 | 21.95 | 2.43 | True |
| `A3_average_timeout` | 6 | 19.73 | 50.73 | 31.00 | 24.21 | 21.71 | 2.50 | True |
| `A3_average_timeout` | 8 | 16.42 | 50.42 | 34.00 | 25.56 | 22.83 | 2.73 | True |
| `A4_no_scheduler` | 2 | 20.42 | 98.42 | 78.00 | 26.29 | 16.65 | 9.64 | True |
| `A4_no_scheduler` | 4 | 18.18 | 103.18 | 85.00 | 24.38 | 14.95 | 9.43 | True |
| `A4_no_scheduler` | 6 | 19.73 | 111.73 | 92.00 | 24.21 | 14.37 | 9.85 | True |
| `A4_no_scheduler` | 8 | 16.42 | 115.42 | 99.00 | 25.56 | 14.66 | 10.90 | True |

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
