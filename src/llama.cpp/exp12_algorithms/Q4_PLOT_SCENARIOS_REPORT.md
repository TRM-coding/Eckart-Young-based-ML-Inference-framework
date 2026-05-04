# Q4 多核心占用场景调度效果报告

日期：2026-05-02

本文档补充一组更适合画图的数据：分别构造 2、4、6、8 个核心被占用时的代表性异构负载场景，每个核心数下 4 组负载，共 16 个场景。实验目标是观察 Q4 scheduler 相对原始 `qwen.q4_0.gguf` baseline 的吞吐提升。

## 实验设置

- 隔离 CPU 集合：`60-79`。
- 实际实验使用 `60-67` 的子集。
- 背景负载：使用 `stress-ng --cpu-load` 对每个核心分别设置负载。
- 原始 baseline 模型：`src/llama.cpp/gguf_models/qwen.q4_0.gguf`。
- Scheduler 模型：`src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`。
- Decode 程序：`build-release-current/decode_svd_test`。
- Decode tokens：`1`。
- Scheduler calibration repeats：`1`。
- Scheduler validation repeats：`5`。
- 原始 Q4 baseline repeats：`5`。
- 本轮不使用 adb，不验证端侧卸载路径。

本轮 scheduler 没有强制沿用之前的 final policy map，而是在 16 个新场景上重新 runtime calibration，并从候选策略中选择：

```text
baseline_no_svd,trunc_even_0.6,trunc_even_0.8,trunc_even_0.85,trunc_even_0.9
```

## 场景设计

| 核心数 | 场景 | CPU | 每核负载 |
|---:|---|---|---|
| 2 | `2c_balanced_mid` | `60,61` | `50,50` |
| 2 | `2c_asym_low_high` | `60,61` | `20,80` |
| 2 | `2c_idle_saturated` | `60,61` | `0,100` |
| 2 | `2c_high` | `60,61` | `80,100` |
| 4 | `4c_gradient` | `60,61,62,63` | `0,20,60,100` |
| 4 | `4c_balanced_high` | `60,61,62,63` | `20,40,60,80` |
| 4 | `4c_front_hot` | `60,61,62,63` | `90,70,20,0` |
| 4 | `4c_high` | `60,61,62,63` | `70,80,90,100` |
| 6 | `6c_ramp_light` | `60,61,62,63,64,65` | `0,10,20,30,40,50` |
| 6 | `6c_mixed` | `60,61,62,63,64,65` | `0,20,40,60,80,100` |
| 6 | `6c_front_hot` | `60,61,62,63,64,65` | `100,80,60,30,10,0` |
| 6 | `6c_high` | `60,61,62,63,64,65` | `50,60,70,80,90,100` |
| 8 | `8c_ramp_light` | `60,61,62,63,64,65,66,67` | `0,10,20,30,40,50,60,70` |
| 8 | `8c_mixed` | `60,61,62,63,64,65,66,67` | `0,20,40,60,80,100,30,50` |
| 8 | `8c_front_hot` | `60,61,62,63,64,65,66,67` | `100,90,80,70,30,20,10,0` |
| 8 | `8c_high` | `60,61,62,63,64,65,66,67` | `30,40,50,60,70,80,90,100` |

## 主要结果

下表使用 5 次重复实验的 decode-only tok/s 中位数。`Scheduler / Original` 是最重要的画图指标，表示 scheduler 相对原始 `qwen.q4_0.gguf` baseline 的吞吐倍率。

| 核心数 | 场景 | 策略 | Original Q4 tok/s | SVD Q4 no-SVD tok/s | Scheduler tok/s | Scheduler / Original |
|---:|---|---|---:|---:|---:|---:|
| 2 | `2c_balanced_mid` | `trunc_even_0.9` | 13.0668 | 9.4159 | 11.5083 | 0.881x |
| 2 | `2c_asym_low_high` | `trunc_even_0.9` | 12.3563 | 9.3453 | 11.4238 | 0.925x |
| 2 | `2c_idle_saturated` | `trunc_even_0.85` | 12.2853 | 11.2512 | 13.1867 | 1.073x |
| 2 | `2c_high` | `trunc_even_0.9` | 12.8478 | 11.1821 | 14.4283 | 1.123x |
| 4 | `4c_gradient` | `trunc_even_0.9` | 15.9322 | 17.8141 | 23.2100 | 1.457x |
| 4 | `4c_balanced_high` | `trunc_even_0.9` | 20.9768 | 18.2116 | 25.0361 | 1.194x |
| 4 | `4c_front_hot` | `trunc_even_0.9` | 19.9424 | 16.8517 | 21.4096 | 1.074x |
| 4 | `4c_high` | `trunc_even_0.9` | 17.9023 | 17.4627 | 21.5618 | 1.204x |
| 6 | `6c_ramp_light` | `trunc_even_0.85` | 25.6931 | 23.1196 | 24.8774 | 0.968x |
| 6 | `6c_mixed` | `trunc_even_0.85` | 18.4709 | 19.7210 | 2.8125 | 0.152x |
| 6 | `6c_front_hot` | `trunc_even_0.9` | 20.7820 | 23.8251 | 25.4950 | 1.227x |
| 6 | `6c_high` | `baseline_no_svd` | 26.5934 | 21.3740 | 21.3740 | 0.804x |
| 8 | `8c_ramp_light` | `trunc_even_0.85` | 26.2936 | 0.6098 | 31.7797 | 1.209x |
| 8 | `8c_mixed` | `trunc_even_0.85` | 20.4348 | 20.7318 | 25.4590 | 1.246x |
| 8 | `8c_front_hot` | `trunc_even_0.85` | 19.4523 | 20.1616 | 27.1675 | 1.397x |
| 8 | `8c_high` | `trunc_even_0.8` | 20.6566 | 24.9655 | 29.1845 | 1.413x |

## 按核心数汇总

| 核心数 | 场景数 | 最小提升 | 中位数提升 | 几何平均提升 | 平均提升 | 最大提升 | 低于原始 baseline 场景数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0.881x | 0.999x | 0.995x | 1.000x | 1.123x | 2 |
| 4 | 4 | 1.074x | 1.199x | 1.224x | 1.232x | 1.457x | 0 |
| 6 | 4 | 0.152x | 0.886x | 0.617x | 0.788x | 1.227x | 3 |
| 8 | 4 | 1.209x | 1.321x | 1.313x | 1.316x | 1.413x | 0 |

## 观察

1. 4 核和 8 核场景最适合画正向加速图。
   - 4 核全部高于原始 Q4 baseline，提升范围为 `1.074x` 到 `1.457x`。
   - 8 核全部高于原始 Q4 baseline，提升范围为 `1.209x` 到 `1.413x`。

2. 2 核场景基本接近持平。
   - 两个低于 baseline，两个高于 baseline。
   - 这说明在核心数较少时，SVD 截断节省的计算量可能不足以覆盖 SVD-capable GGUF/runtime 的额外开销。

3. 6 核场景暴露出策略边界。
   - `6c_front_hot` 有正向提升，`1.227x`。
   - `6c_mixed` 出现明显负收益，scheduler validation 中位数只有 `2.8125 tok/s`。
   - `6c_high` calibration 后选择了 `baseline_no_svd`，但 SVD GGUF 的 no-SVD runtime 仍低于原始 Q4 baseline。
   - 这说明如果论文中要画整体性能图，6 核这组需要被作为“当前策略仍需保护机制”的反例，而不是简单删除。

4. `8c_ramp_light` 中 SVD Q4 no-SVD baseline 出现异常低速。
   - 该场景 SVD no-SVD baseline 为 `0.6098 tok/s`，但原始 Q4 baseline 为 `26.2936 tok/s`，scheduler 为 `31.7797 tok/s`。
   - 因此相对 SVD no-SVD 的 `52.117x` 不应使用；相对原始 baseline 的 `1.209x` 更可信。

## 建议画图方式

建议画两张图：

1. 主图：按核心数分组的柱状图。
   - x 轴：场景。
   - y 轴：`Scheduler / Original Q4 baseline`。
   - 用颜色区分 2、4、6、8 核。
   - 在 `y=1.0` 画一条水平线。

2. 辅图：按核心数汇总的箱线图或点图。
   - x 轴：核心数。
   - y 轴：`Scheduler / Original Q4 baseline`。
   - 可以显示 4 核和 8 核稳定收益，2 核基本持平，6 核存在策略失效点。

如果只想展示当前策略的正向能力，建议主图优先展示 4 核和 8 核；如果想写得更科学，应保留 2 核和 6 核，并说明当前 scheduler 还需要加入更强的 fallback/guard policy。

## 数据文件

场景定义：

```text
src/llama.cpp/exp12_algorithms/scenarios_q4_plot_2_4_6_8.json
```

Scheduler 测量：

```text
src/llama.cpp/exp12_algorithms/results/q4_plot_scheduler_2_4_6_8_20260502_r1/decisions.csv
src/llama.cpp/exp12_algorithms/results/q4_plot_scheduler_2_4_6_8_20260502_r1/raw.csv
```

原始 Q4 baseline 测量：

```text
src/llama.cpp/exp12_algorithms/results/q4_plot_original_baseline_2_4_6_8_20260502_r1/summary.csv
src/llama.cpp/exp12_algorithms/results/q4_plot_original_baseline_2_4_6_8_20260502_r1/raw.csv
```

合并后的画图表：

```text
src/llama.cpp/exp12_algorithms/results/q4_plot_2_4_6_8_20260502_summary/q4_plot_combined.csv
src/llama.cpp/exp12_algorithms/results/q4_plot_2_4_6_8_20260502_summary/q4_plot_summary_by_cores.csv
```
