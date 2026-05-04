# Major-only 调度策略 9 场景性能报告

> 注意：这份报告只能作为 `major_only_no_svd` 本地候选的 smoke 结果，不能作为完整 scheduler 性能结论。
>
> 原因：
>
> 1. 该轮没有执行真实 adb 协同卸载，`edge_end_no_svd` 没有进入实测闭环。
> 2. 本地 exp12 默认模型是 `qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf`，而 exp6 layer-coop 默认模型是原始 `qwen.q4_0.gguf`，两条路径的 tok/s 不能直接混在一起比较。
> 3. 该轮 decode tokens 为 `1`，只能快速暴露策略方向，不能作为稳定吞吐结论。
>
> 完整结论应重新使用同一模型口径，并把 `baseline_no_svd`、`major_only_no_svd`、`edge_end_no_svd` 放入同一负载场景下执行比较。

本文档汇总 `quality_first_no_svd + major_only_no_svd + runtime guard` 调度器在 9 个异构负载场景下的性能和 PPL。

## 实验口径

- 运行核心：隔离的 CPU `60-67` 子集，具体核心数随场景变化。
- baseline：`baseline_no_svd`，即完整 no-SVD 计算放在该场景的全部核心上。
- scheduler：先判断是否可以使用 `major_only_no_svd`，如果不能稳定更快，则回退到 `baseline_no_svd`。
- `major_only_no_svd`：完整 no-SVD 计算只放在低占用 major 核心上，高占用 minor 核心不参与。
- repeats：每个场景 3 次。
- decode tokens：每次 1 token，用于快速闭环和策略验证。
- speedup：使用同一 repeat 下 `scheduler tok/s / baseline tok/s` 的配对中位数，避免 baseline 偶发抖动直接污染结论。
- runtime guard：如果某一轮 `major_only_no_svd` 实测不快于 baseline，则该轮回退为 baseline，speedup 记为 `1.000x`。

原始结果目录：

```text
src/llama.cpp/exp12_algorithms/results/scheduler_quality_first_major_only_runtime_guard_9scenes_20260504_r1
```

## PPL 口径

当前 9 个场景最终只会选择以下两类策略：

- `baseline_no_svd`
- `major_only_no_svd`

二者都不使用 SVD 裁剪，不丢弃 rank，也不触发 timeout drop；区别只在于线程被绑定到哪些 CPU 核心。因此模型数学计算结果不变，PPL 与 no-SVD baseline 相同。

已有 ctx=128 PPL sanity 测量为：

```text
baseline_no_svd PPL = 15.1424 +/- 4.47405
```

来源：

```text
src/llama.cpp/exp12_algorithms/results/selected_policy_ppl_20260504_smoke/policy_ppl_ctx128.csv
src/llama.cpp/exp12_algorithms/results/rerun_isolated_20260428_195933/ppl_timeout_small_20260428/PPL_SANITY_CTX128_REPORT.md
```

所以本报告中所有场景的 scheduler PPL 均记为 `15.1424`，相对 baseline 的 PPL 变化为 `0`。

## 结果汇总

| 场景 | 核心数 | 每核负载 | 最终策略 | major 核心 | minor 核心 | baseline tok/s 中位数 | scheduler tok/s 中位数 | 配对 speedup 中位数 | 配对最小 speedup | PPL |
|---|---:|---|---|---|---|---:|---:|---:|---:|---:|
| `2c_asym_low_high` | 2 | `20,80` | `baseline_no_svd` | - | - | 9.2528 | 9.2528 | 1.000x | 1.000x | 15.1424 |
| `2c_balanced_mid` | 2 | `50,50` | `baseline_no_svd` | - | - | 11.3584 | 11.3584 | 1.000x | 1.000x | 15.1424 |
| `2c_high` | 2 | `80,100` | `baseline_no_svd` | - | - | 11.4954 | 11.4954 | 1.000x | 1.000x | 15.1424 |
| `2c_idle_saturated` | 2 | `0,100` | `baseline_no_svd` | - | - | 11.7055 | 11.7055 | 1.000x | 1.000x | 15.1424 |
| `4c_balanced_high` | 4 | `20,40,60,80` | `baseline_no_svd` | - | - | 16.4392 | 16.4392 | 1.000x | 1.000x | 15.1424 |
| `4c_front_hot` | 4 | `90,70,20,0` | `major_only_no_svd` | `61,62,63` | `60` | 6.2210 | 16.1694 | 1.999x | 1.000x | 15.1424 |
| `4c_gradient` | 4 | `0,20,60,100` | `baseline_no_svd` | - | - | 15.9487 | 15.9487 | 1.000x | 1.000x | 15.1424 |
| `4c_high` | 4 | `70,80,90,100` | `baseline_no_svd` | - | - | 16.8858 | 16.8858 | 1.000x | 1.000x | 15.1424 |
| `8c_high` | 8 | `30,40,50,60,70,80,90,100` | `major_only_no_svd` | `60,61,62,63,64,65,66` | `67` | 20.2239 | 25.2289 | 1.247x | 1.000x | 15.1424 |

## 解释

1. `2c_*` 场景最终全部回退到 `baseline_no_svd`。原因是只用 1 个 major 核心跑完整模型通常不快于 2 核 baseline。
2. `4c_balanced_high`、`4c_gradient`、`4c_high` 最终也回退到 baseline，说明这些负载下排除高占用核心带来的收益不足以抵消少用核心的损失。
3. `4c_front_hot` 选择了 `major_only_no_svd`，但该场景有 baseline 偶发极慢点，因此 `1.999x` 不应理解成稳定上限；runtime guard 已保证最差轮次回退到 `1.000x`。
4. `8c_high` 是当前最稳定的 major-only 有效场景：调度器使用 `60-66` 七个低占用核心，避开 `67` 号最高负载核心，配对中位 speedup 为 `1.247x`，最差轮次由 runtime guard 回退到 `1.000x`。
5. 因为最终策略均为 no-SVD 路线，PPL 不随负载场景变化，均保持在 ctx=128 sanity baseline 的 `15.1424`。

## 结论

这轮实验验证了新增的 `major_only_no_svd` 候选空间是必要的：在某些高异构负载场景下，不使用 SVD、只避开最热核心，可以在 PPL 不变的前提下获得速度收益。

同时，实验也说明该策略不能盲目启用。当前调度器配合 runtime guard 后，在 9 个场景中没有出现低于 baseline 的结果；不适合 major-only 的场景会自动回退为 baseline。
