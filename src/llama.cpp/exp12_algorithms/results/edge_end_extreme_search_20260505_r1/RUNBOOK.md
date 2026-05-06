# 8 核极端负载 edge_end 搜索说明

目标：筛选满足以下条件的绘图数据：

1. `scheduler` 选择 `edge_end_no_svd`。
2. 8 个 PC 核心都有较高背景负载。
3. `edge_end_no_svd` 的速度同时高于 `baseline_no_svd` 和 `major_only_no_svd`。

## 当前设备状态

截至本轮检查，`adb -P 5038 devices -l` 只枚举到：

```text
emulator-5554 product:sdk_gphone64_x86_64 model:sdk_gphone64_x86_64
```

并且设备属性为：

```text
ro.product.cpu.abi = x86_64
ro.boot.qemu      = 1
```

因此当前 5038 上不是 arm64 真机。现有 `build-android/layer_mobile_server` 和 `build-android-official-cpu/layer_mobile_server` 都是 arm64-v8a，不能在该 x86_64 emulator 上作为手机端执行。为了避免污染论文数据，本轮没有把 emulator 当成端侧卸载设备使用。

## 真机恢复后直接运行

先确认设备列表里出现 arm64 真机：

```bash
adb -P 5038 devices -l
adb -P 5038 shell 'getprop ro.product.model; getprop ro.product.cpu.abi; getprop ro.boot.qemu'
```

如果 `/data/local/tmp/CE_Ada` 缺模型或 server，需要先推送：

```bash
adb -P 5038 shell 'mkdir -p /data/local/tmp/CE_Ada'
adb -P 5038 push build-android/layer_mobile_server /data/local/tmp/CE_Ada/
adb -P 5038 push build-android/layer_tail_local_bench /data/local/tmp/CE_Ada/
adb -P 5038 push src/llama.cpp/gguf_models/qwen.q4_0.gguf /data/local/tmp/CE_Ada/
adb -P 5038 shell 'chmod +x /data/local/tmp/CE_Ada/layer_mobile_server /data/local/tmp/CE_Ada/layer_tail_local_bench'
```

然后运行 8 核极端搜索：

```bash
python3 src/llama.cpp/exp12_algorithms/benchmark_q4_lossless_scheduler.py \
  --out-dir src/llama.cpp/exp12_algorithms/results/edge_end_extreme_search_20260505_r1 \
  --scenarios-json src/llama.cpp/exp12_algorithms/results/edge_end_extreme_search_20260505_r1/scenarios.json \
  --tokens 32 \
  --repeats 2 \
  --max-major-candidates 0 \
  --splits 2,4,6,8 \
  --pc-host 10.126.59.25 \
  --timeout-s 300
```

最后导出绘图 CSV：

```bash
python3 src/llama.cpp/exp12_algorithms/export_edge_end_winners.py \
  --raw src/llama.cpp/exp12_algorithms/results/edge_end_extreme_search_20260505_r1/raw.csv \
  --out src/llama.cpp/exp12_algorithms/results/edge_end_extreme_search_20260505_r1/plot_edge_end_winners.csv \
  --min-occupied 8 \
  --min-load 70
```

## 已生成文件

- `scenarios.json`：12 组 8 核全占用极端负载候选。
- `plot_edge_end_winners_from_history_target_filter.csv`：用历史 raw 按本次严格条件筛选，当前为 0 行，因为历史真实 edge_end 数据没有 8 核全占用胜出点。
- `plot_edge_end_winners_history_relaxed.csv`：用历史 raw 放宽负载条件后的 sanity CSV，可证明筛选器能正确抽出 edge_end 胜过 baseline 和 major_only 的真实历史点。
