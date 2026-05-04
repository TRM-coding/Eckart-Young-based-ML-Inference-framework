# Layer-coop 协同工具状态

## 当前状态

准备单独运行 exp6 的 `layer_coop_client` / `layer_mobile_server` 工具，以确认当前协同计算速度是否下降。

但在检查阶段，adb 真机状态变为：

```text
10.20.0.3:5555 offline
emulator-5554 device
```

并且：

```text
adb -P 5038 -s 10.20.0.3:5555 shell 'echo phone_ok'
```

返回：

```text
adb: device offline
```

因此当前无法给出有效的协同测速结果。不能使用 emulator 代替，因为 emulator 上没有 `/data/local/tmp/CE_Ada/qwen.q4_0.gguf` 和 `layer_mobile_server`，且性能不代表真机。

## 真机恢复后建议命令

真机 online 后，可用以下命令快速验证协同速度：

```bash
python src/llama.cpp/exp6_decode_svd_model/benchmark_layer_coop_split_load.py \
  --pc-cpus 60-67 \
  --loads 0 \
  --splits 2,4,8,12 \
  --tokens 32 \
  --threads 8 \
  --repeats 1 \
  --timeout-s 180 \
  --out-dir src/llama.cpp/exp6_decode_svd_model/results/layer_coop_tool_check_20260504_r1
```

如果要测高负载协同：

```bash
python src/llama.cpp/exp6_decode_svd_model/benchmark_layer_coop_split_load.py \
  --pc-cpus 60-67 \
  --loads 80 \
  --splits 2,4,8,12 \
  --tokens 32 \
  --threads 8 \
  --repeats 1 \
  --timeout-s 180 \
  --out-dir src/llama.cpp/exp6_decode_svd_model/results/layer_coop_tool_check_load80_20260504_r1
```

## 结论

当前第二个任务被 adb 真机 offline 阻塞，尚未完成有效测速。需要真机恢复为 `device` 且 `adb shell echo phone_ok` 成功后再跑。
