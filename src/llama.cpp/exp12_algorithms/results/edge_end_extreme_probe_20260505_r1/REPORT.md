# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `8c_probe_all_100` | `100,100,100,100,100,100,100,100` | `baseline_no_svd` | `` | 29.4343 | 29.4343 | 1.000x | 15.1424 |
| `8c_probe_ramp_hi` | `80,90,90,100,100,100,100,100` | `baseline_no_svd` | `` | 30.7242 | 30.7242 | 1.000x | 15.1424 |
