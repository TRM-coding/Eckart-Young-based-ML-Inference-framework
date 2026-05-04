# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `3`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `6c_extreme_2idle` | `90,90,90,100,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 1.3660 | 13.2369 | 9.690x | 15.1424 |
| `7c_extreme_1idle` | `80,90,90,100,100,100,100,0` | `baseline_no_svd` | `` | 28.6195 | 28.6195 | 1.000x | 15.1424 |
| `8c_all_100` | `100,100,100,100,100,100,100,100` | `baseline_no_svd` | `` | 28.6006 | 28.6006 | 1.000x | 15.1424 |
| `8c_extreme_all_hot` | `80,90,90,90,100,100,100,100` | `baseline_no_svd` | `` | 28.2171 | 28.2171 | 1.000x | 15.1424 |
