# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `20,80` | `baseline_no_svd` | `` | 12.5699 | 12.5699 | 1.000x | 15.1424 |
