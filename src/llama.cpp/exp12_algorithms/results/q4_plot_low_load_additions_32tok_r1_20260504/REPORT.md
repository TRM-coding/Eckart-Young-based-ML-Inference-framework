# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_low_01` | `10,20,0,0,0,0,0,0` | `baseline_no_svd` | `` | 29.5560 | 29.5560 | 1.000x | 15.1424 |
| `2c_low_02` | `20,30,0,0,0,0,0,0` | `baseline_no_svd` | `` | 30.4393 | 30.4393 | 1.000x | 15.1424 |
| `4c_low_01` | `10,10,20,20,0,0,0,0` | `baseline_no_svd` | `` | 29.6965 | 29.6965 | 1.000x | 15.1424 |
| `4c_low_02` | `10,20,30,40,0,0,0,0` | `baseline_no_svd` | `` | 30.3009 | 30.3009 | 1.000x | 15.1424 |
| `6c_low_01` | `10,10,20,20,30,30,0,0` | `major_only_no_svd` | `p=1;major=66` | 11.7154 | 19.1346 | 1.633x | 15.1424 |
| `6c_low_02` | `10,20,20,30,30,40,0,0` | `baseline_no_svd` | `` | 18.1481 | 18.1481 | 1.000x | 15.1424 |
| `8c_low_01` | `10,10,20,20,30,30,40,40` | `baseline_no_svd` | `` | 23.3079 | 23.3079 | 1.000x | 15.1424 |
| `8c_low_02` | `10,20,20,30,30,40,40,50` | `baseline_no_svd` | `` | 17.1809 | 17.1809 | 1.000x | 15.1424 |
