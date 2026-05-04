# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `3`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_asym_low_high` | `20,80,0,0,0,0,0,0` | `baseline_no_svd` | `` | 32.1983 | 32.1983 | 1.000x | 15.1424 |
| `2c_balanced_mid` | `50,50,0,0,0,0,0,0` | `baseline_no_svd` | `` | 32.0594 | 32.0594 | 1.000x | 15.1424 |
| `2c_high` | `80,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=4;major=62,63,64,65` | 13.1621 | 28.9963 | 2.203x | 15.1424 |
| `2c_idle_saturated` | `0,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=4;major=60,62,63,64` | 30.2208 | 32.3851 | 1.072x | 15.1424 |
| `4c_balanced_high` | `20,40,60,80,0,0,0,0` | `major_only_no_svd` | `p=4;major=64,65,66,67` | 8.4959 | 23.5896 | 2.777x | 15.1424 |
| `4c_front_hot` | `90,70,20,0,0,0,0,0` | `baseline_no_svd` | `` | 32.1275 | 32.1275 | 1.000x | 15.1424 |
| `4c_gradient` | `0,20,60,100,0,0,0,0` | `baseline_no_svd` | `` | 30.4100 | 30.4100 | 1.000x | 15.1424 |
| `4c_high` | `70,80,90,100,0,0,0,0` | `major_only_no_svd` | `p=4;major=64,65,66,67` | 2.2583 | 22.6875 | 10.046x | 15.1424 |
| `8c_high` | `30,40,50,60,70,80,90,100` | `baseline_no_svd` | `` | 26.5266 | 26.5266 | 1.000x | 15.1424 |
