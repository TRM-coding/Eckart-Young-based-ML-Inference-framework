# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `8c_all_100` | `100,100,100,100,100,100,100,100` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 25.0060 | 31.3177 | 1.252x | 15.1424 |
| `8c_all_80` | `80,80,80,80,80,80,80,80` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 24.3876 | 59.4275 | 2.437x | 15.1424 |
| `8c_all_90` | `90,90,90,90,90,90,90,90` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 24.7791 | 57.6429 | 2.326x | 15.1424 |
| `8c_alt_hi` | `80,100,80,100,90,100,90,100` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 24.9247 | 61.3778 | 2.463x | 15.1424 |
| `8c_back_hot` | `80,80,90,90,100,100,100,100` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 25.9027 | 60.7169 | 2.344x | 15.1424 |
| `8c_front_hot` | `100,100,100,100,90,90,80,80` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 21.9225 | 58.0954 | 2.650x | 15.1424 |
| `8c_mixed_hi` | `80,90,100,100,80,90,100,100` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 25.5380 | 59.4246 | 2.327x | 15.1424 |
| `8c_ramp_hi` | `80,90,90,100,100,100,100,100` | `edge_end_no_svd` | `M=0;Phone=[0,28);PC=empty` | 25.8411 | 46.6954 | 1.807x | 15.1424 |
