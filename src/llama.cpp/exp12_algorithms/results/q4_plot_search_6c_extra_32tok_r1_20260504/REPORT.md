# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `6c_plot_extra_01` | `10,30,70,90,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 10.3500 | 17.2867 | 1.670x | 15.1424 |
| `6c_plot_extra_02` | `20,40,70,90,100,100,0,0` | `baseline_no_svd` | `` | 35.8380 | 35.8380 | 1.000x | 15.1424 |
| `6c_plot_extra_03` | `30,50,80,90,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 9.5357 | 21.1675 | 2.220x | 15.1424 |
| `6c_plot_extra_04` | `10,60,80,90,100,100,0,0` | `baseline_no_svd` | `` | 36.5887 | 36.5887 | 1.000x | 15.1424 |
| `6c_plot_extra_05` | `40,60,80,90,100,100,0,0` | `baseline_no_svd` | `` | 38.3694 | 38.3694 | 1.000x | 15.1424 |
| `6c_plot_extra_06` | `20,70,80,90,100,100,0,0` | `baseline_no_svd` | `` | 36.2656 | 36.2656 | 1.000x | 15.1424 |
| `6c_plot_extra_07` | `55,65,85,95,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 3.6881 | 16.7317 | 4.537x | 15.1424 |
| `6c_plot_extra_08` | `15,45,75,95,100,100,0,0` | `baseline_no_svd` | `` | 19.3843 | 19.3843 | 1.000x | 15.1424 |
