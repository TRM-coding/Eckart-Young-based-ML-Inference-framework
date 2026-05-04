# Q4 Lossless Scheduler Benchmark

- Q4 model: `/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.q4_0.gguf`
- tokens: `32`
- repeats: `1`
- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.

| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |
|---|---|---|---|---:|---:|---:|---:|
| `2c_plot_search_01` | `50,80,0,0,0,0,0,0` | `baseline_no_svd` | `` | 37.5178 | 37.5178 | 1.000x | 15.1424 |
| `2c_plot_search_02` | `60,90,0,0,0,0,0,0` | `baseline_no_svd` | `` | 35.9108 | 35.9108 | 1.000x | 15.1424 |
| `2c_plot_search_03` | `70,90,0,0,0,0,0,0` | `baseline_no_svd` | `` | 39.4878 | 39.4878 | 1.000x | 15.1424 |
| `2c_plot_search_04` | `80,100,0,0,0,0,0,0` | `baseline_no_svd` | `` | 41.1240 | 41.1240 | 1.000x | 15.1424 |
| `2c_plot_search_05` | `40,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=1;major=62` | 4.5714 | 26.4200 | 5.779x | 15.1424 |
| `2c_plot_search_06` | `55,95,0,0,0,0,0,0` | `major_only_no_svd` | `p=1;major=62` | 19.1913 | 24.6528 | 1.285x | 15.1424 |
| `2c_plot_search_07` | `65,85,0,0,0,0,0,0` | `baseline_no_svd` | `` | 41.4417 | 41.4417 | 1.000x | 15.1424 |
| `2c_plot_search_08` | `75,100,0,0,0,0,0,0` | `major_only_no_svd` | `p=1;major=62` | 2.9975 | 28.5988 | 9.541x | 15.1424 |
| `4c_plot_search_01` | `30,50,70,90,0,0,0,0` | `baseline_no_svd` | `` | 40.2307 | 40.2307 | 1.000x | 15.1424 |
| `4c_plot_search_02` | `40,60,80,100,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 22.1425 | 26.5455 | 1.199x | 15.1424 |
| `4c_plot_search_03` | `50,50,80,90,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 21.0872 | 27.2058 | 1.290x | 15.1424 |
| `4c_plot_search_04` | `60,70,80,90,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 3.5987 | 28.9737 | 8.051x | 15.1424 |
| `4c_plot_search_05` | `20,60,90,100,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 6.6143 | 26.4220 | 3.995x | 15.1424 |
| `4c_plot_search_06` | `70,80,90,100,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 12.7079 | 26.1769 | 2.060x | 15.1424 |
| `4c_plot_search_07` | `45,65,85,95,0,0,0,0` | `baseline_no_svd` | `` | 43.9689 | 43.9689 | 1.000x | 15.1424 |
| `4c_plot_search_08` | `30,70,80,100,0,0,0,0` | `major_only_no_svd` | `p=1;major=64` | 0.8108 | 22.0322 | 27.172x | 15.1424 |
| `6c_plot_search_01` | `30,40,60,70,90,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 0.6074 | 30.2334 | 49.773x | 15.1424 |
| `6c_plot_search_02` | `40,50,70,80,90,100,0,0` | `baseline_no_svd` | `` | 45.6166 | 45.6166 | 1.000x | 15.1424 |
| `6c_plot_search_03` | `50,60,70,80,90,100,0,0` | `baseline_no_svd` | `` | 45.0036 | 45.0036 | 1.000x | 15.1424 |
| `6c_plot_search_04` | `60,60,80,80,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 18.1698 | 22.0302 | 1.212x | 15.1424 |
| `6c_plot_search_05` | `20,50,80,90,100,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 4.4156 | 20.3341 | 4.605x | 15.1424 |
| `6c_plot_search_06` | `70,80,80,90,90,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 5.2315 | 22.2706 | 4.257x | 15.1424 |
| `6c_plot_search_07` | `35,55,75,85,95,100,0,0` | `baseline_no_svd` | `` | 41.9584 | 41.9584 | 1.000x | 15.1424 |
| `6c_plot_search_08` | `45,65,75,85,95,100,0,0` | `major_only_no_svd` | `p=1;major=66` | 6.4954 | 21.2515 | 3.272x | 15.1424 |
| `8c_plot_search_01` | `20,40,60,80,90,90,100,100` | `baseline_no_svd` | `` | 38.9184 | 38.9184 | 1.000x | 15.1424 |
| `8c_plot_search_02` | `30,50,70,80,90,90,100,100` | `baseline_no_svd` | `` | 34.8700 | 34.8700 | 1.000x | 15.1424 |
| `8c_plot_search_03` | `40,60,70,80,90,100,100,100` | `baseline_no_svd` | `` | 38.0762 | 38.0762 | 1.000x | 15.1424 |
| `8c_plot_search_04` | `50,60,70,80,90,90,100,100` | `baseline_no_svd` | `` | 37.4386 | 37.4386 | 1.000x | 15.1424 |
| `8c_plot_search_05` | `60,70,80,80,90,90,100,100` | `baseline_no_svd` | `` | 39.0692 | 39.0692 | 1.000x | 15.1424 |
| `8c_plot_search_06` | `70,70,80,90,90,100,100,100` | `baseline_no_svd` | `` | 41.0695 | 41.0695 | 1.000x | 15.1424 |
| `8c_plot_search_07` | `30,60,90,90,90,100,100,100` | `baseline_no_svd` | `` | 43.2248 | 43.2248 | 1.000x | 15.1424 |
| `8c_plot_search_08` | `40,40,80,80,100,100,100,100` | `baseline_no_svd` | `` | 39.7571 | 39.7571 | 1.000x | 15.1424 |
