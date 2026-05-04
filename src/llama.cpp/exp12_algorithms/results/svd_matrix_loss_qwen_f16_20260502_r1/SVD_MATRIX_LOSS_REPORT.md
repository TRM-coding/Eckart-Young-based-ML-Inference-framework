# SVD Matrix Loss Profile

- model: `src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf`
- rates: `0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9`
- layer reduction: `sum`
- loss definition: spectral norm of the discarded residual matrix.
- note: SVD factors store `U * sqrt(S)` and `sqrt(S) * Vh`; singular values are recovered from factor norms.

## Outputs

- `svd_matrix_loss.csv`: per layer, per FFN matrix, per rate.
- `svd_layer_loss.csv`: layer-level reduction over up/gate/down.
- `svd_loss_for_scheduler.csv`: same layer-level table for profile builders.
- `svd_singular_summary.csv`: singular-value summary per matrix.
