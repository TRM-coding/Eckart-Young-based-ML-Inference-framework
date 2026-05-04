#!/usr/bin/env python3
"""Measure per-layer SVD truncation loss from GGUF SVD factors.

The loss is the spectral norm of the discarded residual matrix.  For an exact
SVD W = U diag(S) Vh, truncating the tail rank slice has residual 2-norm equal
to the largest discarded singular value.  The generated SVD GGUF stores
U*sqrt(S) and sqrt(S)*Vh, so we recover S from the column/row norms of the two
factors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize


EXP_DIR = Path(__file__).resolve().parent
SVD_NAME_RE = re.compile(r"blk\.(\d+)\.ffn_(up|gate|down)_svd_([uv])\.weight$")


def parse_rates(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def tensor_to_float32(tensor: Any) -> np.ndarray:
    raw = np.array(tensor.data, copy=False)
    shape = tuple(int(x) for x in tensor.shape[::-1])
    if tensor.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        array = raw.astype(np.float32, copy=False).reshape(shape)
    else:
        array = dequantize(raw, tensor.tensor_type).reshape(shape)
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    return np.ascontiguousarray(array, dtype=np.float32)


def collect_svd_pairs(reader: GGUFReader) -> dict[tuple[int, str], dict[str, Any]]:
    pairs: dict[tuple[int, str], dict[str, Any]] = {}
    for tensor in reader.tensors:
        match = SVD_NAME_RE.match(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        matrix = match.group(2)
        side = match.group(3)
        pairs.setdefault((layer, matrix), {})[side] = tensor
    return pairs


def singular_values_from_factors(u_factor: np.ndarray, v_factor: np.ndarray) -> np.ndarray:
    if u_factor.shape[1] != v_factor.shape[0]:
        raise ValueError(f"SVD factor rank mismatch: U {u_factor.shape}, V {v_factor.shape}")
    # U stores U*sqrt(S), V stores sqrt(S)*Vh.  The singular value can be
    # recovered from either squared norm; average the two estimates to reduce
    # F16 / quantization noise.
    s_from_u = np.sum(u_factor.astype(np.float64) ** 2, axis=0)
    s_from_v = np.sum(v_factor.astype(np.float64) ** 2, axis=1)
    s = 0.5 * (s_from_u + s_from_v)
    s = np.maximum(s, 0.0)
    # The generation script stores SVD components sorted descending.  Re-sort to
    # guard against small quantization inversions.
    return np.sort(s)[::-1].astype(np.float64)


def keep_rank(total_rank: int, rate: float) -> int:
    if rate <= 0.0:
        return total_rank
    if rate >= 0.999:
        return 0
    k_trunc = int(math.ceil(rate * float(total_rank)))
    k_keep = total_rank - k_trunc
    return max(0, min(total_rank, k_keep))


def residual_spectral_norm(singular_values: np.ndarray, rate: float) -> tuple[int, int, float, float]:
    total_rank = int(singular_values.shape[0])
    keep = keep_rank(total_rank, rate)
    trunc = total_rank - keep
    if trunc <= 0 or keep >= total_rank:
        residual = 0.0
    elif keep <= 0:
        residual = float(singular_values[0])
    else:
        residual = float(singular_values[keep])
    full = float(singular_values[0]) if total_rank else 0.0
    relative = residual / full if full > 0.0 else 0.0
    return keep, trunc, residual, relative


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure SVD residual spectral-norm loss.")
    parser.add_argument("--model", type=Path, default=Path("src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf"))
    parser.add_argument("--rates", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9")
    parser.add_argument("--layer-reduction", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/svd_matrix_loss_20260502_r1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rates = parse_rates(args.rates)
    reader = GGUFReader(str(args.model))
    pairs = collect_svd_pairs(reader)

    matrix_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    singular_rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], np.ndarray] = {}

    for (layer, matrix), pair in sorted(pairs.items()):
        if "u" not in pair or "v" not in pair:
            raise ValueError(f"missing SVD factor for layer={layer} matrix={matrix}: {pair.keys()}")
        u = tensor_to_float32(pair["u"])
        v = tensor_to_float32(pair["v"])
        s = singular_values_from_factors(u, v)
        cache[(layer, matrix)] = s
        singular_rows.append(
            {
                "layer": layer,
                "matrix": matrix,
                "rank": len(s),
                "sigma_max": float(s[0]) if len(s) else 0.0,
                "sigma_min": float(s[-1]) if len(s) else 0.0,
                "sigma_mean": float(np.mean(s)) if len(s) else 0.0,
            }
        )
        for rate in rates:
            keep, trunc, loss, rel = residual_spectral_norm(s, rate)
            matrix_rows.append(
                {
                    "scope": "matrix",
                    "layer": layer,
                    "matrix": matrix,
                    "rate": rate,
                    "rank_total": len(s),
                    "rank_keep": keep,
                    "rank_truncated": trunc,
                    "loss": loss,
                    "relative_loss": rel,
                    "loss_definition": "spectral_norm_discarded_residual",
                }
            )

    layers = sorted({layer for layer, _ in cache})
    for layer in layers:
        for rate in rates:
            vals = [
                residual_spectral_norm(cache[(layer, matrix)], rate)[2]
                for matrix in ("up", "gate", "down")
                if (layer, matrix) in cache
            ]
            rels = [
                residual_spectral_norm(cache[(layer, matrix)], rate)[3]
                for matrix in ("up", "gate", "down")
                if (layer, matrix) in cache
            ]
            if args.layer_reduction == "sum":
                loss = float(sum(vals))
                rel = float(sum(rels))
            elif args.layer_reduction == "mean":
                loss = float(sum(vals) / len(vals))
                rel = float(sum(rels) / len(rels))
            else:
                loss = float(max(vals))
                rel = float(max(rels))
            layer_rows.append(
                {
                    "scope": "layer",
                    "layer": layer,
                    "matrix": "ffn_up+ffn_gate+ffn_down",
                    "rate": rate,
                    "loss": loss,
                    "relative_loss": rel,
                    "reduction": args.layer_reduction,
                    "loss_definition": "spectral_norm_discarded_residual",
                }
            )

    write_csv(
        args.out_dir / "svd_matrix_loss.csv",
        matrix_rows,
        [
            "scope",
            "layer",
            "matrix",
            "rate",
            "rank_total",
            "rank_keep",
            "rank_truncated",
            "loss",
            "relative_loss",
            "loss_definition",
        ],
    )
    write_csv(
        args.out_dir / "svd_layer_loss.csv",
        layer_rows,
        ["scope", "layer", "matrix", "rate", "loss", "relative_loss", "reduction", "loss_definition"],
    )
    write_csv(
        args.out_dir / "svd_singular_summary.csv",
        singular_rows,
        ["layer", "matrix", "rank", "sigma_max", "sigma_min", "sigma_mean"],
    )
    write_csv(
        args.out_dir / "svd_loss_for_scheduler.csv",
        layer_rows,
        ["scope", "layer", "matrix", "rate", "loss", "relative_loss", "reduction", "loss_definition"],
    )

    report = [
        "# SVD Matrix Loss Profile",
        "",
        f"- model: `{args.model}`",
        f"- rates: `{','.join(str(rate) for rate in rates)}`",
        f"- layer reduction: `{args.layer_reduction}`",
        "- loss definition: spectral norm of the discarded residual matrix.",
        "- note: SVD factors store `U * sqrt(S)` and `sqrt(S) * Vh`; singular values are recovered from factor norms.",
        "",
        "## Outputs",
        "",
        "- `svd_matrix_loss.csv`: per layer, per FFN matrix, per rate.",
        "- `svd_layer_loss.csv`: layer-level reduction over up/gate/down.",
        "- `svd_loss_for_scheduler.csv`: same layer-level table for profile builders.",
        "- `svd_singular_summary.csv`: singular-value summary per matrix.",
    ]
    (args.out_dir / "SVD_MATRIX_LOSS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "matrices": len(cache), "matrix_rows": len(matrix_rows), "layer_rows": len(layer_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
