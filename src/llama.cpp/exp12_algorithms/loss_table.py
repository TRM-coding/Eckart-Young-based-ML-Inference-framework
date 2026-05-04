"""Utilities for measured SVD truncation loss tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def parse_loss_key(layer: int, rate: float) -> tuple[int, str]:
    return layer, f"{rate:.6g}"


def load_measured_loss_table(path: Path | None, reduction: str = "sum") -> dict[tuple[int, str], float]:
    if path is None:
        return {}
    rows: list[dict[str, str]]
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    by_layer_rate: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row.get("scope") == "layer":
            key = parse_loss_key(int(row["layer"]), float(row["rate"]))
            by_layer_rate[key] = [float(row["loss"])]
            continue
        if row.get("scope") not in ("matrix", None, ""):
            continue
        if not row.get("loss"):
            continue
        key = parse_loss_key(int(row["layer"]), float(row["rate"]))
        by_layer_rate.setdefault(key, []).append(float(row["loss"]))

    out: dict[tuple[int, str], float] = {}
    for key, values in by_layer_rate.items():
        clean = [value for value in values if math.isfinite(value)]
        if not clean:
            continue
        if reduction == "sum":
            out[key] = float(sum(clean))
        elif reduction == "mean":
            out[key] = float(sum(clean) / len(clean))
        elif reduction == "max":
            out[key] = float(max(clean))
        else:
            raise ValueError(f"unknown measured loss reduction: {reduction}")
    return out


def measured_or_heuristic_loss(
    table: dict[tuple[int, str], float],
    layer: int,
    rate: float,
) -> float:
    key = parse_loss_key(layer, rate)
    if key in table:
        return table[key]
    return (rate * rate) * (1.0 + 0.015 * layer)
