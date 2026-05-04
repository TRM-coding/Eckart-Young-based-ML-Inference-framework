#!/usr/bin/env python3
"""Attach measured layer-coop offload candidates to an exp12 scheduler profile.

The layer-coop runtime reports split `M` with the convention:

  PC    = [0, M)
  Phone = [M, n_layers)

This helper reads `benchmark_layer_coop_split_load.py` raw.csv files, aggregates
repeats by median, and writes those measured end-to-end per-token latencies into
the scheduler profile as `offload_candidates`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(statistics.median(clean))


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def build_candidates(raw_csvs: list[Path], load_pct: int | None, n_layers: int) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, str]]] = {}
    row_sources: dict[int, list[str]] = {}
    for raw_csv in raw_csvs:
        for row in read_csv(raw_csv):
            if row.get("status") != "ok":
                continue
            if load_pct is not None and int(float(row.get("load_pct", -1))) != load_pct:
                continue
            if not row.get("split_m"):
                continue
            m = max(0, min(n_layers, int(float(row["split_m"]))))
            groups.setdefault(m, []).append(row)
            row_sources.setdefault(m, []).append(str(raw_csv))

    candidates: list[dict[str, Any]] = []
    for m, rows in sorted(groups.items()):
        total_ms = median([fnum(row, "steady_ms_per_token", math.nan) for row in rows])
        if total_ms is None:
            continue
        pc_ms = median([fnum(row, "prefix_decode_ms", math.nan) / max(1.0, fnum(row, "generated", 1.0)) for row in rows])
        phone_ms = median([fnum(row, "server_decode_ms", math.nan) / max(1.0, fnum(row, "generated", 1.0)) for row in rows])
        network_ms = median([fnum(row, "network_wait_ms", math.nan) / max(1.0, fnum(row, "generated", 1.0)) for row in rows])
        throughput = median([fnum(row, "steady_throughput", math.nan) for row in rows])
        candidates.append(
            {
                "m": m,
                "total_ms": round(total_ms, 6),
                "pc_ms": round(pc_ms or 0.0, 6),
                "phone_ms": round(phone_ms or 0.0, 6),
                "network_ms": round(network_ms or 0.0, 6),
                "steady_throughput": round(throughput or 0.0, 6),
                "repeats": len(rows),
                "source": ";".join(sorted(set(row_sources.get(m, [])))),
            }
        )
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Attach measured layer-offload candidates to a scheduler profile.")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--layer-coop-raw", required=True, type=Path, action="append")
    parser.add_argument("--load-pct", type=int, help="Only use rows from this PC load. Defaults to all rows.")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--out-profile", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    n_layers = int(profile.get("n_layers", args.n_layers))
    candidates = build_candidates(args.layer_coop_raw, args.load_pct, n_layers)
    profile["offload_candidates"] = candidates
    profile["offload_candidate_source"] = {
        "raw_csv": [str(path) for path in args.layer_coop_raw],
        "load_pct": args.load_pct,
        "semantics": "PC=[0,M), Phone=[M,n_layers)",
        "aggregation": "median by M",
    }
    args.out_profile.parent.mkdir(parents=True, exist_ok=True)
    args.out_profile.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(
            args.out_csv,
            candidates,
            ["m", "total_ms", "pc_ms", "phone_ms", "network_ms", "steady_throughput", "repeats", "source"],
        )
    print(json.dumps({"out_profile": str(args.out_profile), "candidates": len(candidates)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
