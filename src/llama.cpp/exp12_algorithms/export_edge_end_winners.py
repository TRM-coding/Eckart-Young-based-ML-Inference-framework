#!/usr/bin/env python3
"""Export plot-ready rows where edge_end_no_svd wins.

The input is a raw.csv produced by benchmark_q4_lossless_scheduler.py.  The
export is intentionally strict: a row is kept only when the best measured
edge_end_no_svd candidate is faster than both the baseline_no_svd median and
the best measured major_only_no_svd candidate for the same scenario.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


M_RE = re.compile(r"M=(\d+)")


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def best(rows: list[dict[str, str]]) -> dict[str, str] | None:
    ok = [row for row in rows if row.get("status") == "ok" and row.get("decode_tok_s")]
    if not ok:
        return None
    return max(ok, key=lambda row: float(row["decode_tok_s"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export edge_end winner rows from a scheduler raw.csv.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-occupied", type=int, default=8)
    parser.add_argument("--min-load", type=int, default=70)
    parser.add_argument("--ppl", type=float, default=15.1424)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.raw.open(newline="")))
    out_rows: list[dict[str, object]] = []
    for scenario in sorted({row["scenario"] for row in rows}):
        sc_rows = [row for row in rows if row["scenario"] == scenario]
        if not sc_rows:
            continue
        loads = [int(x) for x in sc_rows[0].get("loads", "").split(",") if x.strip()]
        occupied = sum(1 for load in loads if load > 0)
        min_load = min(loads) if loads else 0
        if occupied < args.min_occupied or min_load < args.min_load:
            continue

        baseline_vals = [
            float(row["decode_tok_s"])
            for row in sc_rows
            if row["policy"] == "baseline_no_svd" and row.get("status") == "ok" and row.get("decode_tok_s")
        ]
        baseline_med = median(baseline_vals)
        major_best = best([row for row in sc_rows if row["policy"] == "major_only_no_svd"])
        edge_best = best([row for row in sc_rows if row["policy"] == "edge_end_no_svd"])
        if baseline_med is None or major_best is None or edge_best is None:
            continue

        major_tok = float(major_best["decode_tok_s"])
        edge_tok = float(edge_best["decode_tok_s"])
        if not (edge_tok > baseline_med and edge_tok > major_tok):
            continue

        m = ""
        match = M_RE.search(edge_best.get("detail", ""))
        if match:
            m = match.group(1)
        out_rows.append(
            {
                "scenario": scenario,
                "cpus": sc_rows[0].get("cpus", ""),
                "loads": ",".join(map(str, loads)),
                "occupied_cores": occupied,
                "baseline_tok_s_median": baseline_med,
                "major_only_best_tok_s": major_tok,
                "major_only_best_detail": major_best.get("detail", ""),
                "edge_end_best_tok_s": edge_tok,
                "edge_end_best_detail": edge_best.get("detail", ""),
                "edge_end_split_m": m,
                "speedup_vs_baseline": edge_tok / baseline_med,
                "speedup_vs_major_only": edge_tok / major_tok,
                "ppl": args.ppl,
                "prefix_decode_ms": edge_best.get("prefix_decode_ms", ""),
                "server_decode_ms": edge_best.get("server_decode_ms", ""),
                "network_wait_ms": edge_best.get("network_wait_ms", ""),
                "client_log": edge_best.get("client_log", ""),
                "server_log": edge_best.get("server_log", ""),
            }
        )

    fieldnames = [
        "scenario",
        "cpus",
        "loads",
        "occupied_cores",
        "baseline_tok_s_median",
        "major_only_best_tok_s",
        "major_only_best_detail",
        "edge_end_best_tok_s",
        "edge_end_best_detail",
        "edge_end_split_m",
        "speedup_vs_baseline",
        "speedup_vs_major_only",
        "ppl",
        "prefix_decode_ms",
        "server_decode_ms",
        "network_wait_ms",
        "client_log",
        "server_log",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
