#!/usr/bin/env python3
"""Benchmark hand-written SVD rate patterns under one or more load scenarios."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark_heterogeneous_schedule_cgroup import start_heterogeneous_load, stop_heterogeneous_load, write_csv
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, decode_once, make_cgroup
from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, cpu_spec


EXP_DIR = Path(__file__).resolve().parent


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def parse_rates(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def pattern_values(pattern: str, rate: float, n_layers: int) -> list[float]:
    if pattern == "all":
        return [rate] * n_layers
    if pattern == "odd":
        return [rate if layer % 2 == 1 else 0.0 for layer in range(n_layers)]
    if pattern == "even":
        return [rate if layer % 2 == 0 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_0":
        return [rate if layer % 3 == 0 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_1":
        return [rate if layer % 3 == 1 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_2":
        return [rate if layer % 3 == 2 else 0.0 for layer in range(n_layers)]
    if pattern == "second_half":
        return [rate if layer >= n_layers // 2 else 0.0 for layer in range(n_layers)]
    if pattern == "first_half":
        return [rate if layer < n_layers // 2 else 0.0 for layer in range(n_layers)]
    raise ValueError(f"unknown pattern: {pattern}")


def write_rate_file(out_dir: Path, pattern: str, rate: float, n_layers: int) -> Path:
    path = out_dir / f"{pattern}_{rate:g}.rates.txt"
    path.write_text(",".join(f"{value:g}" for value in pattern_values(pattern, rate, n_layers)) + "\n")
    return path


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    keys = sorted({(row["scenario"], row["policy"]) for row in rows})
    baselines: dict[str, float] = {}
    for scenario, policy in keys:
        vals = [float(row["decode_tok_s"]) for row in rows if row["scenario"] == scenario and row["policy"] == policy and row.get("decode_tok_s")]
        if policy == "baseline_no_svd" and vals:
            baselines[scenario] = float(statistics.median(vals))
    for scenario, policy in keys:
        vals = [float(row["decode_tok_s"]) for row in rows if row["scenario"] == scenario and row["policy"] == policy and row.get("decode_tok_s")]
        med = median(vals)
        base = baselines.get(scenario)
        out.append({
            "scenario": scenario,
            "policy": policy,
            "runs": len(vals),
            "tok_s_median": med,
            "tok_s_min": min(vals) if vals else None,
            "speedup_median": med / base if med is not None and base else None,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", required=True, type=Path)
    parser.add_argument("--patterns", default="all,odd,even,mod3_0,mod3_1,mod3_2")
    parser.add_argument("--rates", default="0.5,0.6,0.7,0.8,0.85,0.9")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/pattern_candidates_latest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text())
    patterns = [item for item in args.patterns.split(",") if item]
    rates = parse_rates(args.rates)
    rate_files = {
        f"{pattern}_{rate:g}": write_rate_file(args.out_dir, pattern, rate, args.n_layers)
        for pattern in patterns
        for rate in rates
    }
    policies = ["baseline_no_svd", *rate_files]
    rows: list[dict[str, Any]] = []
    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        run_cgroup = make_cgroup(args.cgroup_root, f"exp12_pattern_run_{scenario}", cpus)
        load_cgroup = make_cgroup(args.cgroup_root, f"exp12_pattern_load_{scenario}", cpus)
        for repeat in range(args.repeats):
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                for policy in policies:
                    rates_file = rate_files.get(policy)
                    row = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rates=rates_file or "off")
                    row.update({
                        "scenario": scenario,
                        "repeat": repeat,
                        "cpus": cpu_spec(cpus),
                        "loads": ",".join(str(load) for load in loads),
                        "policy": policy,
                    })
                    rows.append(row)
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)
    write_csv(args.out_dir / "raw.csv", rows, ["scenario", "repeat", "cpus", "loads", "policy", "status", "decode_tok_s", "generation_decode_ms", "elapsed_s", "output_tail"])
    summary = summarize(rows)
    write_csv(args.out_dir / "summary.csv", summary, ["scenario", "policy", "runs", "tok_s_median", "tok_s_min", "speedup_median"])
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
