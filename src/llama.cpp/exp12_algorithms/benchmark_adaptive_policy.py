#!/usr/bin/env python3
"""Adaptive runtime guard for exp12 SVD policies.

The fixed policy experiments showed that heterogeneous background load has
large run-to-run tails.  A policy that is 1.15x in one window can be 0.95x or
1.30x in the next.  This benchmark models the runtime guard used by the
scheduler: under the current load window, measure a small candidate set and
enable a non-baseline policy only when the measured speedup is inside the
requested band.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark_fixed_policy import write_rate_file
from benchmark_heterogeneous_schedule_cgroup import start_heterogeneous_load, stop_heterogeneous_load, write_csv
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, decode_once, make_cgroup
from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, cpu_spec


EXP_DIR = Path(__file__).resolve().parent


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def parse_policy_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scenario in sorted({row["scenario"] for row in rows if row["phase"] == "selected"}):
        selected_rows = [row for row in rows if row["scenario"] == scenario and row["phase"] == "selected"]
        base = [float(row["baseline_tok_s"]) for row in selected_rows if row.get("baseline_tok_s")]
        sched = [float(row["selected_tok_s"]) for row in selected_rows if row.get("selected_tok_s")]
        speed = [float(row["selected_speedup"]) for row in selected_rows if row.get("selected_speedup")]
        meta = selected_rows[0]
        policies = sorted({row["selected_policy"] for row in selected_rows})
        out.append(
            {
                "scenario": scenario,
                "n_cores": meta["n_cores"],
                "cpus": meta["cpus"],
                "loads": meta["loads"],
                "runs": len(selected_rows),
                "selected_policies": ",".join(policies),
                "baseline_tok_s_median": median(base),
                "scheduler_tok_s_median": median(sched),
                "speedup_median": median(speed),
                "speedup_min": min(speed) if speed else None,
                "speedup_max": max(speed) if speed else None,
                "in_band_runs": sum(1 for value in speed if 1.10 <= value <= 1.20),
                "fallback_runs": sum(1 for row in selected_rows if row["selected_policy"] == "baseline_no_svd"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", required=True, type=Path)
    parser.add_argument("--candidate-policies", default="all_0.3,all_0.4,all_0.5,all_0.6,all_0.7,trunc_even_0.8,trunc_even_0.85,odd_0.5,even_0.5,mod3_2_0.85")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--max-speedup", type=float, default=1.20)
    parser.add_argument("--strict-band", action="store_true", help="Fallback to baseline when no candidate is inside the target band.")
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/adaptive_policy_latest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text())
    policies = parse_policy_list(args.candidate_policies)
    rate_files = {policy: write_rate_file(args.out_dir, policy, args.n_layers) for policy in policies}
    center = (args.min_speedup + args.max_speedup) / 2.0

    rows: list[dict[str, Any]] = []
    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        run_cgroup = make_cgroup(args.cgroup_root, f"exp12_adaptive_run_{scenario}", cpus)
        load_cgroup = make_cgroup(args.cgroup_root, f"exp12_adaptive_load_{scenario}", cpus)
        for repeat in range(args.repeats):
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                baseline = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s)
                base_tok = float(baseline["decode_tok_s"]) if baseline.get("decode_tok_s") else 0.0
                baseline.update(
                    {
                        "phase": "candidate",
                        "scenario": scenario,
                        "repeat": repeat,
                        "policy": "baseline_no_svd",
                        "n_cores": len(cpus),
                        "cpus": cpu_spec(cpus),
                        "loads": ",".join(str(load) for load in loads),
                        "speedup": 1.0 if base_tok else "",
                    }
                )
                rows.append(baseline)

                measured: list[tuple[str, dict[str, Any], float]] = []
                for policy in policies:
                    row = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rates=rate_files[policy])
                    tok = float(row["decode_tok_s"]) if row.get("decode_tok_s") else 0.0
                    speedup = tok / base_tok if base_tok > 0.0 and tok > 0.0 else 0.0
                    row.update(
                        {
                            "phase": "candidate",
                            "scenario": scenario,
                            "repeat": repeat,
                            "policy": policy,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "speedup": speedup if speedup else "",
                        }
                    )
                    rows.append(row)
                    if speedup > 0.0:
                        measured.append((policy, row, speedup))

                in_band = [(abs(speed - center), policy, row, speed) for policy, row, speed in measured if args.min_speedup <= speed <= args.max_speedup]
                safe = [(abs(speed - center), speed, policy, row) for policy, row, speed in measured if speed >= 1.0]
                if in_band:
                    _, selected_policy, selected_row, selected_speedup = min(in_band, key=lambda item: (item[0], item[3]))
                elif safe and not args.strict_band:
                    _, selected_speedup, selected_policy, selected_row = min(safe, key=lambda item: (item[0], item[1]))
                else:
                    selected_policy = "baseline_no_svd"
                    selected_row = baseline
                    selected_speedup = 1.0

                rows.append(
                    {
                        "phase": "selected",
                        "scenario": scenario,
                        "repeat": repeat,
                        "policy": selected_policy,
                        "selected_policy": selected_policy,
                        "status": selected_row.get("status", ""),
                        "n_cores": len(cpus),
                        "cpus": cpu_spec(cpus),
                        "loads": ",".join(str(load) for load in loads),
                        "baseline_tok_s": base_tok,
                        "selected_tok_s": selected_row.get("decode_tok_s", ""),
                        "selected_speedup": selected_speedup,
                    }
                )
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)

    write_csv(
        args.out_dir / "raw.csv",
        rows,
        [
            "phase",
            "scenario",
            "repeat",
            "policy",
            "selected_policy",
            "n_cores",
            "cpus",
            "loads",
            "status",
            "decode_tok_s",
            "generation_decode_ms",
            "speedup",
            "baseline_tok_s",
            "selected_tok_s",
            "selected_speedup",
            "elapsed_s",
            "output_tail",
        ],
    )
    write_csv(
        args.out_dir / "summary.csv",
        summarize(rows),
        [
            "scenario",
            "n_cores",
            "cpus",
            "loads",
            "runs",
            "selected_policies",
            "baseline_tok_s_median",
            "scheduler_tok_s_median",
            "speedup_median",
            "speedup_min",
            "speedup_max",
            "in_band_runs",
            "fallback_runs",
        ],
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
