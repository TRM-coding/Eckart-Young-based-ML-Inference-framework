#!/usr/bin/env python3
"""Closed-loop runtime optimizer for exp12 scheduling candidates.

This script treats the scheduler as a policy selector over runtime-measured
candidates.  It uses a short calibration pass under the target heterogeneous
load, selects the fastest candidate only when it beats no-SVD by a configurable
margin, and then validates the selected policy with fresh runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

from benchmark_heterogeneous_schedule_cgroup import (
    start_heterogeneous_load,
    stop_heterogeneous_load,
    scenario_defaults,
    write_csv,
)
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, decode_once, make_cgroup
from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, cpu_spec


EXP_DIR = Path(__file__).resolve().parent


def write_even_rate_file(out_dir: Path, name: str, rate: float, n_layers: int = 28) -> Path:
    path = out_dir / f"{name}.rates.txt"
    values = [rate if layer % 2 == 1 else 0.0 for layer in range(n_layers)]
    path.write_text(",".join(f"{value:g}" for value in values) + "\n")
    return path


def pattern_values(pattern: str, rate: float, n_layers: int = 28) -> list[float]:
    if pattern == "all":
        return [rate] * n_layers
    if pattern in ("odd", "trunc_even"):
        return [rate if layer % 2 == 1 else 0.0 for layer in range(n_layers)]
    if pattern == "even":
        return [rate if layer % 2 == 0 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_0":
        return [rate if layer % 3 == 0 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_1":
        return [rate if layer % 3 == 1 else 0.0 for layer in range(n_layers)]
    if pattern == "mod3_2":
        return [rate if layer % 3 == 2 else 0.0 for layer in range(n_layers)]
    raise ValueError(f"unknown pattern: {pattern}")


def write_pattern_rate_file(out_dir: Path, pattern: str, rate: float, n_layers: int = 28) -> Path:
    path = out_dir / f"{pattern}_{rate:g}.rates.txt"
    path.write_text(",".join(f"{value:g}" for value in pattern_values(pattern, rate, n_layers)) + "\n")
    return path


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def min_value(values: list[float]) -> float | None:
    return min(values) if values else None


def run_policy(
    policy: str,
    binary: Path,
    model: Path,
    run_cgroup: Path,
    cpus: list[int],
    tokens: int,
    timeout_s: float,
    rate_files: dict[str, Path],
) -> dict[str, Any]:
    if policy == "baseline_no_svd":
        return decode_once(binary, model, run_cgroup, cpus, tokens, timeout_s)
    if policy.startswith("trunc_even_"):
        return decode_once(binary, model, run_cgroup, cpus, tokens, timeout_s, rates=rate_files[policy])
    if policy in rate_files:
        return decode_once(binary, model, run_cgroup, cpus, tokens, timeout_s, rates=rate_files[policy])
    raise ValueError(f"unknown policy: {policy}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop runtime optimized scheduler benchmark.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--calibration-repeats", type=int, default=2)
    parser.add_argument("--validation-repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--max-target-speedup", type=float, default=1.20)
    parser.add_argument("--guard-min-speedup", type=float, default=1.00)
    parser.add_argument("--rates", default="0.6,0.8,0.9")
    parser.add_argument("--patterns", default="odd", help="Comma-separated rate patterns: odd,all,even,mod3_0,mod3_1,mod3_2")
    parser.add_argument("--policy-overrides-json", type=Path)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/runtime_optimized_scheduler_20260429_r1")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text()) if args.scenarios_json else scenario_defaults()
    policy_overrides = json.loads(args.policy_overrides_json.read_text()) if args.policy_overrides_json else {}
    rate_values = [float(item) for item in args.rates.split(",") if item.strip()]
    patterns = [item for item in args.patterns.split(",") if item.strip()]
    if patterns == ["odd"]:
        policies = ["baseline_no_svd"] + [f"trunc_even_{rate:g}" for rate in rate_values]
        rate_files = {f"trunc_even_{rate:g}": write_even_rate_file(args.out_dir, f"trunc_even_{rate:g}", rate) for rate in rate_values}
    else:
        rate_files = {
            f"{pattern}_{rate:g}": write_pattern_rate_file(args.out_dir, pattern, rate)
            for pattern in patterns
            for rate in rate_values
        }
        policies = ["baseline_no_svd", *rate_files.keys()]

    raw_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        run_cgroup = make_cgroup(args.cgroup_root, f"exp12_rtopt_run_{scenario}", cpus)
        load_cgroup = make_cgroup(args.cgroup_root, f"exp12_rtopt_load_{scenario}", cpus)

        calibration: dict[str, list[float]] = {policy: [] for policy in policies}
        for rep in range(args.calibration_repeats):
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                for policy in policies:
                    row = run_policy(policy, args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rate_files)
                    row.update(
                        {
                            "phase": "calibration",
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": policy,
                        }
                    )
                    raw_rows.append(row)
                    if row.get("decode_tok_s"):
                        calibration[policy].append(float(row["decode_tok_s"]))
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)

        baseline_med = median(calibration["baseline_no_svd"])
        baseline_min = min_value(calibration["baseline_no_svd"])
        best_policy = "baseline_no_svd"
        best_med = baseline_med
        best_key: tuple[float, float, float] | None = None
        for policy in policies[1:]:
            med = median(calibration[policy])
            worst = min_value(calibration[policy])
            if med is None or worst is None or baseline_med is None or baseline_min is None:
                continue
            med_speedup = med / baseline_med
            worst_speedup = worst / baseline_min if baseline_min > 0 else 0.0
            if med_speedup < args.min_speedup or worst_speedup < args.guard_min_speedup:
                continue
            # Prefer policies in the requested 10-20% band; otherwise choose
            # the smallest overshoot above the band. This avoids selecting a
            # very aggressive truncation when a gentler one already works.
            overshoot = max(0.0, med_speedup - args.max_target_speedup)
            target_distance = abs(min(med_speedup, args.max_target_speedup) - ((args.min_speedup + args.max_target_speedup) / 2.0))
            key = (overshoot, target_distance, -med_speedup)
            if best_key is None or key < best_key:
                best_key = key
                best_policy = policy
                best_med = med
        selected_policy = best_policy
        if baseline_med is None or best_med is None or best_policy == "baseline_no_svd":
            selected_policy = "baseline_no_svd"
        selected_policy = str(policy_overrides.get(scenario, selected_policy))

        validation_base: list[float] = []
        validation_sched: list[float] = []
        if selected_policy == "edge_end":
            for rep in range(args.validation_repeats):
                load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
                try:
                    base_row = run_policy("baseline_no_svd", args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rate_files)
                    base_row.update(
                        {
                            "phase": "validation",
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": "baseline_no_svd",
                            "role": "baseline",
                            "selected_policy": selected_policy,
                        }
                    )
                    raw_rows.append(base_row)
                    if base_row.get("decode_tok_s"):
                        validation_base.append(float(base_row["decode_tok_s"]))
                    raw_rows.append(
                        {
                            "phase": "validation",
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": "edge_end",
                            "role": "scheduler",
                            "selected_policy": selected_policy,
                            "status": "skipped_edge_end_no_adb",
                        }
                    )
                finally:
                    stop_heterogeneous_load(load_cgroup, load_procs)
        else:
            for rep in range(args.validation_repeats):
                load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
                try:
                    base_row = run_policy("baseline_no_svd", args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rate_files)
                    base_row.update(
                        {
                            "phase": "validation",
                            "scenario": scenario,
                            "repeat": rep,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                            "policy": "baseline_no_svd",
                            "role": "baseline",
                            "selected_policy": selected_policy,
                        }
                    )
                    raw_rows.append(base_row)
                    if base_row.get("decode_tok_s"):
                        validation_base.append(float(base_row["decode_tok_s"]))

                    if selected_policy == "baseline_no_svd":
                        sched_row = dict(base_row)
                        sched_row.update(
                            {
                                "policy": "baseline_no_svd",
                                "role": "scheduler",
                                "status": "same_as_baseline_no_svd",
                            }
                        )
                    else:
                        sched_row = run_policy(selected_policy, args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rate_files)
                        sched_row.update(
                            {
                                "phase": "validation",
                                "scenario": scenario,
                                "repeat": rep,
                                "n_cores": len(cpus),
                                "cpus": cpu_spec(cpus),
                                "loads": ",".join(str(load) for load in loads),
                                "policy": selected_policy,
                                "role": "scheduler",
                                "selected_policy": selected_policy,
                            }
                        )
                    raw_rows.append(sched_row)
                    if sched_row.get("decode_tok_s"):
                        validation_sched.append(float(sched_row["decode_tok_s"]))
                finally:
                    stop_heterogeneous_load(load_cgroup, load_procs)

        val_base_med = median(validation_base)
        val_sched_med = median(validation_sched)
        decision_rows.append(
            {
                "scenario": scenario,
                "n_cores": len(cpus),
                "cpus": cpu_spec(cpus),
                "loads": ",".join(str(load) for load in loads),
                "calibration_baseline_tok_s": baseline_med,
                "calibration_best_policy": best_policy,
                "calibration_best_tok_s": best_med,
                "calibration_best_speedup": best_med / baseline_med if best_med and baseline_med else None,
                "selected_policy": selected_policy,
                "validation_baseline_tok_s": val_base_med,
                "validation_scheduler_tok_s": val_sched_med,
                "validation_speedup": val_sched_med / val_base_med if val_sched_med and val_base_med else None,
            }
        )

    write_csv(
        args.out_dir / "raw.csv",
        raw_rows,
        [
            "phase",
            "scenario",
            "repeat",
            "role",
            "selected_policy",
            "n_cores",
            "cpus",
            "loads",
            "policy",
            "status",
            "decode_tok_s",
            "generation_decode_ms",
            "prefill_decode_ms",
            "elapsed_s",
            "succeed",
            "top1_id",
            "top1_piece",
            "output_tail",
        ],
    )
    write_csv(
        args.out_dir / "decisions.csv",
        decision_rows,
        [
            "scenario",
            "n_cores",
            "cpus",
            "loads",
            "calibration_baseline_tok_s",
            "calibration_best_policy",
            "calibration_best_tok_s",
            "calibration_best_speedup",
            "selected_policy",
            "validation_baseline_tok_s",
            "validation_scheduler_tok_s",
            "validation_speedup",
        ],
    )

    lines = [
        "# Runtime Optimized Scheduler",
        "",
        f"- calibration repeats: `{args.calibration_repeats}`",
        f"- validation repeats: `{args.validation_repeats}`",
        f"- candidate policies: `{','.join(policies)}`",
        f"- minimum calibration speedup to enable a non-baseline policy: `{args.min_speedup:.3f}x`",
        "",
        "## Validation",
        "",
        "| scenario | cores | selected | baseline tok/s | scheduler tok/s | speedup |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in decision_rows:
        speed = row["validation_speedup"]
        lines.append(
            f"| `{row['scenario']}` | {row['n_cores']} | `{row['selected_policy']}` | "
            f"{row['validation_baseline_tok_s'] or 'n/a'} | {row['validation_scheduler_tok_s'] or 'n/a'} | "
            f"{speed:.3f}x |" if speed else
            f"| `{row['scenario']}` | {row['n_cores']} | `{row['selected_policy']}` | n/a | n/a | n/a |"
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "scenarios": len(scenarios), "raw_rows": len(raw_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
