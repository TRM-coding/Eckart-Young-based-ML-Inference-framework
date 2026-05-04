#!/usr/bin/env python3
"""Benchmark no-SVD major-only candidates under heterogeneous load.

Baseline uses all cores in the scenario.  Major-only candidates run the same
no-SVD model on the lowest-utilization prefix of those cores, leaving the
remaining high-utilization cores unused.  This keeps PPL equal to no-SVD while
testing whether avoiding busy cores improves throughput.
"""

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


def sorted_by_load(cpus: list[int], loads: list[int]) -> list[int]:
    return [cpu for cpu, _ in sorted(zip(cpus, loads), key=lambda item: (item[1], item[0]))]


def summarize(rows: list[dict[str, Any]], min_speedup: float, max_speedup: float) -> list[dict[str, Any]]:
    out = []
    for scenario in sorted({row["scenario"] for row in rows}):
        base_vals = [
            float(row["decode_tok_s"])
            for row in rows
            if row["scenario"] == scenario and row["policy"] == "baseline_all_cores" and row.get("decode_tok_s")
        ]
        base = median(base_vals)
        candidates = []
        for policy in sorted({row["policy"] for row in rows if row["scenario"] == scenario and row["policy"] != "baseline_all_cores"}):
            vals = [
                float(row["decode_tok_s"])
                for row in rows
                if row["scenario"] == scenario and row["policy"] == policy and row.get("decode_tok_s")
            ]
            med = median(vals)
            speed = med / base if med is not None and base else None
            meta = next(row for row in rows if row["scenario"] == scenario and row["policy"] == policy)
            candidates.append((policy, meta, len(vals), med, speed))
        in_band = [item for item in candidates if item[4] is not None and min_speedup <= item[4] <= max_speedup]
        safe = [item for item in candidates if item[4] is not None and item[4] >= 1.0]
        if in_band:
            center = (min_speedup + max_speedup) / 2.0
            selected = min(in_band, key=lambda item: abs(float(item[4]) - center))
        elif safe:
            selected = max(safe, key=lambda item: float(item[4]))
        else:
            meta = next(row for row in rows if row["scenario"] == scenario)
            selected = ("baseline_all_cores", meta, len(base_vals), base, 1.0)

        for policy, meta, runs, med, speed in candidates:
            out.append(
                {
                    "scenario": scenario,
                    "loads": meta["loads"],
                    "candidate_policy": policy,
                    "candidate_cpus": meta["run_cpus"],
                    "baseline_tok_s_median": base,
                    "candidate_tok_s_median": med,
                    "speedup_vs_baseline": speed,
                    "runs": runs,
                    "selected": "1" if policy == selected[0] else "0",
                    "selected_policy": selected[0],
                    "selected_speedup": selected[4],
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--min-speedup", type=float, default=1.10)
    parser.add_argument("--max-speedup", type=float, default=1.20)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/major_only_policy_latest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text())
    rows: list[dict[str, Any]] = []

    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        ordered = sorted_by_load(cpus, loads)
        load_cgroup = make_cgroup(args.cgroup_root, f"exp12_major_only_load_{scenario}", cpus)
        run_groups = {
            "baseline_all_cores": (cpus, make_cgroup(args.cgroup_root, f"exp12_major_only_run_{scenario}_all", cpus))
        }
        for p in range(1, len(ordered)):
            run_cpus = ordered[:p]
            run_groups[f"major_only_p{p}"] = (
                run_cpus,
                make_cgroup(args.cgroup_root, f"exp12_major_only_run_{scenario}_p{p}", run_cpus),
            )

        for repeat in range(args.repeats):
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                for policy, (run_cpus, run_cgroup) in run_groups.items():
                    row = decode_once(args.binary, args.model, run_cgroup, run_cpus, args.tokens, args.timeout_s)
                    row.update(
                        {
                            "scenario": scenario,
                            "repeat": repeat,
                            "loads": ",".join(str(load) for load in loads),
                            "policy": policy,
                            "run_cpus": cpu_spec(run_cpus),
                            "n_run_cpus": len(run_cpus),
                        }
                    )
                    rows.append(row)
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)

    write_csv(
        args.out_dir / "raw.csv",
        rows,
        ["scenario", "repeat", "loads", "policy", "run_cpus", "n_run_cpus", "status", "decode_tok_s", "generation_decode_ms", "elapsed_s", "output_tail"],
    )
    summary = summarize(rows, args.min_speedup, args.max_speedup)
    write_csv(
        args.out_dir / "summary.csv",
        summary,
        [
            "scenario",
            "loads",
            "candidate_policy",
            "candidate_cpus",
            "baseline_tok_s_median",
            "candidate_tok_s_median",
            "speedup_vs_baseline",
            "runs",
            "selected",
            "selected_policy",
            "selected_speedup",
        ],
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows), "scenarios": len(scenarios)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
