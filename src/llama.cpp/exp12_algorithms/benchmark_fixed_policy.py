#!/usr/bin/env python3
"""Validate a fixed runtime policy map without re-running expensive search."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from benchmark_heterogeneous_schedule_cgroup import start_heterogeneous_load, stop_heterogeneous_load, write_csv
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, decode_once, make_cgroup
from benchmark_runtime_optimized_scheduler import pattern_values
from run_exp12_local import DEFAULT_BINARY, DEFAULT_MODEL, cpu_spec


EXP_DIR = Path(__file__).resolve().parent


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def parse_policy(policy: str) -> tuple[str, float] | None:
    if policy in ("baseline_no_svd", "edge_end", "edge_end_no_svd"):
        return None
    if policy.startswith("trunc_even_"):
        return ("odd", float(policy.rsplit("_", 1)[1]))
    pattern, rate_s = policy.rsplit("_", 1)
    return pattern, float(rate_s)


def write_rate_file(out_dir: Path, policy: str, n_layers: int) -> Path | None:
    parsed = parse_policy(policy)
    if parsed is None:
        return None
    pattern, rate = parsed
    path = out_dir / f"{policy}.rates.txt"
    path.write_text(",".join(f"{value:g}" for value in pattern_values(pattern, rate, n_layers)) + "\n")
    return path


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for scenario in sorted({row["scenario"] for row in rows}):
        base = [
            float(row["decode_tok_s"])
            for row in rows
            if row["scenario"] == scenario and row["role"] == "baseline" and row.get("decode_tok_s")
        ]
        sched = [
            float(row["decode_tok_s"])
            for row in rows
            if row["scenario"] == scenario and row["role"] == "scheduler" and row.get("decode_tok_s")
        ]
        base_med = median(base)
        sched_med = median(sched)
        meta = next(row for row in rows if row["scenario"] == scenario)
        out.append(
            {
                "scenario": scenario,
                "n_cores": meta["n_cores"],
                "cpus": meta["cpus"],
                "loads": meta["loads"],
                "selected_policy": meta["selected_policy"],
                "baseline_runs": len(base),
                "scheduler_runs": len(sched),
                "baseline_tok_s_median": base_med,
                "scheduler_tok_s_median": sched_med,
                "speedup_median": sched_med / base_med if sched_med is not None and base_med else None,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--scenarios-json", required=True, type=Path)
    parser.add_argument("--policy-map", required=True, type=Path)
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/fixed_policy_latest")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text())
    policy_map = json.loads(args.policy_map.read_text())
    rate_files = {
        policy: write_rate_file(args.out_dir, policy, args.n_layers)
        for policy in sorted(set(policy_map.values()))
        if policy not in ("baseline_no_svd", "edge_end", "edge_end_no_svd")
    }

    rows: list[dict[str, Any]] = []
    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        selected = str(policy_map.get(scenario, "baseline_no_svd"))
        run_cgroup = make_cgroup(args.cgroup_root, f"exp12_fixed_run_{scenario}", cpus)
        load_cgroup = make_cgroup(args.cgroup_root, f"exp12_fixed_load_{scenario}", cpus)
        for repeat in range(args.repeats):
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                base = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s)
                base.update(
                    {
                        "scenario": scenario,
                        "repeat": repeat,
                        "role": "baseline",
                        "selected_policy": selected,
                        "n_cores": len(cpus),
                        "cpus": cpu_spec(cpus),
                        "loads": ",".join(str(load) for load in loads),
                    }
                )
                rows.append(base)
                if selected in ("baseline_no_svd", "edge_end", "edge_end_no_svd"):
                    sched = dict(base)
                    sched.update(
                        {
                            "role": "scheduler",
                            "status": "same_as_baseline_no_svd" if selected == "baseline_no_svd" else f"skipped_{selected}_no_adb",
                            "decode_tok_s": "" if selected != "baseline_no_svd" else base.get("decode_tok_s", ""),
                            "generation_decode_ms": "" if selected != "baseline_no_svd" else base.get("generation_decode_ms", ""),
                        }
                    )
                else:
                    sched = decode_once(args.binary, args.model, run_cgroup, cpus, args.tokens, args.timeout_s, rates=rate_files[selected])
                    sched.update(
                        {
                            "scenario": scenario,
                            "repeat": repeat,
                            "role": "scheduler",
                            "selected_policy": selected,
                            "n_cores": len(cpus),
                            "cpus": cpu_spec(cpus),
                            "loads": ",".join(str(load) for load in loads),
                        }
                    )
                rows.append(sched)
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)

    write_csv(
        args.out_dir / "raw.csv",
        rows,
        ["scenario", "repeat", "role", "selected_policy", "n_cores", "cpus", "loads", "status", "decode_tok_s", "generation_decode_ms", "elapsed_s", "output_tail"],
    )
    summary = summarize(rows)
    write_csv(
        args.out_dir / "summary.csv",
        summary,
        ["scenario", "n_cores", "cpus", "loads", "selected_policy", "baseline_runs", "scheduler_runs", "baseline_tok_s_median", "scheduler_tok_s_median", "speedup_median"],
    )
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
