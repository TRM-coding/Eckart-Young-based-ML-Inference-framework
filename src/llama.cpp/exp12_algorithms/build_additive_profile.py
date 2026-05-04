#!/usr/bin/env python3
"""Build Algorithmv2 5.2 scheduler profiles from additive core-speed curves.

Input:
  - single_core.csv from validate_core_additivity.py
  - optional additivity_error.csv for group-size calibration

Output:
  - one profile JSON per utilization scenario
  - each profile contains all p core splits, sorted by effective speed
  - each split contains layer/rate candidates with Tmain/Ttail

The generated profile is exact with respect to the measured additive model:
the scheduler can then perform an exhaustive DP over the finite candidate set
and choose the optimal plan under that model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from loss_table import load_measured_loss_table, measured_or_heuristic_loss


EXP_DIR = Path(__file__).resolve().parent


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            cpus.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def cpu_spec(cpus: list[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def parse_rates(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def median(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return None
    return float(statistics.median(clean))


def load_single_core_speeds(single_core_csv: Path) -> dict[tuple[int, int], float]:
    rows = read_csv(single_core_csv)
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        ms_s = row.get("generation_decode_ms")
        if not ms_s:
            continue
        ms = float(ms_s)
        if ms <= 0.0:
            continue
        key = (int(row["cpu"]), int(row["load_pct"]))
        grouped.setdefault(key, []).append(ms)
    speeds: dict[tuple[int, int], float] = {}
    for key, values in grouped.items():
        med = median(values)
        if med is not None and med > 0:
            speeds[key] = 1.0 / med
    return speeds


def load_calibration(error_csv: Path | None) -> dict[tuple[int, int], float]:
    if error_csv is None or not error_csv.exists():
        return {}
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in read_csv(error_csv):
        measured_s = row.get("measured_ms")
        predicted_s = row.get("predicted_ms")
        if not measured_s or not predicted_s:
            continue
        measured = float(measured_s)
        predicted = float(predicted_s)
        if measured <= 0.0 or predicted <= 0.0:
            continue
        key = (int(row["n_cpus"]), int(row["load_pct"]))
        grouped.setdefault(key, []).append(measured / predicted)
    return {key: float(statistics.median(values)) for key, values in grouped.items() if values}


def speed_for_cpu(speeds: dict[tuple[int, int], float], cpu: int, load: int) -> float:
    exact = speeds.get((cpu, load))
    if exact is not None:
        return exact
    available = sorted((abs(load - u), u, speed) for (c, u), speed in speeds.items() if c == cpu)
    if available:
        return available[0][2]
    raise KeyError(f"no speed measurements for cpu {cpu}")


def split_time_ms(
    work_units: float,
    cpus: list[int],
    util: dict[int, int],
    speeds: dict[tuple[int, int], float],
    calibration: dict[tuple[int, int], float],
) -> float:
    if work_units <= 0.0:
        return 0.0
    if not cpus:
        return math.inf
    total_speed = sum(speed_for_cpu(speeds, cpu, util[cpu]) for cpu in cpus)
    if total_speed <= 0.0:
        return math.inf
    avg_load = int(round(sum(util[cpu] for cpu in cpus) / len(cpus) / 10.0) * 10)
    factor = calibration_factor(calibration, len(cpus), avg_load)
    return work_units / total_speed * factor


def calibration_factor(calibration: dict[tuple[int, int], float], n_cpus: int, load: int) -> float:
    exact = calibration.get((n_cpus, load))
    if exact is not None:
        return exact
    if not calibration:
        return 1.0
    candidates = []
    for (n, u), factor in calibration.items():
        # Prefer nearby core counts first, then nearby load. This avoids making
        # unmeasured 5/7-core splits look unrealistically perfect.
        distance = abs(n - n_cpus) * 1000 + abs(u - load)
        candidates.append((distance, factor))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def layer_work_units(n_layers: int, layer: int) -> float:
    # Keep the model low-dimensional but not completely flat. The shape factors
    # sum approximately to one across layers.
    shape = 1.0 + 0.08 * ((layer % 4) - 1.5)
    denom = sum(1.0 + 0.08 * ((i % 4) - 1.5) for i in range(n_layers))
    return shape / denom


def candidate_loss(layer: int, rate: float, loss_table: dict[tuple[int, str], float] | None = None) -> float:
    return measured_or_heuristic_loss(loss_table or {}, layer, rate)


def build_profile_for_utilization(
    cpus: list[int],
    util: dict[int, int],
    speeds: dict[tuple[int, int], float],
    calibration: dict[tuple[int, int], float],
    n_layers: int,
    rates: list[float],
    timeout_budget_ms: float,
    request_deadline_factor: float,
    local_deadline_factor: float,
    reference_full_local_ms: float | None,
    tx_base_ms: float,
    tx_per_layer_ms: float,
    end_per_layer_ms: float,
    loss_table: dict[tuple[int, str], float] | None = None,
) -> dict[str, Any]:
    sorted_cpus = sorted(cpus, key=lambda cpu: speed_for_cpu(speeds, cpu, util[cpu]), reverse=True)
    full_local_ms = sum(
        split_time_ms(layer_work_units(n_layers, layer), sorted_cpus, util, speeds, calibration)
        for layer in range(n_layers)
    )
    deadline_base_ms = reference_full_local_ms if reference_full_local_ms is not None else full_local_ms
    local_deadline_ms = deadline_base_ms * local_deadline_factor
    request_deadline_ms = deadline_base_ms * request_deadline_factor

    core_splits = []
    for p in range(1, len(sorted_cpus) + 1):
        major = sorted_cpus[:p]
        minor = sorted_cpus[p:]
        split_layers = []
        split_rates = [0.0] if not minor else rates
        for layer in range(n_layers):
            work = layer_work_units(n_layers, layer)
            candidates = []
            for rate in split_rates:
                rate = max(0.0, min(0.99, rate))
                main_work = work * (1.0 - rate)
                tail_work = work * rate
                main_ms = split_time_ms(main_work, major, util, speeds, calibration)
                tail_ms = split_time_ms(tail_work, minor, util, speeds, calibration) if minor else 0.0
                weight = tail_ms if tail_ms > 0 else rate
                candidates.append(
                    {
                        "rate": round(rate, 6),
                        "main_ms": round(main_ms, 6),
                        "tail_ms": round(tail_ms, 6),
                        "loss": round(candidate_loss(layer, rate, loss_table), 8),
                        "weight": round(weight, 8),
                    }
                )
            split_layers.append({"layer": layer, "candidates": candidates})
        core_splits.append(
            {
                "split_id": f"p{p}_major_{cpu_spec(major).replace(',', '_')}",
                "p": p,
                "major_cpus": major,
                "minor_cpus": minor,
                "layers": split_layers,
            }
        )

    tx_by_split = []
    end_by_split = []
    for split_m in range(n_layers + 1):
        offloaded = max(0, n_layers - split_m - 1)
        tx_by_split.append(round(tx_base_ms + tx_per_layer_ms * offloaded, 6))
        end_by_split.append(round(end_per_layer_ms * offloaded, 6))

    return {
        "source": "additive core-speed profile from validate_core_additivity.py",
        "model": "T ~= work / sum(core_speed(util)) * calibration(n_cpus, avg_util)",
        "loss_source": "measured spectral residual norm" if loss_table else "heuristic placeholder",
        "n_layers": n_layers,
        "cpus": cpus,
        "utilization": {str(cpu): util[cpu] for cpu in cpus},
        "sorted_cpus_by_effective_speed": sorted_cpus,
        "full_local_ms": round(full_local_ms, 6),
        "deadline_base_ms": round(deadline_base_ms, 6),
        "local_deadline_ms": round(local_deadline_ms, 6),
        "request_deadline_ms": round(request_deadline_ms, 6),
        "timeout_budget_ms": timeout_budget_ms,
        "tx_ms_by_split_m": tx_by_split,
        "end_ms_by_split_m": end_by_split,
        "core_splits": core_splits,
    }


def uniform_util(cpus: list[int], load: int) -> dict[int, int]:
    return {cpu: load for cpu in cpus}


def parse_util_vector(cpus: list[int], spec: str) -> dict[int, int]:
    util = {cpu: 0 for cpu in cpus}
    for item in spec.split(","):
        if not item.strip():
            continue
        cpu_s, load_s = item.split(":", 1)
        util[int(cpu_s)] = int(load_s)
    return util


def main() -> int:
    parser = argparse.ArgumentParser(description="Build additive Algorithmv2 scheduler profiles.")
    parser.add_argument("--single-core-csv", required=True, type=Path)
    parser.add_argument("--additivity-error-csv", type=Path)
    parser.add_argument("--cpus", default="60-67")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--rates", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--uniform-loads", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--utilization")
    parser.add_argument("--timeout-budget-ms", type=float, default=40.0)
    parser.add_argument("--local-deadline-factor", type=float, default=1.05)
    parser.add_argument("--request-deadline-factor", type=float, default=1.35)
    parser.add_argument("--deadline-ms", type=float, help="Fixed local deadline base. Defaults to idle p=M full local time.")
    parser.add_argument("--tx-base-ms", type=float, default=3.0)
    parser.add_argument("--tx-per-layer-ms", type=float, default=0.09)
    parser.add_argument("--end-per-layer-ms", type=float, default=0.62)
    parser.add_argument("--loss-table", type=Path, help="CSV from measure_svd_matrix_loss.py, normally svd_loss_for_scheduler.csv")
    parser.add_argument("--loss-reduction", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/additive_profiles_latest")
    args = parser.parse_args()

    cpus = parse_cpu_list(args.cpus)
    rates = parse_rates(args.rates)
    speeds = load_single_core_speeds(args.single_core_csv)
    calibration = load_calibration(args.additivity_error_csv)
    loss_table = load_measured_loss_table(args.loss_table, args.loss_reduction)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[tuple[str, dict[int, int]]] = []
    if args.utilization:
        scenarios.append(("custom", parse_util_vector(cpus, args.utilization)))
    else:
        for load in [int(item) for item in args.uniform_loads.split(",") if item.strip()]:
            scenarios.append((f"load_{load}", uniform_util(cpus, load)))

    reference_full_local_ms = args.deadline_ms
    if reference_full_local_ms is None:
        idle_util = uniform_util(cpus, 0)
        idle_sorted = sorted(cpus, key=lambda cpu: speed_for_cpu(speeds, cpu, idle_util[cpu]), reverse=True)
        reference_full_local_ms = sum(
            split_time_ms(layer_work_units(args.n_layers, layer), idle_sorted, idle_util, speeds, calibration)
            for layer in range(args.n_layers)
        )

    index = []
    for name, util in scenarios:
        profile = build_profile_for_utilization(
            cpus=cpus,
            util=util,
            speeds=speeds,
            calibration=calibration,
            n_layers=args.n_layers,
            rates=rates,
            timeout_budget_ms=args.timeout_budget_ms,
            request_deadline_factor=args.request_deadline_factor,
            local_deadline_factor=args.local_deadline_factor,
            reference_full_local_ms=reference_full_local_ms,
            tx_base_ms=args.tx_base_ms,
            tx_per_layer_ms=args.tx_per_layer_ms,
            end_per_layer_ms=args.end_per_layer_ms,
            loss_table=loss_table,
        )
        path = args.out_dir / f"profile_{name}.json"
        path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n")
        index.append(
            {
                "name": name,
                "profile": str(path),
                "full_local_ms": profile["full_local_ms"],
                "local_deadline_ms": profile["local_deadline_ms"],
                "request_deadline_ms": profile["request_deadline_ms"],
            }
        )

    (args.out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    report = [
        "# Additive Scheduler Profiles",
        "",
        f"- CPUs: `{cpu_spec(cpus)}`",
        f"- Rates: `{','.join(str(rate) for rate in rates)}`",
        f"- Scenarios: `{', '.join(item['name'] for item in index)}`",
        f"- Loss source: `{'measured spectral residual norm' if loss_table else 'heuristic placeholder'}`",
        "",
        "| scenario | full local ms | local deadline | request deadline | profile |",
        "|---|---:|---:|---:|---|",
    ]
    for item in index:
        report.append(
            f"| `{item['name']}` | {item['full_local_ms']:.4f} | "
            f"{item['local_deadline_ms']:.4f} | {item['request_deadline_ms']:.4f} | `{Path(item['profile']).name}` |"
        )
    report.extend([
        "",
        "The scheduler should prefer `p=M`, `rate=0` when that local full-execution plan is feasible, because it has zero loss and the tie-breaker prefers larger `p`.",
        "",
    ])
    (args.out_dir / "PROFILE_REPORT.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(args.out_dir), "profiles": index}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
