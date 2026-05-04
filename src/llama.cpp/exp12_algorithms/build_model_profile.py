#!/usr/bin/env python3
"""Build scheduler profiles from a trained sklearn latency model.

This connects the measured latency predictor to scheduler.py without changing
the DP solver: the model predicts the base latency for a CPU set under a load
vector, and this script converts those predictions into the existing
core_splits profile schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import subprocess
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


def parse_loads(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item.strip()]


def uniform_util(cpus: list[int], load: int) -> dict[int, int]:
    return {cpu: load for cpu in cpus}


def parse_util_vector(cpus: list[int], spec: str) -> dict[int, int]:
    util = {cpu: 0 for cpu in cpus}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cpu_s, load_s = item.split(":", 1)
        util[int(cpu_s)] = int(load_s)
    return util


def layer_work_units(n_layers: int, layer: int) -> float:
    shape = 1.0 + 0.08 * ((layer % 4) - 1.5)
    denom = sum(1.0 + 0.08 * ((i % 4) - 1.5) for i in range(n_layers))
    return shape / denom


def candidate_loss(layer: int, rate: float, loss_table: dict[tuple[int, str], float] | None = None) -> float:
    return measured_or_heuristic_loss(loss_table or {}, layer, rate)


def row_to_features(active_cpus: list[int], util: dict[int, int], q_pct: int, cpus: list[int]) -> list[float]:
    active_set = set(active_cpus)
    active = {cpu: 1.0 if cpu in active_set else 0.0 for cpu in cpus}
    q_values = {cpu: float(util.get(cpu, 0)) for cpu in cpus}
    active_loads = [q_values[cpu] for cpu in cpus if active[cpu] > 0.0]

    if active_loads:
        load_min = min(active_loads)
        load_mean = sum(active_loads) / len(active_loads)
        load_max = max(active_loads)
        mean = load_mean
        load_std = math.sqrt(sum((value - mean) ** 2 for value in active_loads) / len(active_loads)) if len(active_loads) > 1 else 0.0
        count_ge_50 = sum(1 for value in active_loads if value >= 50)
        count_ge_80 = sum(1 for value in active_loads if value >= 80)
        count_eq_100 = sum(1 for value in active_loads if value >= 100)
    else:
        load_min = load_mean = load_max = load_std = 0.0
        count_ge_50 = count_ge_80 = count_eq_100 = 0

    values: list[float] = [float(q_pct), float(len(active_cpus))]
    values.extend(active[cpu] for cpu in cpus)
    values.extend(q_values[cpu] for cpu in cpus)
    values.extend(active[cpu] * q_values[cpu] for cpu in cpus)
    values.extend(
        [
            load_min,
            load_mean,
            load_max,
            load_std,
            float(count_ge_50),
            float(count_ge_80),
            float(count_eq_100),
            load_std * float(count_ge_80),
            load_std * float(count_eq_100),
            load_max * float(count_ge_80),
        ]
    )
    return values


class LatencyPredictor:
    def __init__(self, model_path: Path) -> None:
        with model_path.open("rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_names = list(payload["feature_names"])
        self.cpus = [int(cpu) for cpu in payload["cpus"]]

    def predict_ms(self, active_cpus: list[int], util: dict[int, int], q_pct: int) -> float:
        import numpy as np

        x = row_to_features(active_cpus, util, q_pct, self.cpus)
        if len(x) != len(self.feature_names):
            raise ValueError(f"feature length mismatch: got {len(x)}, expected {len(self.feature_names)}")
        pred = float(self.model.predict(np.asarray([x], dtype=float))[0])
        return max(0.0, pred)


def build_profile(
    predictor: LatencyPredictor,
    cpus: list[int],
    util: dict[int, int],
    n_layers: int,
    rates: list[float],
    q_pct: int,
    timeout_budget_ms: float,
    request_deadline_factor: float,
    local_deadline_factor: float,
    reference_full_local_ms: float | None,
    tx_base_ms: float,
    tx_per_layer_ms: float,
    end_per_layer_ms: float,
    loss_table: dict[tuple[int, str], float] | None = None,
    scheduler_min_speedup_vs_baseline: float = 1.0,
    scheduler_max_speedup_vs_baseline: float = 0.0,
    quality_first_no_svd: bool = False,
) -> dict[str, Any]:
    # Algorithmv2 assumes major cores are the lower-utilization cores and
    # minor/tail cores are the higher-utilization cores.  The latency model is
    # still useful as a tie-breaker among cores with similar utilization, but
    # it must not invert the major/minor semantic.
    sorted_cpus = sorted(cpus, key=lambda cpu: (util.get(cpu, 0), predictor.predict_ms([cpu], util, q_pct), cpu))
    full_base_ms = predictor.predict_ms(sorted_cpus, util, q_pct)
    full_local_ms = sum(layer_work_units(n_layers, layer) * full_base_ms for layer in range(n_layers))
    deadline_base_ms = reference_full_local_ms if reference_full_local_ms is not None else full_local_ms
    local_deadline_ms = deadline_base_ms * local_deadline_factor
    request_deadline_ms = deadline_base_ms * request_deadline_factor

    core_splits = []
    for p in range(1, len(sorted_cpus) + 1):
        major = sorted_cpus[:p]
        minor = sorted_cpus[p:]
        major_base_ms = predictor.predict_ms(major, util, q_pct)
        minor_base_ms = predictor.predict_ms(minor, util, q_pct) if minor else 0.0
        split_rates = [0.0] if not minor else rates
        layers = []
        for layer in range(n_layers):
            work = layer_work_units(n_layers, layer)
            candidates = []
            for rate in split_rates:
                rate = max(0.0, min(0.99, rate))
                main_ms = major_base_ms * work * (1.0 - rate)
                tail_ms = minor_base_ms * work * rate if minor else 0.0
                weight = tail_ms if tail_ms > 0.0 else rate
                candidates.append(
                    {
                        "rate": round(rate, 6),
                        "main_ms": round(main_ms, 6),
                        "tail_ms": round(tail_ms, 6),
                        "loss": round(candidate_loss(layer, rate, loss_table), 8),
                        "weight": round(weight, 8),
                    }
                )
            layers.append({"layer": layer, "candidates": candidates})
        core_splits.append(
            {
                "split_id": f"model_p{p}_major_{cpu_spec(major).replace(',', '_')}",
                "p": p,
                "major_cpus": major,
                "minor_cpus": minor,
                "model_pred_major_base_ms": round(major_base_ms, 6),
                "model_pred_minor_base_ms": round(minor_base_ms, 6),
                "layers": layers,
            }
        )

    tx_by_split = []
    end_by_split = []
    for split_m in range(n_layers + 1):
        offloaded = max(0, n_layers - split_m - 1)
        tx_by_split.append(round(tx_base_ms + tx_per_layer_ms * offloaded, 6))
        end_by_split.append(round(end_per_layer_ms * offloaded, 6))

    return {
        "source": "sklearn latency-model profile from fit_latency_model.py",
        "model": "T_pred = exp(ExtraTrees(features)) - 1; layer/rate scaling is linear because current measured model has fixed Q",
        "model_limitations": [
            "Current trained model was fit with fixed Q_pct=100 measurements.",
            "Q/rate scaling in this profile is therefore a linear approximation around the measured workload.",
        ],
        "loss_source": "measured spectral residual norm" if loss_table else "heuristic placeholder",
        "n_layers": n_layers,
        "cpus": cpus,
        "utilization": {str(cpu): util[cpu] for cpu in cpus},
        "q_pct": q_pct,
        "sorted_cpus_by_model_single_core_latency": sorted_cpus,
        "model_pred_full_base_ms": round(full_base_ms, 6),
        "full_local_ms": round(full_local_ms, 6),
        "baseline_no_svd_ms": round(full_local_ms, 6),
        "deadline_base_ms": round(deadline_base_ms, 6),
        "local_deadline_ms": round(local_deadline_ms, 6),
        "request_deadline_ms": round(request_deadline_ms, 6),
        "timeout_budget_ms": timeout_budget_ms,
        "scheduler_min_speedup_vs_baseline": scheduler_min_speedup_vs_baseline,
        "scheduler_max_speedup_vs_baseline": scheduler_max_speedup_vs_baseline,
        "quality_first_no_svd": quality_first_no_svd,
        "tx_ms_by_split_m": tx_by_split,
        "end_ms_by_split_m": end_by_split,
        "core_splits": core_splits,
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def run_scheduler(profile: Path, quantum_ms: float) -> dict[str, Any]:
    data = json.loads(profile.read_text())
    cmd = [
        "python3",
        str(EXP_DIR / "scheduler.py"),
        "--profile",
        str(profile),
        "--local-deadline-ms",
        str(data["local_deadline_ms"]),
        "--request-deadline-ms",
        str(data["request_deadline_ms"]),
        "--timeout-budget-ms",
        str(data.get("timeout_budget_ms", 0.0)),
        "--quantum-ms",
        str(quantum_ms),
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode not in (0, 2):
        return {"status": f"returncode_{completed.returncode}", "output": completed.stdout}
    try:
        return {"status": "ok", **json.loads(completed.stdout)}
    except json.JSONDecodeError:
        return {"status": "bad_json", "output": completed.stdout}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Build model-based scheduler profiles.")
    parser.add_argument("--model", type=Path, default=EXP_DIR / "results/latency_model_20260429_r1/best_model.pkl")
    parser.add_argument("--cpus", default="60-67")
    parser.add_argument("--loads", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--util-vector", help='Optional heterogeneous vector, e.g. "60:0,61:20,62:80"')
    parser.add_argument("--rates", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--q-pct", type=int, default=100)
    parser.add_argument("--timeout-budget-ms", type=float, default=8.0)
    parser.add_argument("--request-deadline-factor", type=float, default=1.04)
    parser.add_argument("--local-deadline-factor", type=float, default=1.01)
    parser.add_argument("--scheduler-min-speedup-vs-baseline", type=float, default=1.0)
    parser.add_argument("--scheduler-max-speedup-vs-baseline", type=float, default=0.0)
    parser.add_argument("--quality-first-no-svd", action="store_true")
    parser.add_argument("--reference-full-local-ms", type=float)
    parser.add_argument("--tx-base-ms", type=float, default=2.0)
    parser.add_argument("--tx-per-layer-ms", type=float, default=0.15)
    parser.add_argument("--end-per-layer-ms", type=float, default=1.0)
    parser.add_argument("--loss-table", type=Path, help="CSV from measure_svd_matrix_loss.py, normally svd_loss_for_scheduler.csv")
    parser.add_argument("--loss-reduction", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--quantum-ms", type=float, default=0.01)
    parser.add_argument("--run-scheduler", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/model_profiles_20260429_r1")
    args = parser.parse_args()

    cpus = parse_cpu_list(args.cpus)
    rates = parse_rates(args.rates)
    predictor = LatencyPredictor(args.model)
    loss_table = load_measured_loss_table(args.loss_table, args.loss_reduction)
    if sorted(cpus) != sorted(predictor.cpus):
        raise SystemExit(f"model CPUs {predictor.cpus} do not match requested CPUs {cpus}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.util_vector:
        scenarios = [("heterogeneous", parse_util_vector(cpus, args.util_vector))]
    else:
        scenarios = [(f"load_{load}", uniform_util(cpus, load)) for load in parse_loads(args.loads)]

    index_rows = []
    sanity_rows = []
    idle_reference = args.reference_full_local_ms
    if idle_reference is None:
        idle_util = uniform_util(cpus, 0)
        idle_reference = predictor.predict_ms(cpus, idle_util, args.q_pct)

    for name, util in scenarios:
        profile = build_profile(
            predictor=predictor,
            cpus=cpus,
            util=util,
            n_layers=args.n_layers,
            rates=rates,
            q_pct=args.q_pct,
            timeout_budget_ms=args.timeout_budget_ms,
            request_deadline_factor=args.request_deadline_factor,
            local_deadline_factor=args.local_deadline_factor,
            reference_full_local_ms=idle_reference,
            tx_base_ms=args.tx_base_ms,
            tx_per_layer_ms=args.tx_per_layer_ms,
            end_per_layer_ms=args.end_per_layer_ms,
            loss_table=loss_table,
            scheduler_min_speedup_vs_baseline=args.scheduler_min_speedup_vs_baseline,
            scheduler_max_speedup_vs_baseline=args.scheduler_max_speedup_vs_baseline,
            quality_first_no_svd=args.quality_first_no_svd,
        )
        profile_path = args.out_dir / f"profile_{name}.json"
        save_json(profile_path, profile)
        index_rows.append(
            {
                "scenario": name,
                "profile": str(profile_path),
                "full_local_ms": profile["full_local_ms"],
                "local_deadline_ms": profile["local_deadline_ms"],
                "request_deadline_ms": profile["request_deadline_ms"],
                "utilization": ",".join(f"{cpu}:{util[cpu]}" for cpu in cpus),
            }
        )
        if args.run_scheduler:
            result = run_scheduler(profile_path, args.quantum_ms)
            local = result.get("local") or {}
            decisions = local.get("decisions") or []
            rates_out = result.get("rates") or []
            sanity_rows.append(
                {
                    "scenario": name,
                    "status": result.get("status"),
                    "feasible": result.get("feasible"),
                    "mode": result.get("mode"),
                    "p": local.get("p"),
                    "major_cpus": cpu_spec(local.get("major_cpus") or []),
                    "minor_cpus": cpu_spec(local.get("minor_cpus") or []),
                    "split_m": result.get("split_m"),
                    "offloaded_layers": len(result.get("offloaded_layers") or []),
                    "clipped_layers": sum(1 for rate in rates_out if float(rate) > 0.0),
                    "max_rate": max([float(rate) for rate in rates_out], default=0.0),
                    "loss": local.get("total_loss"),
                    "main_ms": local.get("total_main_ms"),
                    "total_ms": result.get("total_ms"),
                    "full_local_ms": profile["full_local_ms"],
                    "estimated_speedup_vs_full_local": float(profile["full_local_ms"]) / float(result.get("total_ms") or math.inf),
                    "local_deadline_ms": profile["local_deadline_ms"],
                    "request_deadline_ms": profile["request_deadline_ms"],
                }
            )

    save_json(
        args.out_dir / "index.json",
        {
            "source_model": str(args.model),
            "q_pct": args.q_pct,
            "model_note": "Current model was trained at fixed Q; profile uses linear layer/rate scaling.",
            "loss_table": str(args.loss_table) if args.loss_table else None,
            "loss_source": "measured spectral residual norm" if loss_table else "heuristic placeholder",
            "profiles": index_rows,
        },
    )
    write_csv(
        args.out_dir / "profile_index.csv",
        index_rows,
        ["scenario", "profile", "full_local_ms", "local_deadline_ms", "request_deadline_ms", "utilization"],
    )
    if args.run_scheduler:
        write_csv(
            args.out_dir / "scheduler_sanity.csv",
            sanity_rows,
            [
                "scenario",
                "status",
                "feasible",
                "mode",
                "p",
                "major_cpus",
                "minor_cpus",
                "split_m",
                "offloaded_layers",
                "clipped_layers",
                "max_rate",
                "loss",
                "main_ms",
                "total_ms",
                "full_local_ms",
                "estimated_speedup_vs_full_local",
                "local_deadline_ms",
                "request_deadline_ms",
            ],
        )

    lines = [
        "# Model-Based Scheduler Profile Report",
        "",
        "This profile set connects the trained sklearn latency model to `scheduler.py` by generating the existing `core_splits` JSON schema.",
        "",
        "## Important Limitation",
        "",
        "The current trained model was fit from fixed-Q measurements. This script therefore uses the model for core-set/load effects and applies linear scaling for layer work and SVD rate. It is a useful scheduler integration, but not yet the final Q-aware Algorithmv2 5.2 profile.",
        "",
        "## Generated Profiles",
        "",
        f"- profiles: {len(index_rows)}",
        f"- n_layers: {args.n_layers}",
        f"- rates: {args.rates}",
        f"- timeout_budget_ms: {args.timeout_budget_ms}",
        f"- idle reference deadline base: {idle_reference:.6f} ms",
        f"- loss source: {'measured spectral residual norm' if loss_table else 'heuristic placeholder'}",
    ]
    if args.run_scheduler:
        lines.extend(
            [
                "",
                "## Scheduler Sanity",
                "",
                "| scenario | mode | p | clipped | max rate | loss | total ms | full local ms | est. speedup |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sanity_rows:
            lines.append(
                f"| {row['scenario']} | {row['mode']} | {row['p']} | {row['clipped_layers']} | "
                f"{float(row['max_rate']):.3f} | {float(row['loss'] or 0):.6f} | "
                f"{float(row['total_ms'] or 0):.6f} | {float(row['full_local_ms']):.6f} | "
                f"{float(row['estimated_speedup_vs_full_local']):.3f}x |"
            )
    (args.out_dir / "MODEL_PROFILE_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "profiles": len(index_rows), "sanity_rows": len(sanity_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
