#!/usr/bin/env python3
"""Small local experiment driver for exp12.

This script does not use adb.  It reuses the existing SVD decode binary and
can optionally keep the process inside a cgroup v2 cpuset when sudo is
available.  The "phone" side is emulated by the timing model in the generated
profile so the scheduling algorithm can be exercised without a real device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from loss_table import load_measured_loss_table, measured_or_heuristic_loss
from scheduler import (
    result_to_json,
    save_json,
    solve_joint_schedule,
    write_rate_file,
    write_timeout_file,
)


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf"
DEFAULT_BINARY = ROOT / "build-release-current/decode_svd_test"


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
    return ",".join(str(cpu) for cpu in cpus)


def make_cgroup(name: str, cpus: list[int]) -> Path | None:
    cg = Path("/sys/fs/cgroup") / name
    try:
        subprocess.run(["sudo", "-n", "mkdir", "-p", str(cg)], check=True)
        subprocess.run(["sudo", "-n", "sh", "-c", f"echo +cpuset > /sys/fs/cgroup/cgroup.subtree_control || true"], check=True)
        subprocess.run(["sudo", "-n", "sh", "-c", f"echo 0 > {cg}/cpuset.mems"], check=True)
        subprocess.run(["sudo", "-n", "sh", "-c", f"echo {cpu_spec(cpus)} > {cg}/cpuset.cpus"], check=True)
        return cg
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_in_cgroup(cmd: list[str], cgroup: Path | None, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    if cgroup is None:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            errors="replace",
        )
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
    wrapped = [
        "sudo",
        "-n",
        "env",
        f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}",
        "bash",
        "-lc",
        script,
    ]
    try:
        return subprocess.run(
            wrapped,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            errors="replace",
        )
    except FileNotFoundError:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            errors="replace",
        )


def start_busy_load(cpus: list[int], count: int) -> list[subprocess.Popen[str]]:
    procs = []
    for cpu in cpus[:count]:
        procs.append(
            subprocess.Popen(
                ["taskset", "-c", str(cpu), "bash", "-lc", "while :; do :; done"],
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    time.sleep(0.5)
    return procs


def stop_busy_load(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def parse_decode_output(output: str) -> dict[str, Any]:
    def f(pattern: str) -> float | None:
        m = re.search(pattern, output)
        return float(m.group(1)) if m else None

    def i(pattern: str) -> int | None:
        m = re.search(pattern, output)
        return int(m.group(1)) if m else None

    top1 = re.search(
        r"Top-5 next-token candidates:\s*\n\s*id=(\d+)\s+logit=([-+0-9.eE]+)\s+piece=\"([^\"]*)\"",
        output,
    )

    return {
        "n_layer": i(r"n_layer:(\d+)"),
        "decode_tok_s": f(r"Decode-only throughput:\s*([0-9.]+)"),
        "generation_decode_ms": f(r"generation_decode=([0-9.]+) ms"),
        "prefill_decode_ms": f(r"prefill_decode=([0-9.]+) ms"),
        "generated_text": (re.search(r"Generated text:\s*(.*)", output) or [None, None])[1],
        "top1_id": int(top1.group(1)) if top1 else None,
        "top1_logit": float(top1.group(2)) if top1 else None,
        "top1_piece": top1.group(3) if top1 else None,
        "succeed": "SUCCEED" in output,
    }


def run_decode(
    binary: Path,
    model: Path,
    cpus: list[int],
    tokens: int,
    rates: Path | str = "off",
    group_a: str = "off",
    group_b: str = "off",
    group_a_share: float = 0.5,
    minor_timeout_ms: int = 0,
    svd_offload_timeout_ms: int = 2,
    layer_timeouts: Path | str = "off",
    cgroup: Path | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / "build-release-current/bin")
    cmd = [
        "taskset",
        "-c",
        cpu_spec(cpus),
        str(binary),
        str(model),
        str(tokens),
        str(len(cpus)),
        "0",
        "off",
        str(rates),
        group_a,
        group_b,
        str(group_a_share),
        str(minor_timeout_ms),
        str(svd_offload_timeout_ms),
        str(layer_timeouts),
    ]
    completed = run_in_cgroup(cmd, cgroup, env)
    parsed = parse_decode_output(completed.stdout)
    parsed.update(
        {
            "returncode": completed.returncode,
            "cmd": cmd,
            "output_tail": "\n".join(completed.stdout.splitlines()[-28:]),
        }
    )
    return parsed


def make_profile(
    n_layers: int,
    per_layer_full_ms: float,
    load_scale: float,
    rates: list[float],
    local_deadline_ms: float,
    request_deadline_ms: float,
    loss_table: dict[tuple[int, str], float] | None = None,
) -> dict[str, Any]:
    layers = []
    for layer in range(n_layers):
        layer_shape = 1.0 + 0.08 * ((layer % 4) - 1.5)
        candidates = []
        for rate in rates:
            # Main path becomes faster as more SVD rank is truncated, but the
            # speedup is deliberately conservative to match the exp10 findings.
            main_ms = per_layer_full_ms * load_scale * layer_shape * (1.0 - 0.42 * rate)
            loss = measured_or_heuristic_loss(loss_table or {}, layer, rate)
            weight = rate * layer_shape
            candidates.append(
                {
                    "rate": round(rate, 4),
                    "main_ms": round(main_ms, 4),
                    "loss": round(loss, 6),
                    "weight": round(weight, 6),
                }
            )
        layers.append({"layer": layer, "candidates": candidates})

    tx_by_split = []
    end_by_split = []
    for split_m in range(n_layers + 1):
        offloaded = max(0, n_layers - split_m - 1)
        tx_by_split.append(round(3.0 + 0.09 * offloaded, 4))
        end_by_split.append(round(per_layer_full_ms * 0.62 * offloaded, 4))

    return {
        "source": "generated from local decode measurement; phone side is CPU-emulated timing, no adb",
        "n_layers": n_layers,
        "per_layer_full_ms": per_layer_full_ms,
        "load_scale": load_scale,
        "local_deadline_ms": local_deadline_ms,
        "request_deadline_ms": request_deadline_ms,
        "loss_source": "measured spectral residual norm" if loss_table else "heuristic placeholder",
        "tx_ms_by_split_m": tx_by_split,
        "end_ms_by_split_m": end_by_split,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small exp12 local scheduling experiment.")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--run-cpus", default="60-67")
    parser.add_argument("--phone-cpus", default="68-75")
    parser.add_argument("--load-cpus", default="60-63")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--load-scale", type=float, default=3.4)
    parser.add_argument("--local-deadline-factor", type=float, default=1.08)
    parser.add_argument("--request-deadline-factor", type=float, default=1.75)
    parser.add_argument("--timeout-budget-ms", type=float, default=8.0)
    parser.add_argument("--quantum-ms", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR / "results/latest")
    parser.add_argument("--loss-table", type=Path, help="CSV from measure_svd_matrix_loss.py, normally svd_loss_for_scheduler.csv")
    parser.add_argument("--loss-reduction", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--use-cgroup", action="store_true")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args()

    run_cpus = parse_cpu_list(args.run_cpus)
    phone_cpus = parse_cpu_list(args.phone_cpus)
    load_cpus = parse_cpu_list(args.load_cpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loss_table = load_measured_loss_table(args.loss_table, args.loss_reduction)

    cgroup = make_cgroup("exp12_algorithms_run", run_cpus) if args.use_cgroup else None
    baseline = {"n_layer": 28, "generation_decode_ms": 32.0, "decode_tok_s": 31.25, "succeed": True}
    loaded = {}

    if not args.skip_decode:
        baseline = run_decode(args.binary, args.model, run_cpus, args.tokens, cgroup=cgroup)
        busy = start_busy_load(load_cpus, args.load_workers)
        try:
            loaded = run_decode(args.binary, args.model, run_cpus, args.tokens, cgroup=cgroup)
        finally:
            stop_busy_load(busy)

    n_layers = int(baseline.get("n_layer") or 28)
    gen_decode_ms = float(baseline.get("generation_decode_ms") or 32.0)
    per_layer_ms = max(0.01, gen_decode_ms / max(1, args.tokens) / n_layers)
    local_deadline = gen_decode_ms * args.local_deadline_factor
    request_deadline = gen_decode_ms * args.request_deadline_factor

    profile = make_profile(
        n_layers=n_layers,
        per_layer_full_ms=per_layer_ms,
        load_scale=args.load_scale,
        rates=[0.0, 0.25, 0.5, 0.75],
        local_deadline_ms=local_deadline,
        request_deadline_ms=request_deadline,
        loss_table=loss_table,
    )
    profile_path = args.out_dir / "profile.json"
    save_json(profile_path, profile)

    from scheduler import load_profile

    layers, meta = load_profile(profile_path)
    result = solve_joint_schedule(
        layers,
        local_deadline_ms=local_deadline,
        request_deadline_ms=request_deadline,
        timeout_budget_ms=args.timeout_budget_ms,
        quantum_ms=args.quantum_ms,
        meta=meta,
    )
    result_obj = result_to_json(result, n_layers)
    result_path = args.out_dir / "schedule.json"
    rates_path = args.out_dir / "rates.txt"
    timeouts_path = args.out_dir / "timeouts.txt"
    save_json(result_path, result_obj)
    if result.local and result.local.decisions:
        write_rate_file(rates_path, result.local.rates(n_layers))
        write_timeout_file(timeouts_path, result.local.timeouts(n_layers))

    scheduled = {}
    if not args.skip_decode and rates_path.exists():
        group_a = cpu_spec(run_cpus[: len(run_cpus) // 2])
        group_b = cpu_spec(run_cpus[len(run_cpus) // 2 :])
        scheduled = run_decode(
            args.binary,
            args.model,
            run_cpus,
            args.tokens,
            rates=rates_path,
            group_a=group_a,
            group_b=group_b,
            group_a_share=0.5,
            layer_timeouts=timeouts_path,
            cgroup=cgroup,
        )

    experiment = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_cpus": run_cpus,
        "phone_cpus_emulated": phone_cpus,
        "load_cpus": load_cpus,
        "cgroup": str(cgroup) if cgroup else None,
        "baseline": baseline,
        "loaded": loaded,
        "schedule": result_obj,
        "scheduled_decode": scheduled,
        "profile": str(profile_path),
        "rates": str(rates_path),
        "timeouts": str(timeouts_path),
    }
    save_json(args.out_dir / "experiment.json", experiment)
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
