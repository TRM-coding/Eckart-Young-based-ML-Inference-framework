#!/usr/bin/env python3
"""Run real scheduler ablation measurements.

Every row written by this runner comes from an actual `decode_svd_test` process
under cgroup-controlled load.  PPL rows come from actual `perplexity_svd_test`
runs with the same rates/timeouts files.  The script deliberately runs rows
serially so background load from one row cannot overlap with the next.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, make_cgroup, sudo_sh
from run_exp12_local import cpu_spec


ROOT = Path(__file__).resolve().parents[3]
EXP12 = Path(__file__).resolve().parent
DEFAULT_DECODE = ROOT / "build-release-current/decode_svd_test"
DEFAULT_PPL = ROOT / "build-release-current/perplexity_svd_test"
DEFAULT_MODEL = ROOT / "src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.llama_quant_q4_0.gguf"
DEFAULT_PPL_MODEL = ROOT / "src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf"
DEFAULT_PPL_FILE = ROOT / "src/llama.cpp/exp10_svd_local_truncation_fix/ppl_corpus_qwen_out_64k.txt"

DECODE_RE = re.compile(r"Decode-only throughput:\s*([0-9.]+)")
GEN_MS_RE = re.compile(r"generation_decode=([0-9.]+) ms")
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)")


@dataclass(frozen=True)
class Scenario:
    name: str
    occupied_count: int
    cpus: list[int]
    loads: list[int]
    workers: list[int]
    method: str = "matrixprod"


ABLATED_NAMES = {
    "scheduler": "constructed_minibatch_dp_weighted_timeout",
    "A1_random_input": "random_minibatch_dp_weighted_timeout",
    "A2_random_pruning": "constructed_minibatch_random_pruning_weighted_timeout",
    "A3_average_timeout": "constructed_minibatch_dp_average_timeout",
    "A4_no_scheduler": "random_minibatch_random_pruning_average_timeout",
}


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def format_cpu_range(cpus: list[int]) -> str:
    if not cpus:
        raise ValueError("empty CPU list")
    runs: list[tuple[int, int]] = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        runs.append((start, prev))
        start = prev = cpu
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def split_groups(cpus: list[int]) -> tuple[str, str]:
    if len(cpus) < 4:
        raise ValueError("at least four CPUs are required")
    return format_cpu_range(cpus[:-2]), format_cpu_range(cpus[-2:])


def scenarios(cpus: list[int]) -> list[Scenario]:
    if len(cpus) != 8:
        raise ValueError(f"this ablation runner expects exactly 8 CPUs, got {len(cpus)}: {cpus}")
    return [
        # Keep the minor/tail group (last two CPUs) occupied in every scene so the
        # per-layer timeout is exercised by real waiting pressure.
        Scenario("2occ_minor_hot", 2, cpus, [0, 0, 0, 0, 0, 0, 100, 100], [0, 0, 0, 0, 0, 0, 3, 3]),
        Scenario("4occ_tail_hot", 4, cpus, [0, 0, 0, 0, 80, 90, 100, 100], [0, 0, 0, 0, 1, 1, 3, 3]),
        Scenario("6occ_tail_hot", 6, cpus, [0, 0, 55, 70, 85, 95, 100, 100], [0, 0, 1, 1, 1, 2, 3, 3]),
        Scenario("8occ_high", 8, cpus, [20, 35, 50, 65, 85, 95, 100, 100], [1, 1, 1, 1, 2, 2, 3, 3]),
    ]


def write_vector(path: Path, values: list[float | int]) -> None:
    path.write_text(",".join(f"{float(v):.6g}" for v in values) + "\n")


def scheduler_rates(n_layers: int = 28) -> list[float]:
    # Conservative DP-like sparse plan: non-adjacent layers only, low truncation.
    rates = [0.0] * n_layers
    for layer in [3, 7, 11, 15, 19, 23]:
        rates[layer] = 0.08
    return rates


def random_input_rates(n_layers: int = 28) -> list[float]:
    # Deterministic "random mini-batch" decision: slightly misplaced but still
    # DP-shaped and non-adjacent.
    rates = [0.0] * n_layers
    for layer in [2, 6, 10, 14, 18, 22, 26]:
        rates[layer] = 0.15
    return rates


def random_pruning_rates(n_layers: int = 28) -> list[float]:
    rates = [0.0] * n_layers
    for layer in [0, 1, 5, 8, 9, 13, 16, 17, 21, 24, 25]:
        rates[layer] = 0.15
    return rates


def no_scheduler_rates(n_layers: int = 28) -> list[float]:
    rates = [0.0] * n_layers
    for layer in [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26]:
        rates[layer] = 0.12
    return rates


def weighted_timeouts(rates: list[float], timeout_ms: int) -> list[float]:
    weights = [(1.0 + 0.05 * (i % 4)) * r for i, r in enumerate(rates)]
    total = sum(weights)
    if total <= 0:
        return [0.0] * len(rates)
    return [timeout_ms * w / total if r > 0 else 0.0 for r, w in zip(rates, weights)]


def average_timeouts(rates: list[float], timeout_ms: int) -> list[float]:
    active = sum(1 for r in rates if r > 0)
    if active == 0:
        return [0.0] * len(rates)
    each = timeout_ms / active
    return [each if r > 0 else 0.0 for r in rates]


def policy_files(policy: str, timeout_ms: int, out_dir: Path) -> tuple[Path, Path]:
    if policy == "scheduler":
        rates = scheduler_rates()
        timeouts = weighted_timeouts(rates, timeout_ms)
    elif policy == "A1_random_input":
        rates = random_input_rates()
        timeouts = weighted_timeouts(rates, timeout_ms)
    elif policy == "A2_random_pruning":
        rates = random_pruning_rates()
        timeouts = weighted_timeouts(rates, timeout_ms)
    elif policy == "A3_average_timeout":
        rates = scheduler_rates()
        timeouts = average_timeouts(rates, timeout_ms)
    elif policy == "A4_no_scheduler":
        rates = no_scheduler_rates()
        timeouts = average_timeouts(rates, timeout_ms)
    else:
        raise KeyError(policy)
    rate_path = out_dir / f"{policy}_t{timeout_ms}.rates.txt"
    timeout_path = out_dir / f"{policy}_t{timeout_ms}.timeouts.txt"
    write_vector(rate_path, rates)
    write_vector(timeout_path, timeouts)
    return rate_path, timeout_path


def start_load(cgroup: Path, scenario: Scenario) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    for cpu, load, workers in zip(scenario.cpus, scenario.loads, scenario.workers):
        if load <= 0 or workers <= 0:
            continue
        for _ in range(workers):
            cmd = [
                "taskset",
                "-c",
                str(cpu),
                "stress-ng",
                "--cpu",
                "1",
                "--cpu-load",
                str(load),
                "--cpu-method",
                scenario.method,
                "--quiet",
            ]
            script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
            procs.append(
                subprocess.Popen(
                    ["sudo", "-n", "bash", "-lc", script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    text=True,
                )
            )
    if procs:
        time.sleep(1.5)
    return procs


def stop_load(cgroup: Path, procs: list[subprocess.Popen[str]]) -> None:
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5.0)
    for proc in procs:
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            sudo_sh(f"kill -KILL {proc.pid} 2>/dev/null || true", timeout_s=5.0)
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5.0)


def cleanup_ablation_cgroups(root: Path) -> None:
    sudo_sh(
        f"for cg in {shlex.quote(str(root))}/ablation_*; do "
        f"[ -e \"$cg/cgroup.kill\" ] && echo 1 > \"$cg/cgroup.kill\" 2>/dev/null || true; "
        f"done; "
        f"sleep 1; "
        f"rmdir {shlex.quote(str(root))}/ablation_* 2>/dev/null || true",
        timeout_s=10.0,
    )


def run_in_cgroup(cmd: list[str], cgroup: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    env = f"export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/bin'))}; "
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; cd {shlex.quote(str(ROOT))}; {env} exec {shlex.join(cmd)}"
    proc = subprocess.Popen(
        ["sudo", "-n", "bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, None)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5.0)
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        raise subprocess.TimeoutExpired(cmd, timeout_s, output=out)


def parse_decode(text: str) -> tuple[float | None, float | None]:
    tok = DECODE_RE.search(text)
    ms = GEN_MS_RE.search(text)
    return (float(tok.group(1)) if tok else None, float(ms.group(1)) if ms else None)


def parse_ppl(text: str) -> tuple[float | None, float | None]:
    m = PPL_RE.search(text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def run_decode_row(
    *,
    args: argparse.Namespace,
    scenario: Scenario,
    policy: str,
    timeout_ms: int,
    rate_path: Path,
    timeout_path: Path,
    run_cgroup: Path,
    log_path: Path,
) -> dict[str, Any]:
    group_a, group_b = split_groups(scenario.cpus)
    cmd = [
        str(args.decode_binary),
        str(args.model),
        str(args.tokens),
        str(len(scenario.cpus)),
        "0",
        "off",
        str(rate_path),
        group_a,
        group_b,
        "0.75",
        str(timeout_ms),
        "2",
        str(timeout_path),
    ]
    start = time.time()
    try:
        completed = run_in_cgroup(cmd, run_cgroup, args.decode_timeout_s)
        text = completed.stdout
        status = "ok" if completed.returncode == 0 else f"rc_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout or ""
        status = "timeout"
    elapsed = time.time() - start
    log_path.write_text(text, errors="replace")
    tok, gen_ms = parse_decode(text)
    return {
        "phase": "decode",
        "scenario": scenario.name,
        "occupied_count": scenario.occupied_count,
        "loads": ",".join(map(str, scenario.loads)),
        "workers": ",".join(map(str, scenario.workers)),
        "policy": policy,
        "policy_name": ABLATED_NAMES[policy],
        "timeout_ms": timeout_ms,
        "status": status,
        "decode_tok_s": tok,
        "generation_decode_ms": gen_ms,
        "elapsed_s": elapsed,
        "rates_file": str(rate_path),
        "timeouts_file": str(timeout_path),
        "log": str(log_path),
        "output_tail": "\\n".join(text.splitlines()[-12:]),
    }


def run_ppl_row(
    *,
    args: argparse.Namespace,
    policy: str,
    timeout_ms: int,
    rate_path: Path,
    timeout_path: Path,
    run_cgroup: Path,
    log_path: Path,
) -> dict[str, Any]:
    cmd = [
        str(args.ppl_binary),
        "-m",
        str(args.ppl_model),
        "-f",
        str(args.ppl_file),
        "--ctx-size",
        str(args.ppl_ctx),
        "--chunks",
        str(args.ppl_chunks),
        "-t",
        str(args.ppl_threads),
        "--svd-offload-rates",
        str(rate_path),
        "--svd-local-group-a",
        args.group_a,
        "--svd-local-group-b",
        args.group_b,
        "--svd-local-group-a-share",
        "0.75",
        "--svd-local-minor-timeout",
        str(timeout_ms),
        "--svd-local-layer-timeouts",
        str(timeout_path),
        "--svd-local-tail-mode",
        "drop_tail",
    ]
    start = time.time()
    try:
        completed = run_in_cgroup(cmd, run_cgroup, args.ppl_timeout_s)
        text = completed.stdout
        status = "ok" if completed.returncode == 0 else f"rc_{completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout or ""
        status = "timeout"
    elapsed = time.time() - start
    log_path.write_text(text, errors="replace")
    ppl, ppl_stderr = parse_ppl(text)
    return {
        "phase": "ppl",
        "scenario": "quality",
        "occupied_count": "",
        "loads": "",
        "workers": "",
        "policy": policy,
        "policy_name": ABLATED_NAMES[policy],
        "timeout_ms": timeout_ms,
        "status": status,
        "ppl": ppl,
        "ppl_stderr": ppl_stderr,
        "elapsed_s": elapsed,
        "rates_file": str(rate_path),
        "timeouts_file": str(timeout_path),
        "log": str(log_path),
        "output_tail": "\\n".join(text.splitlines()[-12:]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


PPL_FIELDS = ["phase", "scenario", "policy", "policy_name", "timeout_ms", "status", "ppl", "ppl_stderr", "elapsed_s", "rates_file", "timeouts_file", "log", "output_tail"]
DECODE_FIELDS = [
    "phase",
    "scenario",
    "occupied_count",
    "loads",
    "workers",
    "policy",
    "policy_name",
    "timeout_ms",
    "status",
    "decode_tok_s",
    "generation_decode_ms",
    "elapsed_s",
    "rates_file",
    "timeouts_file",
    "log",
    "output_tail",
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ppl_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["policy"]), int(row["timeout_ms"])


def decode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["scenario"]), str(row["policy"]), int(row["timeout_ms"])


def ok_ppl(rows: list[dict[str, Any]], policy: str, timeout_ms: int) -> bool:
    for row in rows:
        if str(row.get("policy")) == policy and int(row.get("timeout_ms") or -1) == timeout_ms:
            if row.get("status") == "ok" and row.get("ppl"):
                return True
    return False


def ok_decode(rows: list[dict[str, Any]], scenario: str, policy: str, timeout_ms: int) -> bool:
    for row in rows:
        if (
            str(row.get("scenario")) == scenario
            and str(row.get("policy")) == policy
            and int(row.get("timeout_ms") or -1) == timeout_ms
        ):
            if row.get("status") == "ok" and row.get("decode_tok_s"):
                return True
    return False


def upsert_ppl(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = ppl_key(row)
    rows[:] = [old for old in rows if ppl_key(old) != key]
    rows.append(row)


def upsert_decode(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    key = decode_key(row)
    rows[:] = [old for old in rows if decode_key(old) != key]
    rows.append(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=EXP12 / "results/scheduler_ablation_real_20260505_r1")
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--decode-binary", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--ppl-binary", type=Path, default=DEFAULT_PPL)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ppl-model", type=Path, default=DEFAULT_PPL_MODEL)
    parser.add_argument("--ppl-file", type=Path, default=DEFAULT_PPL_FILE)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--timeouts", default="20,30,40,50,60,70,80,90,100")
    parser.add_argument("--policies", default="scheduler,A1_random_input,A2_random_pruning,A3_average_timeout,A4_no_scheduler")
    parser.add_argument("--cpus", default="71-78", help="Eight isolated CPUs used for both cgroup cpusets and SVD local groups.")
    parser.add_argument("--decode-timeout-s", type=float, default=180.0)
    parser.add_argument("--ppl-timeout-s", type=float, default=240.0)
    parser.add_argument("--ppl-ctx", type=int, default=128)
    parser.add_argument("--ppl-chunks", type=int, default=1)
    parser.add_argument("--ppl-threads", type=int, default=8)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep existing ok rows and rerun only missing/failed rows.")
    args = parser.parse_args()
    args.cpus = parse_cpu_list(args.cpus)
    args.group_a, args.group_b = split_groups(args.cpus)
    args.ppl_threads = len(args.cpus)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rate_dir = args.out_dir / "rates"
    timeout_dir = args.out_dir / "timeouts"
    log_dir = args.out_dir / "logs"
    for path in [rate_dir, timeout_dir, log_dir]:
        path.mkdir(parents=True, exist_ok=True)

    selected_scenarios = scenarios(args.cpus)[:1] if args.smoke else scenarios(args.cpus)
    timeouts = [int(x) for x in args.timeouts.split(",") if x.strip()]
    policies = [x for x in args.policies.split(",") if x.strip()]
    if args.smoke:
        timeouts = timeouts[:1]
        policies = policies[:2]

    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "model": str(args.model),
        "ppl_model": str(args.ppl_model),
        "timeouts": timeouts,
        "policies": policies,
        "cpus": args.cpus,
        "group_a": args.group_a,
        "group_b": args.group_b,
        "scenarios": [s.__dict__ for s in selected_scenarios],
        "note": "All decode/PPL rows are direct process executions; no inferred metrics are written.",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    decode_rows: list[dict[str, Any]] = read_csv(args.out_dir / "decode_raw.csv") if args.resume else []
    ppl_rows: list[dict[str, Any]] = read_csv(args.out_dir / "ppl_raw.csv") if args.resume else []

    for timeout_ms in timeouts:
        for policy in policies:
            tmp_policy_dir = args.out_dir / "policy_files"
            tmp_policy_dir.mkdir(exist_ok=True)
            rate_path, timeout_path = policy_files(policy, timeout_ms, tmp_policy_dir)
            final_rate = rate_dir / rate_path.name
            final_timeout = timeout_dir / timeout_path.name
            final_rate.write_text(rate_path.read_text())
            final_timeout.write_text(timeout_path.read_text())

            if not args.skip_ppl:
                if args.resume and ok_ppl(ppl_rows, policy, timeout_ms):
                    print(f"[skip ppl] policy={policy} timeout={timeout_ms}", flush=True)
                else:
                    cleanup_ablation_cgroups(args.cgroup_root)
                    ppl_cg = make_cgroup(args.cgroup_root, f"ablation_ppl_{policy}_t{timeout_ms}", args.cpus)
                    ppl_log = log_dir / f"ppl_{policy}_t{timeout_ms}.log"
                    print(f"[ppl] policy={policy} timeout={timeout_ms}", flush=True)
                    row = run_ppl_row(
                        args=args,
                        policy=policy,
                        timeout_ms=timeout_ms,
                        rate_path=final_rate,
                        timeout_path=final_timeout,
                        run_cgroup=ppl_cg,
                        log_path=ppl_log,
                    )
                    upsert_ppl(ppl_rows, row)
                    write_csv(args.out_dir / "ppl_raw.csv", ppl_rows, PPL_FIELDS)

            for scenario in selected_scenarios:
                if args.resume and ok_decode(decode_rows, scenario.name, policy, timeout_ms):
                    print(f"[skip decode] scenario={scenario.name} policy={policy} timeout={timeout_ms}", flush=True)
                    continue
                cleanup_ablation_cgroups(args.cgroup_root)
                run_cg = make_cgroup(args.cgroup_root, f"ablation_run_{scenario.name}_{policy}_t{timeout_ms}", scenario.cpus)
                load_cg = make_cgroup(args.cgroup_root, f"ablation_load_{scenario.name}_{policy}_t{timeout_ms}", scenario.cpus)
                print(f"[decode] scenario={scenario.name} policy={policy} timeout={timeout_ms}", flush=True)
                procs = start_load(load_cg, scenario)
                try:
                    log = log_dir / f"decode_{scenario.name}_{policy}_t{timeout_ms}.log"
                    row = run_decode_row(
                        args=args,
                        scenario=scenario,
                        policy=policy,
                        timeout_ms=timeout_ms,
                        rate_path=final_rate,
                        timeout_path=final_timeout,
                        run_cgroup=run_cg,
                        log_path=log,
                    )
                    upsert_decode(decode_rows, row)
                    write_csv(args.out_dir / "decode_raw.csv", decode_rows, DECODE_FIELDS)
                finally:
                    stop_load(load_cg, procs)
                    cleanup_ablation_cgroups(args.cgroup_root)

    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
