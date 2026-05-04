#!/usr/bin/env python3
"""Q4 lossless scheduler benchmark with local and phone-offload candidates.

This benchmark intentionally uses the original Q4 GGUF for every no-SVD path:

  - baseline_no_svd: all PC cores
  - major_only_no_svd: low-load PC core prefixes
  - edge_end_no_svd: layer-level PC/Android coop

SVD candidates should be benchmarked separately with an SVD-capable GGUF and
then merged only when the selected strategy actually uses SVD.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_heterogeneous_schedule_cgroup import start_heterogeneous_load, stop_heterogeneous_load
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, ROOT, decode_once, make_cgroup
from run_exp12_local import cpu_spec


EXP12 = Path(__file__).resolve().parent
EXP6 = ROOT / "src/llama.cpp/exp6_decode_svd_model"
DEFAULT_Q4 = ROOT / "src/llama.cpp/gguf_models/qwen.q4_0.gguf"
DEFAULT_DECODE = ROOT / "build-release-current/decode_svd_test"
DEFAULT_LAYER_CLIENT = ROOT / "build-release-current/layer_coop_client"


STEADY_RE = __import__("re").compile(
    r"\[layer-coop-steady\]\s+generated=(?P<generated>\d+)\s+"
    r"steady_decode_ms=(?P<steady_decode_ms>[0-9.eE+-]+)\s+"
    r"steady_throughput=(?P<steady_throughput>[0-9.eE+-]+)\s+tok/s\s+"
    r"steady_ms_per_token=(?P<steady_ms_per_token>[0-9.eE+-]+)\s+"
    r"prefix_decode_ms=(?P<prefix_decode_ms>[0-9.eE+-]+)\s+"
    r"server_decode_ms=(?P<server_decode_ms>[0-9.eE+-]+)\s+"
    r"network_wait_ms=(?P<network_wait_ms>[0-9.eE+-]+)"
)


ADB_SERIAL = ""


def adb(args: list[str], timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    serial_args = ["-s", ADB_SERIAL] if ADB_SERIAL else []
    return subprocess.run(
        ["adb", "-P", "5038", *serial_args, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        errors="replace",
    )


def stop_phone_server() -> None:
    try:
        adb(["shell", "pkill -f layer_mobile_server || true"], timeout_s=5.0)
    except subprocess.TimeoutExpired:
        pass


def parse_cpu_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def scenario_defaults() -> list[dict[str, Any]]:
    return [
        {"scenario": "2c_balanced_mid", "cpus": [60, 61], "loads": [50, 50]},
        {"scenario": "2c_asym_low_high", "cpus": [60, 61], "loads": [20, 80]},
        {"scenario": "2c_idle_saturated", "cpus": [60, 61], "loads": [0, 100]},
        {"scenario": "2c_high", "cpus": [60, 61], "loads": [80, 100]},
        {"scenario": "4c_balanced_high", "cpus": [60, 61, 62, 63], "loads": [20, 40, 60, 80]},
        {"scenario": "4c_front_hot", "cpus": [60, 61, 62, 63], "loads": [90, 70, 20, 0]},
        {"scenario": "4c_gradient", "cpus": [60, 61, 62, 63], "loads": [0, 20, 60, 100]},
        {"scenario": "4c_high", "cpus": [60, 61, 62, 63], "loads": [70, 80, 90, 100]},
        {"scenario": "8c_high", "cpus": [60, 61, 62, 63, 64, 65, 66, 67], "loads": [30, 40, 50, 60, 70, 80, 90, 100]},
    ]


def parse_steady(text: str) -> dict[str, Any]:
    matches = list(STEADY_RE.finditer(text))
    if not matches:
        return {}
    obj: dict[str, Any] = matches[-1].groupdict()
    for key in list(obj):
        obj[key] = float(obj[key]) if key != "generated" else int(obj[key])
    return obj


def run_layer_coop(
    *,
    client_binary: Path,
    pc_model: Path,
    android_dir: str,
    android_model: str,
    pc_host: str,
    run_cgroup: Path,
    pc_cpus: list[int],
    split_m: int,
    tokens: int,
    threads: int,
    port: int,
    out_dir: Path,
    label: str,
    timeout_s: float,
) -> dict[str, Any]:
    client_log = out_dir / f"{label}_client.log"
    server_log = out_dir / f"{label}_server.log"
    stop_phone_server()

    client_cmd = [
        str(client_binary),
        str(pc_model),
        str(tokens),
        str(threads),
        f"listen:{port}",
        str(split_m),
        "0",
    ]
    client_script = (
        f"echo $$ > {shlex.quote(str(run_cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; exec {shlex.join(client_cmd)}"
    )
    client_proc = subprocess.Popen(
        ["sudo", "-n", "bash", "-lc", client_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    time.sleep(1.0)

    server_cmd = (
        f"cd {shlex.quote(android_dir)} && "
        f"LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff LAYER_COOP_KEEP_HOT=1 "
        f"taskset -a ff ./layer_mobile_server {shlex.quote(android_model)} 0 {split_m} {threads} "
        f"{pc_host}:{port}"
    )
    server_proc = subprocess.Popen(
        ["adb", "-P", "5038", *(["-s", ADB_SERIAL] if ADB_SERIAL else []), "shell", server_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    status = "ok"
    started = time.time()
    try:
        client_out, _ = client_proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        status = "client_timeout"
        try:
            os.killpg(client_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            client_out, _ = client_proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            client_out = ""
    finally:
        stop_phone_server()
        try:
            server_out, _ = server_proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            server_out, _ = server_proc.communicate(timeout=5)

    client_log.write_text(client_out, errors="replace")
    server_log.write_text(server_out, errors="replace")
    steady = parse_steady(client_out)
    if status == "ok" and client_proc.returncode not in (0, None):
        status = f"client_rc_{client_proc.returncode}"
    if status == "ok" and not steady:
        status = "missing_steady"

    return {
        "status": status,
        "decode_tok_s": steady.get("steady_throughput"),
        "generation_decode_ms": steady.get("steady_ms_per_token"),
        "steady_decode_ms": steady.get("steady_decode_ms"),
        "prefix_decode_ms": steady.get("prefix_decode_ms"),
        "server_decode_ms": steady.get("server_decode_ms"),
        "network_wait_ms": steady.get("network_wait_ms"),
        "generated": steady.get("generated"),
        "elapsed_s": time.time() - started,
        "client_log": str(client_log.relative_to(out_dir)),
        "server_log": str(server_log.relative_to(out_dir)),
    }


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def selected_major_prefixes(cpus: list[int], loads: list[int], max_candidates: int) -> list[tuple[int, list[int]]]:
    sorted_pairs = sorted(zip(cpus, loads), key=lambda item: (item[1], item[0]))
    ordered_cpus = [cpu for cpu, _ in sorted_pairs]
    candidates = list(range(1, len(ordered_cpus)))
    if max_candidates > 0 and len(candidates) > max_candidates:
        keep = sorted(set([candidates[-1], candidates[len(candidates) // 2], candidates[max(0, len(candidates) - 2)]]))
        candidates = keep[-max_candidates:]
    return [(p, ordered_cpus[:p]) for p in candidates]


def choose_split_candidates(cpus: list[int], loads: list[int], override: list[int] | None = None) -> list[int]:
    if override:
        return sorted(set(split for split in override if 0 < split <= 28))
    avg = sum(loads) / len(loads)
    if len(cpus) <= 2:
        return [2, 4, 6, 8, 12, 16, 20, 24]
    if avg >= 70:
        return [2, 4, 6, 8, 12]
    if avg >= 45:
        return [4, 8, 12, 16, 20]
    return [8, 12, 16, 20]


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scenario in sorted({str(row["scenario"]) for row in raw_rows}):
        rows = [row for row in raw_rows if row["scenario"] == scenario and row.get("decode_tok_s")]
        by_policy: dict[str, list[float]] = {}
        for row in rows:
            by_policy.setdefault(str(row["policy"]), []).append(float(row["decode_tok_s"]))
        baseline = median(by_policy.get("baseline_no_svd", []))
        best_policy = "baseline_no_svd"
        best_tok = baseline or 0.0
        best_detail = ""
        for policy, vals in by_policy.items():
            tok = median(vals)
            if tok is not None and tok > best_tok:
                best_policy = policy
                best_tok = tok
                details = [row for row in rows if row["policy"] == policy]
                best_detail = str(details[0].get("detail", "")) if details else ""
        out.append(
            {
                "scenario": scenario,
                "loads": rows[0].get("loads", "") if rows else "",
                "baseline_tok_s": baseline,
                "selected_policy": best_policy,
                "selected_detail": best_detail,
                "selected_tok_s": best_tok,
                "speedup_vs_baseline": best_tok / baseline if baseline else None,
                "ppl": 15.1424,
                "ppl_note": "no-SVD selected path; same as Q4 baseline ctx128 sanity",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Q4 lossless scheduler candidates.")
    parser.add_argument("--out-dir", type=Path, default=EXP12 / "results/q4_lossless_scheduler_20260504_r1")
    parser.add_argument("--scenarios-json", type=Path)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--decode-binary", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--q4-model", type=Path, default=DEFAULT_Q4)
    parser.add_argument("--layer-client", type=Path, default=DEFAULT_LAYER_CLIENT)
    parser.add_argument("--android-dir", default="/data/local/tmp/CE_Ada")
    parser.add_argument("--android-model", default="./qwen.q4_0.gguf")
    parser.add_argument("--adb-serial", default="", help="ADB serial for the Android phone. Use this when an emulator is also connected.")
    parser.add_argument("--pc-host", default="10.126.59.25")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--coop-threads", type=int, default=8, help="Thread count passed to layer_coop_client/server.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--max-major-candidates", type=int, default=0, help="0 means enumerate every p=1..N-1 major-only candidate.")
    parser.add_argument("--skip-offload", action="store_true")
    parser.add_argument("--splits", type=lambda s: [int(x) for x in s.split(",") if x], help="Override layer-coop split M candidates.")
    parser.add_argument("--base-port", type=int, default=18800)
    args = parser.parse_args()
    global ADB_SERIAL
    ADB_SERIAL = args.adb_serial

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.scenarios_json.read_text()) if args.scenarios_json else scenario_defaults()

    raw_rows: list[dict[str, Any]] = []
    port_i = 0
    stop_phone_server()
    for scenario_obj in scenarios:
        scenario = str(scenario_obj["scenario"])
        cpus = [int(cpu) for cpu in scenario_obj["cpus"]]
        loads = [int(load) for load in scenario_obj["loads"]]
        run_cgroup = make_cgroup(args.cgroup_root, f"q4_lossless_run_{scenario}", cpus)
        load_cgroup = make_cgroup(args.cgroup_root, f"q4_lossless_load_{scenario}", cpus)
        major_candidates = selected_major_prefixes(cpus, loads, args.max_major_candidates)
        split_candidates = [] if args.skip_offload else choose_split_candidates(cpus, loads, args.splits)
        for rep in range(args.repeats):
            print(f"== {scenario} rep={rep} baseline/major/offload ==", flush=True)
            load_procs = start_heterogeneous_load(load_cgroup, cpus, loads)
            try:
                base = decode_once(args.decode_binary, args.q4_model, run_cgroup, cpus, args.tokens, args.timeout_s)
                base.update({
                    "scenario": scenario, "repeat": rep, "policy": "baseline_no_svd",
                    "detail": cpu_spec(cpus), "cpus": cpu_spec(cpus), "loads": ",".join(map(str, loads)),
                })
                raw_rows.append(base)

                for p, major_cpus in major_candidates:
                    major_cgroup = make_cgroup(args.cgroup_root, f"q4_lossless_major_{scenario}_p{p}", major_cpus)
                    row = decode_once(args.decode_binary, args.q4_model, major_cgroup, major_cpus, args.tokens, args.timeout_s)
                    row.update({
                        "scenario": scenario, "repeat": rep, "policy": "major_only_no_svd",
                        "detail": f"p={p};major={cpu_spec(major_cpus)}",
                        "cpus": cpu_spec(cpus), "loads": ",".join(map(str, loads)),
                    })
                    raw_rows.append(row)

                for split_m in split_candidates:
                    row = run_layer_coop(
                        client_binary=args.layer_client,
                        pc_model=args.q4_model,
                        android_dir=args.android_dir,
                        android_model=args.android_model,
                        pc_host=args.pc_host,
                        run_cgroup=run_cgroup,
                        pc_cpus=cpus,
                        split_m=split_m,
                        tokens=args.tokens,
                        threads=args.coop_threads,
                        port=args.base_port + port_i,
                        out_dir=args.out_dir,
                        label=f"{scenario}_r{rep}_M{split_m}",
                        timeout_s=args.timeout_s,
                    )
                    port_i += 1
                    row.update({
                        "scenario": scenario, "repeat": rep, "policy": "edge_end_no_svd",
                        "detail": f"M={split_m};PC=[0,{split_m});Phone=[{split_m},28)",
                        "cpus": cpu_spec(cpus), "loads": ",".join(map(str, loads)),
                    })
                    raw_rows.append(row)
            finally:
                stop_heterogeneous_load(load_cgroup, load_procs)
                stop_phone_server()
            write_csv(
                args.out_dir / "raw.csv",
                raw_rows,
                [
                    "scenario", "repeat", "policy", "detail", "status", "cpus", "loads",
                    "decode_tok_s", "generation_decode_ms", "generated",
                    "steady_decode_ms", "prefix_decode_ms", "server_decode_ms", "network_wait_ms",
                    "prefill_decode_ms", "elapsed_s", "client_log", "server_log", "output_tail",
                ],
            )

    summary = summarize(raw_rows)
    write_csv(
        args.out_dir / "summary.csv",
        summary,
        ["scenario", "loads", "baseline_tok_s", "selected_policy", "selected_detail", "selected_tok_s", "speedup_vs_baseline", "ppl", "ppl_note"],
    )
    lines = [
        "# Q4 Lossless Scheduler Benchmark",
        "",
        f"- Q4 model: `{args.q4_model}`",
        f"- tokens: `{args.tokens}`",
        f"- repeats: `{args.repeats}`",
        "- PPL: selected no-SVD paths keep baseline PPL by construction; ctx128 reference PPL = `15.1424 +/- 4.47405`.",
        "",
        "| scenario | loads | selected | detail | baseline tok/s | selected tok/s | speedup | PPL |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['scenario']}` | `{row['loads']}` | `{row['selected_policy']}` | `{row['selected_detail']}` | "
            f"{float(row['baseline_tok_s'] or 0):.4f} | {float(row['selected_tok_s'] or 0):.4f} | "
            f"{float(row['speedup_vs_baseline'] or 0):.3f}x | {float(row['ppl']):.4f} |"
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(raw_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
