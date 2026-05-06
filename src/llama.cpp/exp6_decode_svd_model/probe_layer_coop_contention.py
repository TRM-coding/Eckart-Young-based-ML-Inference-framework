#!/usr/bin/env python3
"""Probe real layer-coop speed under stronger PC contention.

This script compares the same layer-level toolchain under the same PC load:

- PC-only baseline: layer_tail_local_bench full_token on the PC cores.
- Cooperative split: layer_coop_client on PC + layer_mobile_server on Android.

The load generator can optionally run stress-ng under chrt/nice to create real
competition with ggml worker threads.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/tianruiming/CE_ADA_LLAMA")
EXP6 = ROOT / "src/llama.cpp/exp6_decode_svd_model"
DEFAULT_OUT = EXP6 / "results/layer_coop_contention_probe_20260506_r1"

BASE_RE = re.compile(r"\[tail-local-bench\]\s+mode=full_token.*?throughput=(?P<tok_s>[0-9.eE+-]+)\s+tok/s")
STEADY_RE = re.compile(
    r"\[layer-coop-steady\]\s+generated=(?P<generated>\d+)\s+"
    r"steady_decode_ms=(?P<steady_decode_ms>[0-9.eE+-]+)\s+"
    r"steady_throughput=(?P<tok_s>[0-9.eE+-]+)\s+tok/s\s+"
    r"steady_ms_per_token=(?P<ms_per_token>[0-9.eE+-]+)\s+"
    r"prefix_decode_ms=(?P<prefix_ms>[0-9.eE+-]+)\s+"
    r"server_decode_ms=(?P<server_ms>[0-9.eE+-]+)\s+"
    r"network_wait_ms=(?P<network_ms>[0-9.eE+-]+)"
)


def cpu_spec(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges = []
    start = prev = cpus[0]
    for cpu in cpus[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def sudo_sh(script: str, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sudo", "-n", "bash", "-lc", script], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s, errors="replace")


def read_cgroup(path: Path, default: str) -> str:
    c = sudo_sh(f"cat {shlex.quote(str(path))}", timeout_s=5)
    return c.stdout.strip() if c.returncode == 0 and c.stdout.strip() else default


def make_cgroup(root: Path, name: str, cpus: list[int]) -> Path:
    cg = root / name
    mems = read_cgroup(root / "cpuset.mems.effective", "0")
    sudo_sh(
        f"mkdir -p {shlex.quote(str(cg))}; "
        f"echo {shlex.quote(mems)} > {shlex.quote(str(cg / 'cpuset.mems'))}; "
        f"echo {shlex.quote(cpu_spec(cpus))} > {shlex.quote(str(cg / 'cpuset.cpus'))}",
        timeout_s=5,
    )
    return cg


def adb(port: str, serial: str, args: list[str], timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    serial_args = ["-s", serial] if serial else []
    return subprocess.run(["adb", "-P", port, *serial_args, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s, errors="replace")


def stop_cgroup(cg: Path) -> None:
    sudo_sh(f"test ! -e {shlex.quote(str(cg / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cg / 'cgroup.kill'))}", timeout_s=5)


def start_load(cg: Path, cpus: list[int], load_pct: int, workers_per_core: int, method: str, prio_mode: str) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    if load_pct <= 0:
        return procs
    for cpu in cpus:
        for _ in range(workers_per_core):
            stress = [
                "taskset", "-c", str(cpu),
                "stress-ng", "--cpu", "1", "--cpu-load", str(load_pct), "--cpu-method", method, "--quiet",
            ]
            if prio_mode == "chrt80":
                cmd = ["chrt", "-f", "80", *stress]
            elif prio_mode == "nice-20":
                cmd = ["nice", "-n", "-20", *stress]
            else:
                cmd = stress
            script = f"echo $$ > {shlex.quote(str(cg / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
            procs.append(subprocess.Popen(["sudo", "-n", "bash", "-lc", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, text=True))
    time.sleep(1.0)
    return procs


def stop_load(cg: Path, procs: list[subprocess.Popen[str]]) -> None:
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    stop_cgroup(cg)


def run_pc_full(
    args: argparse.Namespace,
    run_cg: Path,
    label: str,
    policy: str = "pc_full",
    run_cpus: list[int] | None = None,
    threads: int | None = None,
) -> dict[str, object]:
    run_cpus = run_cpus or args.pc_cpus
    threads = threads or args.threads
    log = args.out_dir / f"{label}_{policy}.log"
    cmd = [
        "taskset", "-c", cpu_spec(run_cpus),
        str(args.pc_full_binary), str(args.pc_model), str(args.tokens), str(threads), "full_token",
    ]
    script = (
        f"echo $$ > {shlex.quote(str(run_cg / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/bin'))}; "
        f"exec {shlex.join(cmd)}"
    )
    c = sudo_sh(script, timeout_s=args.timeout_s)
    log.write_text(c.stdout, errors="replace")
    m = BASE_RE.search(c.stdout)
    return {
        "policy": policy,
        "status": "ok" if c.returncode == 0 and m else f"rc_{c.returncode}",
        "tok_s": float(m.group("tok_s")) if m else "",
        "run_cpus": cpu_spec(run_cpus),
        "threads": threads,
        "log": log.name,
    }


def run_split(args: argparse.Namespace, run_cg: Path, label: str, port: int, split: int) -> dict[str, object]:
    client_log = args.out_dir / f"{label}_M{split}_client.log"
    server_log = args.out_dir / f"{label}_M{split}_server.log"
    adb(args.adb_port, args.adb_serial, ["shell", "pkill -f layer_mobile_server || true"], timeout_s=5)
    client_cmd = [str(args.client_binary), str(args.pc_model), str(args.tokens), str(args.threads), f"listen:{port}", str(split), "0"]
    client_script = (
        f"echo $$ > {shlex.quote(str(run_cg / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/bin'))}; "
        f"LAYER_COOP_CPU_MASK=0x0 LAYER_COOP_PRIO={args.client_prio} exec {shlex.join(client_cmd)}"
    )
    cp = subprocess.Popen(["sudo", "-n", "bash", "-lc", client_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    time.sleep(args.client_start_delay_s)
    server_cmd = (
        f"cd {shlex.quote(args.android_dir)} && "
        f"LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff LAYER_COOP_KEEP_HOT=1 LAYER_COOP_PRIO={args.server_prio} "
        f"taskset -a ff ./layer_mobile_server {shlex.quote(args.android_model)} 0 {split} {args.threads} {args.pc_host}:{port}"
    )
    sp = subprocess.Popen(["adb", "-P", args.adb_port, "-s", args.adb_serial, "shell", server_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    status = "ok"
    try:
        cout, _ = cp.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        status = "client_timeout"
        try:
            os.killpg(cp.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        cout, _ = cp.communicate(timeout=5)
    adb(args.adb_port, args.adb_serial, ["shell", "pkill -f layer_mobile_server || true"], timeout_s=5)
    try:
        sout, _ = sp.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(sp.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sout, _ = sp.communicate(timeout=5)
    client_log.write_text(cout, errors="replace")
    server_log.write_text(sout, errors="replace")
    m = STEADY_RE.search(cout)
    row: dict[str, object] = {"policy": "split", "split_m": split, "status": status, "log": client_log.name, "server_log": server_log.name}
    if m:
        row.update({k: float(v) if k != "generated" else int(v) for k, v in m.groupdict().items()})
    elif status == "ok":
        row["status"] = f"client_rc_{cp.returncode}"
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["label", "load_pct", "workers_per_core", "method", "prio_mode", "policy", "split_m", "status", "tok_s", "ms_per_token", "prefix_ms", "server_ms", "network_ms", "generated", "run_cpus", "threads", "log", "server_log"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup/tianruiming-exclusive"))
    p.add_argument("--pc-cpus", type=parse_cpu_list, default=list(range(71, 80)))
    p.add_argument("--pc-host", default="10.20.0.2")
    p.add_argument("--adb-port", default="5038")
    p.add_argument("--adb-serial", default="10.20.0.3:5555")
    p.add_argument("--pc-model", type=Path, default=ROOT / "src/llama.cpp/gguf_models/qwen.q4_0.gguf")
    p.add_argument("--android-dir", default="/data/local/tmp/CE_Ada")
    p.add_argument("--android-model", default="./qwen.q4_0.gguf")
    p.add_argument("--pc-full-binary", type=Path, default=ROOT / "build-release-current/layer_tail_local_bench")
    p.add_argument("--client-binary", type=Path, default=ROOT / "build-release-current/layer_coop_client")
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--loads", default="0,20,50,80,100")
    p.add_argument("--splits", default="2,4,6,8")
    p.add_argument("--workers-per-core", default="1")
    p.add_argument("--major-counts", default="")
    p.add_argument("--load-method", default="matrixprod")
    p.add_argument("--prio-modes", default="normal")
    p.add_argument("--client-prio", default="0")
    p.add_argument("--server-prio", default="0")
    p.add_argument("--base-port", type=int, default=19000)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--client-start-delay-s", type=float, default=1.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_cg = make_cgroup(args.cgroup_root, "layer_coop_contention_run", args.pc_cpus)
    load_cg = make_cgroup(args.cgroup_root, "layer_coop_contention_load", args.pc_cpus)
    loads = [int(x) for x in args.loads.split(",") if x]
    splits = [int(x) for x in args.splits.split(",") if x]
    worker_counts = [int(x) for x in args.workers_per_core.split(",") if x]
    major_counts = [int(x) for x in args.major_counts.split(",") if x]
    prio_modes = [x for x in args.prio_modes.split(",") if x]
    rows: list[dict[str, object]] = []
    case_i = 0
    for prio_mode in prio_modes:
        for workers in worker_counts:
            for load in loads:
                label = f"{prio_mode}_w{workers}_load{load}"
                procs = start_load(load_cg, args.pc_cpus, load, workers, args.load_method, prio_mode)
                try:
                    base = run_pc_full(args, run_cg, label)
                    base.update({"label": label, "load_pct": load, "workers_per_core": workers, "method": args.load_method, "prio_mode": prio_mode})
                    rows.append(base)
                    for count in major_counts:
                        major_cpus = args.pc_cpus[:count]
                        major = run_pc_full(args, run_cg, label, policy=f"major_only_{count}c", run_cpus=major_cpus, threads=count)
                        major.update({"label": label, "load_pct": load, "workers_per_core": workers, "method": args.load_method, "prio_mode": prio_mode})
                        rows.append(major)
                    for split in splits:
                        row = run_split(args, run_cg, label, args.base_port + case_i, split)
                        case_i += 1
                        row.update({"label": label, "load_pct": load, "workers_per_core": workers, "method": args.load_method, "prio_mode": prio_mode})
                        rows.append(row)
                    write_rows(args.out_dir / "raw.csv", rows)
                finally:
                    stop_load(load_cg, procs)
                    adb(args.adb_port, args.adb_serial, ["shell", "pkill -f layer_mobile_server || true"], timeout_s=5)
    write_rows(args.out_dir / "raw.csv", rows)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
