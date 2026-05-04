#!/usr/bin/env python3
"""Sweep layer-level PC/Android cooperative inference over PC load and split point."""

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
DEFAULT_OUT = ROOT / "src/llama.cpp/exp6_decode_svd_model/results/layer_coop_split_load_20260502_r1"


STEADY_RE = re.compile(
    r"\[layer-coop-steady\]\s+generated=(?P<generated>\d+)\s+"
    r"steady_decode_ms=(?P<steady_decode_ms>[0-9.eE+-]+)\s+"
    r"steady_throughput=(?P<steady_throughput>[0-9.eE+-]+)\s+tok/s\s+"
    r"steady_ms_per_token=(?P<steady_ms_per_token>[0-9.eE+-]+)\s+"
    r"prefix_decode_ms=(?P<prefix_decode_ms>[0-9.eE+-]+)\s+"
    r"server_decode_ms=(?P<server_decode_ms>[0-9.eE+-]+)\s+"
    r"network_wait_ms=(?P<network_wait_ms>[0-9.eE+-]+)"
)
PROFILE_RE = re.compile(
    r"\[layer-coop-profile\]\s+requests=(?P<requests>\d+)\s+"
    r"send_bytes=(?P<send_bytes>\d+)\s+recv_bytes=(?P<recv_bytes>\d+)\s+"
    r"send_ms=(?P<send_ms>[0-9.eE+-]+)\s+recv_header_ms=(?P<recv_header_ms>[0-9.eE+-]+)\s+"
    r"roundtrip_ms=(?P<roundtrip_ms>[0-9.eE+-]+)"
)
SERVER_FINAL_RE = re.compile(
    r"\[layer-coop-server-final\]\s+requests=(?P<server_requests>\d+)\s+"
    r"header_wait_ms=(?P<header_wait_ms>[0-9.eE+-]+)\s+"
    r"payload_recv_ms=(?P<payload_recv_ms>[0-9.eE+-]+)\s+"
    r"decode_ms=(?P<server_final_decode_ms>[0-9.eE+-]+)\s+"
    r"send_ms=(?P<server_send_ms>[0-9.eE+-]+)"
)


def cpu_spec(cpus: list[int]) -> str:
    if not cpus:
        return ""
    ranges: list[str] = []
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
            start_s, end_s = part.split("-", 1)
            cpus.extend(range(int(start_s), int(end_s) + 1))
        else:
            cpus.append(int(part))
    return cpus


def sudo_sh(script: str, timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["sudo", "-n", "bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        errors="replace",
    )


def read_cgroup_file(path: Path, default: str) -> str:
    completed = sudo_sh(f"cat {shlex.quote(str(path))}", timeout_s=5.0)
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else default


def make_cgroup(root: Path, name: str, cpus: list[int]) -> Path:
    cg = root / name
    mems = read_cgroup_file(root / "cpuset.mems.effective", "0")
    sudo_sh(
        f"mkdir -p {shlex.quote(str(cg))}; "
        f"echo {shlex.quote(mems)} > {shlex.quote(str(cg / 'cpuset.mems'))}; "
        f"echo {shlex.quote(cpu_spec(cpus))} > {shlex.quote(str(cg / 'cpuset.cpus'))}",
        timeout_s=5.0,
    )
    return cg


def start_load(cgroup: Path, cpus: list[int], load_pct: int) -> subprocess.Popen[str] | None:
    if load_pct <= 0:
        return None
    cmd = [
        "taskset",
        "-c",
        cpu_spec(cpus),
        "stress-ng",
        "--cpu",
        str(len(cpus)),
        "--cpu-load",
        str(load_pct),
        "--cpu-method",
        "matrixprod",
        "--quiet",
    ]
    script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
    proc = subprocess.Popen(
        ["sudo", "-n", "bash", "-lc", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    time.sleep(1.0)
    return proc


def stop_load(proc: subprocess.Popen[str] | None, cgroup: Path | None) -> None:
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if cgroup is not None:
        sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5.0)


def adb(args: list[str], timeout_s: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-P", "5038", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        errors="replace",
    )


def stop_phone_server() -> None:
    adb(["shell", "pkill -f layer_mobile_server || true"], timeout_s=5.0)


def parse_match(pattern: re.Pattern[str], text: str) -> dict[str, str]:
    matches = list(pattern.finditer(text))
    return matches[-1].groupdict() if matches else {}


def run_one(args: argparse.Namespace, run_cgroup: Path, out_dir: Path, load_pct: int, split: int, repeat: int, port: int) -> dict[str, object]:
    label = f"load{load_pct}_split{split}_r{repeat}"
    client_log = out_dir / f"{label}_client.log"
    server_log = out_dir / f"{label}_server.log"

    stop_phone_server()
    client_cmd = [
        str(args.client_binary),
        str(args.pc_model),
        str(args.tokens),
        str(args.threads),
        f"listen:{port}",
        str(split),
        "0",
    ]
    client_script = (
        f"echo $$ > {shlex.quote(str(run_cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; "
        f"exec {shlex.join(client_cmd)}"
    )
    client_proc = subprocess.Popen(
        ["sudo", "-n", "bash", "-lc", client_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    time.sleep(args.client_start_delay_s)

    server_cmd = (
        f"cd {shlex.quote(args.android_dir)} && "
        f"LD_LIBRARY_PATH=. LAYER_COOP_CPU_MASK=0xff LAYER_COOP_KEEP_HOT=1 "
        f"taskset -a ff ./layer_mobile_server {shlex.quote(args.android_model)} 0 {split} {args.threads} "
        f"{args.pc_host}:{port}"
    )
    server_proc = subprocess.Popen(
        ["adb", "-P", "5038", "shell", server_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )

    status = "ok"
    client_out = ""
    server_out = ""
    try:
        client_out, _ = client_proc.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        status = "client_timeout"
        try:
            os.killpg(client_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        client_out, _ = client_proc.communicate(timeout=5)
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

    if client_proc.returncode not in (0, None) and status == "ok":
        status = f"client_rc_{client_proc.returncode}"

    steady = parse_match(STEADY_RE, client_out)
    profile = parse_match(PROFILE_RE, client_out)
    server = parse_match(SERVER_FINAL_RE, server_out)
    if not steady and status == "ok":
        status = "missing_steady"

    row: dict[str, object] = {
        "status": status,
        "load_pct": load_pct,
        "split_m": split,
        "repeat": repeat,
        "tokens": args.tokens,
        "threads": args.threads,
        "port": port,
        "client_log": str(client_log.relative_to(out_dir)),
        "server_log": str(server_log.relative_to(out_dir)),
    }
    row.update(steady)
    row.update(profile)
    row.update(server)
    for key in [
        "steady_decode_ms",
        "steady_throughput",
        "steady_ms_per_token",
        "prefix_decode_ms",
        "server_decode_ms",
        "network_wait_ms",
        "send_ms",
        "recv_header_ms",
        "roundtrip_ms",
        "header_wait_ms",
        "payload_recv_ms",
        "server_final_decode_ms",
        "server_send_ms",
    ]:
        if key in row and row[key] != "":
            row[key] = float(row[key])
    for key in ["generated", "requests", "send_bytes", "recv_bytes", "server_requests"]:
        if key in row and row[key] != "":
            row[key] = int(row[key])
    return row


def write_report(out_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    best_by_load: dict[int, dict[str, object]] = {}
    for row in ok_rows:
        load = int(row["load_pct"])
        if load not in best_by_load or float(row["steady_throughput"]) > float(best_by_load[load]["steady_throughput"]):
            best_by_load[load] = row

    lines: list[str] = []
    lines.append("# Layer Coop Split/PC-Load Sweep Report")
    lines.append("")
    lines.append(f"日期：2026-05-02")
    lines.append("")
    lines.append("## 设置")
    lines.append("")
    lines.append(f"- tokens: `{args.tokens}`")
    lines.append(f"- PC cpus: `{cpu_spec(args.pc_cpus)}`")
    lines.append(f"- PC loads: `{','.join(map(str, args.loads))}`")
    lines.append(f"- split M: `{','.join(map(str, args.splits))}`")
    lines.append(f"- repeats: `{args.repeats}`")
    lines.append(f"- phone server: `LAYER_COOP_KEEP_HOT=1`")
    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    lines.append("| PC load | split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok | status |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        gen = float(row.get("generated") or args.tokens)
        def per_tok(name: str) -> str:
            value = row.get(name)
            return f"{float(value) / gen:.2f}" if isinstance(value, (float, int)) and gen else ""
        lines.append(
            f"| {row['load_pct']} | {row['split_m']} | "
            f"{float(row['steady_throughput']):.2f}" if row.get("steady_throughput") != "" and row.get("steady_throughput") is not None else f"| {row['load_pct']} | {row['split_m']} | "
        )
        if row.get("steady_throughput") != "" and row.get("steady_throughput") is not None:
            lines[-1] += (
                f" | {float(row['steady_ms_per_token']):.2f}"
                f" | {per_tok('prefix_decode_ms')}"
                f" | {per_tok('server_decode_ms')}"
                f" | {per_tok('network_wait_ms')}"
                f" | {row['status']} |"
            )
        else:
            lines[-1] += f" |  |  |  |  | {row['status']} |"
    lines.append("")
    lines.append("## Best Split Per Load")
    lines.append("")
    lines.append("| PC load | best split M | tok/s | ms/token | PC prefix ms/tok | phone tail ms/tok | net/sync ms/tok |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for load in sorted(best_by_load):
        row = best_by_load[load]
        gen = float(row.get("generated") or args.tokens)
        lines.append(
            f"| {load} | {row['split_m']} | {float(row['steady_throughput']):.2f} | "
            f"{float(row['steady_ms_per_token']):.2f} | "
            f"{float(row['prefix_decode_ms']) / gen:.2f} | "
            f"{float(row['server_decode_ms']) / gen:.2f} | "
            f"{float(row['network_wait_ms']) / gen:.2f} |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- `split M` 表示 PC 计算 `[0,M)`，手机计算 `[M,28)`。")
    lines.append("- `net/sync ms/tok = client roundtrip - server decode`，包含 TCP 传输、收包、发包与两端同步等待。")
    lines.append("- 本轮使用 64-token decode 作为 sweep；最终候选点建议再用 128-token 重复验证。")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--client-binary", type=Path, default=ROOT / "build-release-current/layer_coop_client")
    parser.add_argument("--pc-model", type=Path, default=ROOT / "src/llama.cpp/gguf_models/qwen.q4_0.gguf")
    parser.add_argument("--android-dir", default="/data/local/tmp/CE_Ada")
    parser.add_argument("--android-model", default="./qwen.q4_0.gguf")
    parser.add_argument("--pc-host", default="10.126.59.25")
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup/tianruiming-exclusive"))
    parser.add_argument("--pc-cpus", type=parse_cpu_list, default=list(range(71, 80)))
    parser.add_argument("--loads", type=lambda s: [int(x) for x in s.split(",") if x], default=[0, 50, 80])
    parser.add_argument("--splits", type=lambda s: [int(x) for x in s.split(",") if x], default=[8, 12, 14, 16, 20])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=18300)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--client-start-delay-s", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_cgroup = make_cgroup(args.cgroup_root, "layer_coop_split_run", args.pc_cpus)
    load_cgroup = make_cgroup(args.cgroup_root, "layer_coop_split_load", args.pc_cpus)
    stop_phone_server()

    rows: list[dict[str, object]] = []
    fieldnames: list[str] = [
        "status", "load_pct", "split_m", "repeat", "tokens", "threads", "generated",
        "steady_decode_ms", "steady_throughput", "steady_ms_per_token",
        "prefix_decode_ms", "server_decode_ms", "network_wait_ms",
        "requests", "send_bytes", "recv_bytes", "send_ms", "recv_header_ms", "roundtrip_ms",
        "server_requests", "header_wait_ms", "payload_recv_ms", "server_final_decode_ms", "server_send_ms",
        "port", "client_log", "server_log",
    ]

    try:
        case_i = 0
        for load in args.loads:
            load_proc = start_load(load_cgroup, args.pc_cpus, load)
            try:
                for split in args.splits:
                    for repeat in range(args.repeats):
                        port = args.base_port + case_i
                        print(f"== load={load} split={split} repeat={repeat} port={port} ==", flush=True)
                        row = run_one(args, run_cgroup, args.out_dir, load, split, repeat, port)
                        rows.append(row)
                        with (args.out_dir / "raw.csv").open("w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for r in rows:
                                writer.writerow({k: r.get(k, "") for k in fieldnames})
                        case_i += 1
            finally:
                stop_load(load_proc, load_cgroup)
    finally:
        stop_phone_server()
        stop_load(None, load_cgroup)

    write_report(args.out_dir, rows, args)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
