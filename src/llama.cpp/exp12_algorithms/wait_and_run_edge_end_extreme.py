#!/usr/bin/env python3
"""Wait for an arm64 Android device, then run the edge_end extreme search."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP12 = ROOT / "src/llama.cpp/exp12_algorithms"


def adb(port: str, args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-P", port, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        timeout=timeout,
    )


def devices(port: str) -> list[str]:
    out = adb(port, ["devices"], timeout=5).stdout.splitlines()
    serials: list[str] = []
    for line in out[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


def prop(port: str, serial: str, name: str) -> str:
    return adb(port, ["-s", serial, "shell", f"getprop {name}"], timeout=5).stdout.strip()


def find_arm64_device(port: str) -> tuple[str, str, str] | None:
    for serial in devices(port):
        abi = prop(port, serial, "ro.product.cpu.abi")
        qemu = prop(port, serial, "ro.boot.qemu")
        model = prop(port, serial, "ro.product.model")
        if abi in {"arm64-v8a", "arm64"} and qemu != "1":
            return serial, model, abi
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-port", default="5039")
    parser.add_argument("--out-dir", type=Path, default=EXP12 / "results/edge_end_extreme_search_20260505_r1")
    parser.add_argument("--wait-s", type=float, default=600.0)
    parser.add_argument("--poll-s", type=float, default=5.0)
    parser.add_argument("--pc-host", default="10.126.59.25")
    args = parser.parse_args()

    deadline = time.time() + args.wait_s
    found: tuple[str, str, str] | None = None
    while time.time() < deadline:
        found = find_arm64_device(args.adb_port)
        if found:
            break
        print(f"waiting for arm64 real device on adb port {args.adb_port} ...", flush=True)
        time.sleep(args.poll_s)
    if not found:
        print(f"no arm64 real device found on adb port {args.adb_port}", flush=True)
        return 2

    serial, model, abi = found
    print(f"found {serial}: model={model} abi={abi}", flush=True)

    adb(args.adb_port, ["-s", serial, "shell", "mkdir -p /data/local/tmp/CE_Ada"], timeout=10)
    for src in [
        ROOT / "build-android/layer_mobile_server",
        ROOT / "build-android/layer_tail_local_bench",
        ROOT / "src/llama.cpp/gguf_models/qwen.q4_0.gguf",
    ]:
        print(f"pushing {src.name}", flush=True)
        subprocess.run(["adb", "-P", args.adb_port, "-s", serial, "push", str(src), "/data/local/tmp/CE_Ada/"], check=True)
    adb(
        args.adb_port,
        ["-s", serial, "shell", "chmod +x /data/local/tmp/CE_Ada/layer_mobile_server /data/local/tmp/CE_Ada/layer_tail_local_bench"],
        timeout=10,
    )

    raw = args.out_dir / "raw.csv"
    cmd = [
        "python3",
        str(EXP12 / "benchmark_q4_lossless_scheduler.py"),
        "--out-dir",
        str(args.out_dir),
        "--scenarios-json",
        str(args.out_dir / "scenarios.json"),
        "--tokens",
        "32",
        "--repeats",
        "2",
        "--max-major-candidates",
        "0",
        "--splits",
        "2,4,6,8",
        "--pc-host",
        args.pc_host,
        "--timeout-s",
        "300",
        "--adb-port",
        args.adb_port,
        "--adb-serial",
        serial,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run(
        [
            "python3",
            str(EXP12 / "export_edge_end_winners.py"),
            "--raw",
            str(raw),
            "--out",
            str(args.out_dir / "plot_edge_end_winners.csv"),
            "--min-occupied",
            "8",
            "--min-load",
            "70",
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
