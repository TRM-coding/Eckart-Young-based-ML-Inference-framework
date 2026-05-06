#!/usr/bin/env python3
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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[3]
EXP12 = ROOT / "src/llama.cpp/exp12_algorithms"
import sys

sys.path.insert(0, str(EXP12))
from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, make_cgroup, sudo_sh  # noqa: E402
from run_exp12_local import cpu_spec, parse_cpu_list  # noqa: E402


FORWARD_RE = re.compile(r"forward_ms_mean=(?P<mean>[0-9.eE+-]+)")
COOP_RE = re.compile(r"\[resnet-coop-steady\].*?split_blocks=(?P<split>\d+).*?throughput=(?P<img_s>[0-9.eE+-]+).*?prefix_ms=(?P<prefix>[0-9.eE+-]+).*?send_ms=(?P<send>[0-9.eE+-]+).*?wait_ms=(?P<wait>[0-9.eE+-]+).*?server_ms=(?P<server>[0-9.eE+-]+)")


def configure_font() -> None:
    for p in [
        Path("/usr/share/fonts/truetype/simsun.ttc"),
        Path("/usr/share/fonts/truetype/simsun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    ]:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(p)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def run_in_cgroup(cmd: list[str], cgroup: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    script = (
        f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/3dparty/oneDNN/src'))}:{shlex.quote(str(ROOT / 'build-release-current/bin'))}:$LD_LIBRARY_PATH; "
        f"exec {shlex.join(cmd)}"
    )
    proc = subprocess.Popen(["sudo", "-n", "bash", "-lc", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, None)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5)
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return subprocess.CompletedProcess(cmd, -999, out, None)


def start_load(cgroup: Path, cpus: list[int], workers_per_core: int, method: str) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    for cpu in cpus:
        for _ in range(workers_per_core):
            cmd = ["taskset", "-c", str(cpu), "stress-ng", "--cpu", "1", "--cpu-load", "100", "--cpu-method", method, "--quiet"]
            script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
            procs.append(subprocess.Popen(["sudo", "-n", "bash", "-lc", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
    time.sleep(1.0)
    return procs


def stop_load(cgroup: Path, procs: list[subprocess.Popen[str]]) -> None:
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.2)


def bench_pc(args: argparse.Namespace, cgroup: Path, run_cpus: list[int], label: str) -> dict[str, Any]:
    cmd = [
        "taskset",
        "-c",
        cpu_spec(run_cpus),
        str(args.pc_binary),
        "--model",
        str(args.model),
        "--image",
        str(args.image),
        "--threads",
        str(len(run_cpus)),
        "--top-k",
        "1",
        "--benchmark",
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    proc = run_in_cgroup(cmd, cgroup, args.timeout_s)
    m = FORWARD_RE.search(proc.stdout)
    img_s = 1000.0 / float(m.group("mean")) if m else 0.0
    return {"policy": label, "status": "ok" if proc.returncode == 0 and m else f"rc_{proc.returncode}", "img_s": img_s, "run_cpus": cpu_spec(run_cpus), "log": proc.stdout[-1200:]}


def bench_coop(args: argparse.Namespace, run_cgroup: Path, split: int, port: int) -> dict[str, Any]:
    out = args.out_dir
    client_log = out / f"coop_M{split}_client.log"
    server_log = out / f"coop_M{split}_server.log"
    client_cmd = [
        "taskset",
        "-c",
        cpu_spec(args.cpus),
        str(args.client_binary),
        str(args.model),
        str(args.image),
        str(args.pc_threads),
        str(port),
        str(split),
        str(args.repeat),
        str(args.warmup),
    ]
    script = (
        f"echo $$ > {shlex.quote(str(run_cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/3dparty/oneDNN/src'))}:{shlex.quote(str(ROOT / 'build-release-current/bin'))}:$LD_LIBRARY_PATH; "
        f"exec {shlex.join(client_cmd)}"
    )
    cp = subprocess.Popen(["sudo", "-n", "bash", "-lc", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    time.sleep(0.5)
    server_cmd = (
        f"cd {shlex.quote(args.android_dir)} && "
        f"LD_LIBRARY_PATH=. taskset -a ff ./resnet_coop_server ./resnet50-f32.gguf {args.phone_threads} {args.pc_host}:{port} {split}"
    )
    sp = subprocess.Popen(["adb", "-P", args.adb_port, "-s", args.adb_serial, "shell", server_cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    try:
        cout, _ = cp.communicate(timeout=args.timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(cp.pid, signal.SIGKILL)
        cout = ""
    try:
        sout, _ = sp.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(sp.pid, signal.SIGKILL)
        sout = ""
    client_log.write_text(cout, errors="replace")
    server_log.write_text(sout, errors="replace")
    m = COOP_RE.search(cout)
    row: dict[str, Any] = {"policy": f"协同推理 M={split}", "split": split, "status": "ok" if m else "failed", "img_s": float(m.group("img_s")) if m else 0.0, "log": client_log.name, "server_log": server_log.name}
    if m:
        row.update({"prefix_ms": m.group("prefix"), "send_ms": m.group("send"), "wait_ms": m.group("wait"), "server_ms": m.group("server")})
    return row


def plot(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    configure_font()
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18, "figure.dpi": 150, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r["policy"].replace(" ", "\n") for r in rows]
    vals = [float(r["img_s"]) for r in rows]
    colors = ["#e6550d" if "协同" in r["policy"] else "#4c78a8" for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="#333333")
    pc = vals[0]
    best_no = max(vals[1:4]) if len(vals) >= 4 else 0.0
    ax.axhline(pc, color="#d73027", linestyle="--", linewidth=2, label=f"PC基线 {pc:.2f} img/s")
    ax.axhline(best_no, color="#377eb8", linestyle=":", linewidth=2.5, label=f"最优No-Offloading {best_no:.2f} img/s")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=16)
    best_coop = max([v for r, v in zip(rows, vals) if "协同" in r["policy"]] or [0.0])
    if best_coop > 0:
        ax.text(0.68, 0.82, f"相对PC基线 {best_coop / pc:.1f}x\n相对No-Offloading {best_coop / best_no:.2f}x", transform=ax.transAxes, ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "#ff7f0e", "boxstyle": "round,pad=0.35"})
    ax.set_title("ResNet50 极端负载下真实协同推理性能", fontsize=24)
    ax.set_ylabel("吞吐量 (images/s)")
    ax.set_xlabel("推理策略")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    figdir = args.out_dir / "figures"
    figdir.mkdir(exist_ok=True)
    fig.savefig(figdir / "resnet_extreme_coop_vs_no_offloading.png")
    fig.savefig(figdir / "resnet_extreme_coop_vs_no_offloading.pdf")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "src/llama.cpp/exp8_resnet50/results/resnet_extreme_coop_20260506_r1")
    p.add_argument("--cpus", default="60-67")
    p.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    p.add_argument("--pc-binary", type=Path, default=ROOT / "build-release-current/run_resnet50")
    p.add_argument("--client-binary", type=Path, default=ROOT / "build-release-current/resnet_coop_client")
    p.add_argument("--model", type=Path, default=ROOT / "src/llama.cpp/gguf_models/resnet50-f32.gguf")
    p.add_argument("--image", type=Path, default=ROOT / "src/llama.cpp/exp8_resnet50/results/resnet_scheduler_figures_real_20260506_r1/imagenet_val_subset/ILSVRC2012_val_00000001.JPEG")
    p.add_argument("--adb-port", default="5038")
    p.add_argument("--adb-serial", default="10.20.0.3:5555")
    p.add_argument("--android-dir", default="/data/local/tmp/CE_Ada/resnet")
    p.add_argument("--pc-host", default="10.20.0.2")
    p.add_argument("--load-method", default="matrixprod")
    p.add_argument("--workers-per-core", type=int, default=2)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--pc-threads", type=int, default=4)
    p.add_argument("--phone-threads", type=int, default=8)
    p.add_argument("--splits", default="12,14,16")
    args = p.parse_args()
    args.cpus = parse_cpu_list(args.cpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_cg = make_cgroup(args.cgroup_root, "resnet_extreme_coop_load", args.cpus)
    run_cg = make_cgroup(args.cgroup_root, "resnet_extreme_coop_run", args.cpus)
    procs = start_load(load_cg, args.cpus, args.workers_per_core, args.load_method)
    try:
        rows: list[dict[str, Any]] = []
        rows.append(bench_pc(args, run_cg, args.cpus, "PC基线"))
        for n in [2, 4, 6]:
            rows.append(bench_pc(args, run_cg, args.cpus[-n:], f"No-Offloading {n}核"))
        for i, split in enumerate([int(x) for x in args.splits.split(",") if x], start=0):
            rows.append(bench_coop(args, run_cg, split, 19100 + i))
        write_csv(args.out_dir / "raw.csv", rows)
        plot(args, rows)
        print(args.out_dir)
    finally:
        stop_load(load_cg, procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
