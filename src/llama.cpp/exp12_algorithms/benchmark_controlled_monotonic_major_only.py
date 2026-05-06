#!/usr/bin/env python3
"""Controlled major-only experiment for monotonic plot data.

All scenarios use the same eight available PC cores.  A configuration index
defines a load intensity, and occupied_count only changes how many of those
cores receive that same intensity.  This makes cross-figure comparisons
physically interpretable: for the same Config, 2 occupied cores should have no
more total background load than 4/6/8 occupied cores.

The benchmark intentionally uses layer_tail_local_bench instead of
decode_svd_test because decode_svd_test raises process/thread priority and can
mask real background contention.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, make_cgroup, sudo_sh
from run_exp12_local import cpu_spec, parse_cpu_list


ROOT = Path(__file__).resolve().parents[3]
EXP12 = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "src/llama.cpp/gguf_models/qwen.q4_0.gguf"
DEFAULT_BINARY = ROOT / "build-release-current/layer_tail_local_bench"
BENCH_RE = re.compile(
    r"\[tail-local-bench\]\s+mode=full_token\s+tokens=(?P<tokens>\d+)\s+threads=(?P<threads>\d+)\s+"
    r"prefill_ms=(?P<prefill_ms>[0-9.eE+-]+)\s+decode_ms=(?P<decode_ms>[0-9.eE+-]+)\s+"
    r"avg_decode_ms=(?P<avg_decode_ms>[0-9.eE+-]+)\s+throughput=(?P<tok_s>[0-9.eE+-]+)\s+tok/s"
)


@dataclass(frozen=True)
class Scenario:
    config_id: int
    occupied_count: int
    cpus: list[int]
    loads: list[int]


def configure_font() -> None:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_vectors() -> list[list[int]]:
    return [
        [10, 10, 10, 10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20, 20, 20, 20],
        [30, 40, 30, 40, 30, 40, 30, 40],
        [50, 50, 60, 60, 50, 50, 60, 60],
        [70, 70, 80, 80, 70, 70, 80, 80],
        [90, 90, 100, 100, 90, 90, 100, 100],
    ]


def scenarios(cpus: list[int]) -> list[Scenario]:
    out: list[Scenario] = []
    for config_id, vector in enumerate(load_vectors(), start=1):
        for occupied_count in [2, 4, 6, 8]:
            loads = [0] * len(cpus)
            for i in range(occupied_count):
                loads[i] = vector[i]
            out.append(Scenario(config_id=config_id, occupied_count=occupied_count, cpus=cpus, loads=loads))
    return out


def run_in_cgroup(cmd: list[str], cgroup: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    script = (
        f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/bin'))}; "
        f"exec {shlex.join(cmd)}"
    )
    proc = subprocess.Popen(
        ["sudo", "-n", "bash", "-lc", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
        sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5)
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return subprocess.CompletedProcess(cmd, -999, out, None)


def start_load(cgroup: Path, cpus: list[int], loads: list[int], method: str) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    for cpu, load in zip(cpus, loads):
        if load <= 0:
            continue
        cmd = ["taskset", "-c", str(cpu), "stress-ng", "--cpu", "1", "--cpu-load", str(load), "--cpu-method", method, "--quiet"]
        script = f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; exec {shlex.join(cmd)}"
        procs.append(
            subprocess.Popen(
                ["sudo", "-n", "bash", "-lc", script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        )
    if procs:
        time.sleep(1.0)
    return procs


def stop_load(cgroup: Path, procs: list[subprocess.Popen[str]]) -> None:
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5)
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.2)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    sudo_sh(f"test ! -e {shlex.quote(str(cgroup / 'cgroup.kill'))} || echo 1 > {shlex.quote(str(cgroup / 'cgroup.kill'))}", timeout_s=5)


def run_bench(binary: Path, model: Path, cgroup: Path, run_cpus: list[int], tokens: int, timeout_s: float) -> dict[str, Any]:
    threads = len(run_cpus)
    cmd = ["taskset", "-c", cpu_spec(run_cpus), str(binary), str(model), str(tokens), str(threads), "full_token"]
    started = time.time()
    completed = run_in_cgroup(cmd, cgroup, timeout_s)
    text = completed.stdout
    match = BENCH_RE.search(text)
    row: dict[str, Any] = {
        "status": "ok" if completed.returncode == 0 and match else f"rc_{completed.returncode}",
        "elapsed_s": time.time() - started,
        "output_tail": "\\n".join(text.splitlines()[-10:]),
    }
    if match:
        row.update({k: float(v) if k not in {"tokens", "threads"} else int(v) for k, v in match.groupdict().items()})
    return row


def ordered_major_candidates(cpus: list[int], loads: list[int], mode: str) -> list[tuple[str, list[int]]]:
    if mode == "idle-only":
        idle = [cpu for cpu, load in zip(cpus, loads) if load <= 0]
        return [("No-Offloading 空闲核", idle)] if idle else []
    ordered = [cpu for cpu, _ in sorted(zip(cpus, loads), key=lambda item: (item[1], item[0]))]
    candidates: list[tuple[str, list[int]]] = []
    for p in range(1, len(cpus)):
        candidates.append((f"No-Offloading {p}核", ordered[:p]))
    return candidates


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for sc in scenarios(args.cpus):
        name = f"{sc.occupied_count}occ_config{sc.config_id}"
        print(f"[scenario] {name} loads={sc.loads}", flush=True)
        load_cg = make_cgroup(args.cgroup_root, f"monotonic_load_{name}", sc.cpus)
        run_all_cg = make_cgroup(args.cgroup_root, f"monotonic_run_{name}_all", sc.cpus)
        procs = start_load(load_cg, sc.cpus, sc.loads, args.load_method)
        try:
            candidates: list[tuple[str, list[int], Path]] = [("基线", sc.cpus, run_all_cg)]
            for label, run_cpus in ordered_major_candidates(sc.cpus, sc.loads, args.candidate_mode):
                candidates.append((label, run_cpus, make_cgroup(args.cgroup_root, f"monotonic_run_{name}_{len(run_cpus)}c", run_cpus)))
            for label, run_cpus, run_cg in candidates:
                vals: list[float] = []
                for repeat in range(args.repeats):
                    row = run_bench(args.binary, args.model, run_cg, run_cpus, args.tokens, args.timeout_s)
                    row.update(
                        {
                            "scenario": name,
                            "config_id": sc.config_id,
                            "occupied_count": sc.occupied_count,
                            "loads": ",".join(map(str, sc.loads)),
                            "occupied_loads": "/".join(str(v) for v in sc.loads[: sc.occupied_count]),
                            "policy": label,
                            "run_cpus": cpu_spec(run_cpus),
                            "n_run_cpus": len(run_cpus),
                            "repeat": repeat,
                        }
                    )
                    if row["status"] == "ok":
                        vals.append(float(row["tok_s"]))
                    all_rows.append(row)
                if vals:
                    print(f"  {label}: {sum(vals)/len(vals):.3f} tok/s", flush=True)
            sc_rows = [r for r in all_rows if r["scenario"] == name and r["status"] == "ok"]
            base_vals = [float(r["tok_s"]) for r in sc_rows if r["policy"] == "基线"]
            baseline = sum(base_vals) / len(base_vals) if base_vals else None
            best_policy = "基线"
            best_tok = baseline if baseline is not None else 0.0
            best_cpus = cpu_spec(sc.cpus)
            for policy in sorted({r["policy"] for r in sc_rows if r["policy"] != "基线"}):
                vals = [float(r["tok_s"]) for r in sc_rows if r["policy"] == policy]
                avg = sum(vals) / len(vals) if vals else 0.0
                if avg > best_tok:
                    best_tok = avg
                    best_policy = policy
                    best_cpus = next(r["run_cpus"] for r in sc_rows if r["policy"] == policy)
            summary.append(
                {
                    "scenario": name,
                    "config_id": sc.config_id,
                    "occupied_count": sc.occupied_count,
                    "loads": ",".join(map(str, sc.loads)),
                    "occupied_loads": "/".join(str(v) for v in sc.loads[: sc.occupied_count]),
                    "baseline_tok_s": baseline if baseline is not None else "",
                    "selected_policy": best_policy,
                    "selected_cpus": best_cpus,
                    "selected_tok_s": best_tok,
                    "speedup": best_tok / baseline if baseline else "",
                }
            )
        finally:
            stop_load(load_cg, procs)
        write_csv(args.out_dir / "raw.csv", all_rows, RAW_FIELDS)
        write_csv(args.out_dir / "summary.csv", summary, SUMMARY_FIELDS)


RAW_FIELDS = [
    "scenario",
    "config_id",
    "occupied_count",
    "loads",
    "occupied_loads",
    "policy",
    "run_cpus",
    "n_run_cpus",
    "repeat",
    "status",
    "tok_s",
    "avg_decode_ms",
    "decode_ms",
    "elapsed_s",
    "output_tail",
]
SUMMARY_FIELDS = [
    "scenario",
    "config_id",
    "occupied_count",
    "loads",
    "occupied_loads",
    "baseline_tok_s",
    "selected_policy",
    "selected_cpus",
    "selected_tok_s",
    "speedup",
]


def plot(args: argparse.Namespace) -> None:
    configure_font()
    rows = list(csv.DictReader((args.out_dir / "summary.csv").open()))
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 12, "figure.dpi": 160, "savefig.dpi": 300})
    for occupied in [2, 4, 6, 8]:
        subset = sorted([r for r in rows if int(r["occupied_count"]) == occupied], key=lambda r: int(r["config_id"]))
        x = list(range(len(subset)))
        baseline = [float(r["baseline_tok_s"] or 0.0) for r in subset]
        schedule = [float(r["selected_tok_s"] or 0.0) for r in subset]
        labels = [f"Config{r['config_id']}" for r in subset]
        fig = plt.figure(figsize=(8.4, 4.0))
        grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.0, 0.82], hspace=0.30)
        ax = fig.add_subplot(grid[0])
        tax = fig.add_subplot(grid[1])
        width = 0.36
        ax.bar([i - width / 2 for i in x], baseline, width=width, label="基线", color="#8da0cb", edgecolor="#34495e", linewidth=0.6)
        bars = ax.bar([i + width / 2 for i in x], schedule, width=width, label="调度策略", color="#fc8d62", edgecolor="#7f3b20", linewidth=0.6)
        ymax = max(max(baseline), max(schedule))
        for bar, row in zip(bars, subset):
            ratio = f"{float(row['speedup']):.2f}x" if row.get("speedup") else "timeout"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.025, f"{row['selected_policy']}\\n{ratio}", ha="center", va="bottom", fontsize=7.8, color="#7f3b20")
        ax.set_title(f"8 核可用，{occupied} 核有占用")
        ax.set_ylabel("吞吐量 (tok/s)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, ymax * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.legend(loc="upper left", ncols=2, frameon=False)
        tax.axis("off")
        tax.text(0.0, 0.95, "各占用核心负载（%）：", ha="left", va="top", fontsize=9.3, fontweight="bold")
        for idx, row in enumerate(subset):
            col = idx // 3
            ridx = idx % 3
            tax.text(col * 0.34, 0.65 - ridx * 0.26, f"Config{row['config_id']}: {row['occupied_loads']}", ha="left", va="center", fontsize=8.5)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.05)
        fig.savefig(fig_dir / f"controlled_occupied_{occupied}c.png", bbox_inches="tight")
        fig.savefig(fig_dir / f"controlled_occupied_{occupied}c.pdf", bbox_inches="tight")
        plt.close(fig)

    # Cross-count sanity plot for monotonicity.
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for metric, style in [("baseline_tok_s", "-o"), ("selected_tok_s", "-s")]:
        for config_id in range(1, 7):
            vals = []
            for occupied in [2, 4, 6, 8]:
                row = next(r for r in rows if int(r["config_id"]) == config_id and int(r["occupied_count"]) == occupied)
                vals.append(float(row[metric] or 0.0))
            alpha = 0.35 if metric == "baseline_tok_s" else 0.75
            ax.plot([2, 4, 6, 8], vals, style, alpha=alpha, label=(f"Config{config_id} 基线" if metric == "baseline_tok_s" else f"Config{config_id} 调度"))
    ax.set_xlabel("有占用的核心数量")
    ax.set_ylabel("吞吐量 (tok/s)")
    ax.set_title("受控负载下吞吐量随占用核心数量变化")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(ncols=3, fontsize=7.2, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "controlled_monotonicity_check.png", bbox_inches="tight")
    fig.savefig(fig_dir / "controlled_monotonicity_check.pdf", bbox_inches="tight")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=EXP12 / "results/controlled_monotonic_major_only_20260506_r1")
    p.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    p.add_argument("--cpus", type=parse_cpu_list, default=list(range(60, 68)))
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--load-method", default="matrixprod")
    p.add_argument("--candidate-mode", choices=["idle-only", "all-prefixes"], default="idle-only")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if len(args.cpus) != 8:
        raise SystemExit(f"--cpus must contain exactly 8 CPUs, got {args.cpus}")
    if not args.plot_only:
        run(args)
    plot(args)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
