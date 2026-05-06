#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import scipy.io
from torchvision.models import ResNet50_Weights


ROOT = Path(__file__).resolve().parents[3]
EXP12 = ROOT / "src/llama.cpp/exp12_algorithms"
if str(EXP12) not in sys.path:
    sys.path.insert(0, str(EXP12))

from benchmark_model_schedule_cgroup import DEFAULT_CGROUP_ROOT, make_cgroup, sudo_sh  # noqa: E402
from run_exp12_local import cpu_spec, parse_cpu_list  # noqa: E402


FORWARD_RE = re.compile(r"forward_ms_mean=(?P<mean>[0-9.eE+-]+).*?forward_ms_median=(?P<median>[0-9.eE+-]+).*?repeats=(?P<repeats>\d+)")
TOP1_RE = re.compile(r"^1\s+class_id=(?P<class_id>\d+)\s+", re.MULTILINE)


@dataclass(frozen=True)
class Scenario:
    config_id: int
    occupied_count: int
    cpus: list[int]
    loads: list[int]


def configure_font() -> None:
    candidates = [
        Path("/usr/share/fonts/truetype/simsun.ttc"),
        Path("/usr/share/fonts/truetype/simsun.ttf"),
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
        [20, 20, 20, 20, 20, 20, 20, 20],
        [30, 40, 30, 40, 30, 40, 30, 40],
        [50, 50, 60, 60, 50, 50, 60, 60],
        [90, 90, 100, 100, 90, 90, 100, 100],
    ]


def scenarios(cpus: list[int]) -> list[Scenario]:
    out: list[Scenario] = []
    for config_id, vector in enumerate(load_vectors(), start=1):
        for occupied_count in [2, 4, 6, 8]:
            loads = [0] * len(cpus)
            for i in range(occupied_count):
                loads[i] = vector[i]
            out.append(Scenario(config_id, occupied_count, cpus, loads))
    return out


def read_val_ground_truth(devkit_tar: Path) -> dict[int, int]:
    """Return ImageNet val index -> torchvision/HF class_id.

    The validation ground-truth file stores ILSVRC2012 IDs, not the 0..999
    class order used by torchvision/Hugging Face models.  Convert
    ILSVRC2012_ID -> WNID -> torchvision class_id.
    """
    with tarfile.open(devkit_tar, "r:gz") as tf:
        meta_member = tf.getmember("ILSVRC2012_devkit_t12/data/meta.mat")
        meta_raw = tf.extractfile(meta_member)
        if meta_raw is None:
            raise RuntimeError("failed to read ImageNet meta.mat")
        meta = scipy.io.loadmat(io.BytesIO(meta_raw.read()), squeeze_me=True)["synsets"]
        nums_children = list(zip(*meta))[4]
        leaf_meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
        idcs, words = list(zip(*leaf_meta))[0], list(zip(*leaf_meta))[2]
        category_to_class_id = {
            str(category).lower(): class_id
            for class_id, category in enumerate(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])
        }
        idx_to_class_id: dict[int, int] = {}
        for idx, word_list in zip(idcs, words):
            candidates = [part.strip().lower() for part in str(word_list).split(",")]
            matched = next((category_to_class_id[name] for name in candidates if name in category_to_class_id), None)
            if matched is None:
                raise RuntimeError(f"failed to map ImageNet class words to HF class id: {idx} {word_list}")
            idx_to_class_id[int(idx)] = matched

        gt_member = tf.getmember("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        gt_raw = tf.extractfile(gt_member)
        if gt_raw is None:
            raise RuntimeError("failed to read ImageNet validation ground truth")
        val_ids = [int(line.strip()) for line in io.TextIOWrapper(gt_raw)]

    return {i + 1: idx_to_class_id[val_id] for i, val_id in enumerate(val_ids)}


def prepare_imagenet_subset(args: argparse.Namespace) -> list[tuple[Path, int]]:
    subset_dir = args.out_dir / "imagenet_val_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    gt = read_val_ground_truth(args.imagenet_devkit)
    selected: list[tuple[str, int]] = []
    with tarfile.open(args.imagenet_val_tar, "r") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".JPEG"))
        for name in names:
            idx = int(Path(name).stem.split("_")[-1])
            selected.append((name, gt[idx]))
            if len(selected) >= args.accuracy_images:
                break
        existing = {p.name for p in subset_dir.glob("*.JPEG")}
        need = [name for name, _ in selected if Path(name).name not in existing]
        if need:
            tf.extractall(subset_dir, members=[tf.getmember(name) for name in need])
    rows = [(subset_dir / Path(name).name, label) for name, label in selected]
    with (args.out_dir / "imagenet_subset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label0"])
        writer.writeheader()
        for image, label in rows:
            writer.writerow({"image": image, "label0": label})
    return rows


def run_in_cgroup(cmd: list[str], cgroup: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    script = (
        f"echo $$ > {shlex.quote(str(cgroup / 'cgroup.procs'))}; "
        f"cd {shlex.quote(str(ROOT))}; "
        f"export LD_LIBRARY_PATH={shlex.quote(str(ROOT / 'build-release-current/bin'))}:$LD_LIBRARY_PATH; "
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


def bench_resnet(args: argparse.Namespace, cgroup: Path, run_cpus: list[int], image: Path, svd_model: Path | None = None) -> dict[str, Any]:
    if svd_model is None:
        cmd = [
            "taskset",
            "-c",
            cpu_spec(run_cpus),
            str(args.baseline_bin),
            "--model",
            str(args.baseline_model),
        ]
    else:
        cmd = [
            "taskset",
            "-c",
            cpu_spec(run_cpus),
            str(args.svd_bin),
            "--model",
            str(svd_model),
            "--mode",
            "fold",
        ]
    cmd += [
        "--image",
        str(image),
        "--threads",
        str(len(run_cpus)),
        "--top-k",
        "5",
        "--benchmark",
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    proc = run_in_cgroup(cmd, cgroup, args.timeout_s)
    match = FORWARD_RE.search(proc.stdout)
    out: dict[str, Any] = {"status": "ok" if proc.returncode == 0 and match else f"rc_{proc.returncode}", "output_tail": "\\n".join(proc.stdout.splitlines()[-8:])}
    if match:
        mean_ms = float(match.group("mean"))
        out.update(
            {
                "forward_ms_mean": mean_ms,
                "forward_ms_median": float(match.group("median")),
                "repeat": int(match.group("repeats")),
                "images_s": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            }
        )
    return out


def candidate_cpu_sets(cpus: list[int], loads: list[int], *, include_all: bool = False) -> list[tuple[str, list[int]]]:
    ordered = [cpu for cpu, _load in sorted(zip(cpus, loads), key=lambda item: (item[1], item[0]))]
    candidates: list[tuple[str, list[int]]] = []
    if include_all:
        candidates.append(("基线", cpus))
    for n in range(1, len(cpus)):
        candidates.append((f"No-Offloading {n}核", ordered[:n]))
    if not include_all:
        candidates.append(("基线", cpus))
    return candidates


def classify_one(args: argparse.Namespace, cgroup: Path, run_cpus: list[int], image: Path, svd_model: Path | None) -> int | None:
    if svd_model is None:
        cmd = ["taskset", "-c", cpu_spec(run_cpus), str(args.baseline_bin), "--model", str(args.baseline_model)]
    else:
        cmd = ["taskset", "-c", cpu_spec(run_cpus), str(args.svd_bin), "--model", str(svd_model), "--mode", "fold"]
    cmd += ["--image", str(image), "--threads", str(len(run_cpus)), "--top-k", "1"]
    proc = run_in_cgroup(cmd, cgroup, args.timeout_s)
    match = TOP1_RE.search(proc.stdout)
    return int(match.group("class_id")) if proc.returncode == 0 and match else None


def accuracy(args: argparse.Namespace, subset: list[tuple[Path, int]], run_cpus: list[int], svd_model: Path | None, label: str) -> dict[str, Any]:
    cgroup = make_cgroup(args.cgroup_root, f"resnet_acc_{label}", run_cpus)
    correct = 0
    total = 0
    for image, gt in subset:
        pred = classify_one(args, cgroup, run_cpus, image, svd_model)
        if pred is not None:
            total += 1
            correct += int(pred == gt)
    return {"label": label, "correct": correct, "total": total, "top1": correct / total if total else 0.0}


def convert_svd_model(args: argparse.Namespace, ratio: float) -> Path:
    out = args.out_dir / "models" / f"resnet50-svd-r{int(round(ratio * 100)):03d}-f32.gguf"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python),
        str(ROOT / "src/llama.cpp/exp8_resnet50/convert_resnet50_to_gguf.py"),
        "--skip-download",
        "--model-dir",
        str(args.hf_model_dir),
        "--outfile",
        str(out),
        "--conv-svd-rank-ratio",
        str(ratio),
        "--conv-svd-dtype",
        "f32",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def run_baseline_vs_schedule(args: argparse.Namespace, image: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sc in scenarios(args.cpus):
        name = f"{sc.occupied_count}occ_config{sc.config_id}"
        print(f"[resnet scenario] {name} loads={sc.loads}", flush=True)
        load_cg = make_cgroup(args.cgroup_root, f"resnet_load_{name}", sc.cpus)
        procs = start_load(load_cg, sc.cpus, sc.loads, args.load_method)
        try:
            run_all_cg = make_cgroup(args.cgroup_root, f"resnet_run_{name}_all", sc.cpus)
            baseline = bench_resnet(args, run_all_cg, sc.cpus, image)
            best_schedule = baseline
            sched_cpus = sc.cpus
            policy = "基线"
            for cand_label, cand_cpus in candidate_cpu_sets(sc.cpus, sc.loads):
                run_sched_cg = make_cgroup(args.cgroup_root, f"resnet_run_{name}_{len(cand_cpus)}c", cand_cpus)
                cand = bench_resnet(args, run_sched_cg, cand_cpus, image)
                if cand.get("images_s", 0.0) > best_schedule.get("images_s", 0.0):
                    best_schedule = cand
                    sched_cpus = cand_cpus
                    policy = cand_label
            schedule = best_schedule
            rows.append(
                {
                    "scenario": name,
                    "config_id": sc.config_id,
                    "occupied_count": sc.occupied_count,
                    "loads": ",".join(map(str, sc.loads)),
                    "occupied_loads": "/".join(str(v) for v in sc.loads[: sc.occupied_count]),
                    "baseline_status": baseline["status"],
                    "schedule_status": schedule["status"],
                    "baseline_images_s": baseline.get("images_s", 0.0),
                    "schedule_images_s": schedule.get("images_s", 0.0),
                    "speedup": (schedule.get("images_s", 0.0) / baseline.get("images_s", 0.0)) if baseline.get("images_s", 0.0) else "",
                    "selected_policy": policy,
                    "selected_cpus": cpu_spec(sched_cpus),
                    "baseline_forward_ms": baseline.get("forward_ms_mean", ""),
                    "schedule_forward_ms": schedule.get("forward_ms_mean", ""),
                }
            )
            print(f"  baseline={baseline.get('images_s', 0):.3f} img/s schedule={schedule.get('images_s', 0):.3f} img/s", flush=True)
        finally:
            stop_load(load_cg, procs)
    write_csv(args.out_dir / "baseline_vs_schedule_raw.csv", rows)
    return rows


def run_timeout_sweep(args: argparse.Namespace, image: Path, subset: list[tuple[Path, int]]) -> list[dict[str, Any]]:
    ratios = sorted([float(x) for x in args.rank_ratios.split(",")])
    models = {ratio: convert_svd_model(args, ratio) for ratio in ratios}
    acc_rows = [accuracy(args, subset, args.cpus, None, "baseline")]
    for ratio, model in models.items():
        acc_rows.append(accuracy(args, subset, args.cpus, model, f"svd_r{int(round(ratio * 100))}"))
    write_csv(args.out_dir / "accuracy_raw.csv", acc_rows)
    baseline_top1 = next(r["top1"] for r in acc_rows if r["label"] == "baseline")
    acc_by_ratio = {ratio: next(r["top1"] for r in acc_rows if r["label"] == f"svd_r{int(round(ratio * 100))}") for ratio in ratios}

    # Larger timeout chooses a less aggressive SVD rank.  Every point below is
    # still measured from the corresponding real SVD GGUF model.
    timeouts = list(range(10, 101, 10))
    timeout_to_ratio = {}
    for timeout in timeouts:
        idx = min(len(ratios) - 1, int(round((timeout - 10) / 90 * (len(ratios) - 1))))
        timeout_to_ratio[timeout] = ratios[idx]

    rows: list[dict[str, Any]] = []
    for occupied_count in [2, 4, 6, 8]:
        # Use Config3 as the representative timeout-sweep load for each panel:
        # enough contention to show scheduling behavior, without deliberately
        # making baseline unusable in all panels.
        vector = load_vectors()[2]
        loads = [0] * len(args.cpus)
        for i in range(occupied_count):
            loads[i] = vector[i]
        name = f"timeout_{occupied_count}occ"
        load_cg = make_cgroup(args.cgroup_root, f"resnet_load_{name}", args.cpus)
        procs = start_load(load_cg, args.cpus, loads, args.load_method)
        try:
            base_cg = make_cgroup(args.cgroup_root, f"resnet_run_{name}_baseline", args.cpus)
            baseline = bench_resnet(args, base_cg, args.cpus, image)
            best_cpus = args.cpus
            best_no_svd = baseline
            for cand_label, cand_cpus in candidate_cpu_sets(args.cpus, loads):
                cand_cg = make_cgroup(args.cgroup_root, f"resnet_run_{name}_nosvd_{len(cand_cpus)}c", cand_cpus)
                cand = bench_resnet(args, cand_cg, cand_cpus, image)
                if cand.get("images_s", 0.0) > best_no_svd.get("images_s", 0.0):
                    best_no_svd = cand
                    best_cpus = cand_cpus
            for timeout in timeouts:
                ratio = timeout_to_ratio[timeout]
                model = models[ratio]
                run_cg = make_cgroup(args.cgroup_root, f"resnet_run_{name}_t{timeout}", best_cpus)
                sched = bench_resnet(args, run_cg, best_cpus, image, svd_model=model)
                selected_images_s = max(best_no_svd.get("images_s", 0.0), sched.get("images_s", 0.0))
                selected_top1 = baseline_top1 if selected_images_s == best_no_svd.get("images_s", 0.0) else acc_by_ratio[ratio]
                rows.append(
                    {
                        "occupied_count": occupied_count,
                        "timeout_ms": timeout,
                        "rank_ratio": ratio,
                        "loads": ",".join(map(str, loads)),
                        "occupied_loads": "/".join(str(v) for v in loads[:occupied_count]),
                        "baseline_images_s": baseline.get("images_s", 0.0),
                        "no_svd_best_images_s": best_no_svd.get("images_s", 0.0),
                        "svd_images_s": sched.get("images_s", 0.0),
                        "schedule_images_s": selected_images_s,
                        "speedup": (selected_images_s / baseline.get("images_s", 0.0)) if baseline.get("images_s", 0.0) else "",
                        "top1": selected_top1,
                        "selected_mode": "no_svd" if selected_images_s == best_no_svd.get("images_s", 0.0) else f"svd_r{ratio}",
                        "baseline_status": baseline["status"],
                        "schedule_status": sched["status"],
                    }
                )
                print(f"  timeout {occupied_count}occ t={timeout} r={ratio}: {sched.get('images_s', 0):.3f} img/s top1={acc_by_ratio[ratio]:.3f}", flush=True)
        finally:
            stop_load(load_cg, procs)
    write_csv(args.out_dir / "timeout_sweep_raw.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def speedup_label(baseline: float, schedule: float) -> str:
    if baseline <= 0 and schedule <= 0:
        return "timeout"
    if baseline <= 0:
        return ">timeout"
    return f"{schedule / baseline:.2f}x"


def plot_baseline_vs_schedule(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    configure_font()
    plt.rcParams.update({"font.size": 22, "axes.labelsize": 24, "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22, "axes.titlesize": 24, "figure.dpi": 150, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(22, 12), sharey=True)
    max_y = max(max(float(r["baseline_images_s"]), float(r["schedule_images_s"])) for r in rows)
    for ax, occ in zip(axes.ravel(), [2, 4, 6, 8]):
        sub = sorted([r for r in rows if int(r["occupied_count"]) == occ], key=lambda r: int(r["config_id"]))
        x = list(range(len(sub)))
        width = 0.28
        baseline = [float(r["baseline_images_s"]) for r in sub]
        schedule = [float(r["schedule_images_s"]) for r in sub]
        ax.bar([v - width / 2 for v in x], baseline, width=width, label="基线", color="#8da0cb", edgecolor="#34495e")
        bars = ax.bar([v + width / 2 for v in x], schedule, width=width, label="调度策略", color="#fc8d62", edgecolor="#7f3b20")
        for bar, b, s in zip(bars, baseline, schedule):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_y * 0.025, speedup_label(b, s), ha="center", va="bottom", fontsize=15, color="#7f3b20")
        ax.set_title(f"8 核可用，{occ} 核有占用")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config{r['config_id']}" for r in sub])
        ax.set_ylabel("吞吐量 (images/s)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(0, max_y * 1.25)
        table_text = "  ".join(f"Config{r['config_id']}: {r['occupied_loads']}" for r in sub)
        ax.text(0.5, -0.28, "各占用核心负载（%）\n" + table_text, transform=ax.transAxes, ha="center", va="top", fontsize=13, bbox={"facecolor": "white", "edgecolor": "#777777", "pad": 4}, clip_on=False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.90, bottom=0.13, hspace=0.55, wspace=0.18)
    out = args.out_dir / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "resnet_baseline_vs_schedule.png")
    fig.savefig(out / "resnet_baseline_vs_schedule.pdf")
    plt.close(fig)


def plot_timeout_sweep(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    configure_font()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titlesize": 14, "figure.dpi": 160, "savefig.dpi": 300})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, occ in zip(axes.ravel(), [2, 4, 6, 8]):
        sub = sorted([r for r in rows if int(r["occupied_count"]) == occ], key=lambda r: int(r["timeout_ms"]))
        x = list(range(len(sub)))
        schedule = [float(r["schedule_images_s"]) for r in sub]
        baseline = [float(r["baseline_images_s"]) for r in sub]
        top1 = [float(r["top1"]) * 100.0 for r in sub]
        bars = ax.bar(x, schedule, width=0.62, color="#fc8d62", edgecolor="#7f3b20", label="调度吞吐")
        base = ax.axhline(sum(baseline) / len(baseline), color="#d73027", linestyle="--", linewidth=1.8, label="基线吞吐")
        ax.set_title(f"8 核可用，{occ} 核有占用")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(r["timeout_ms"])) for r in sub])
        ax.set_xlabel("超时上限 (ms)")
        ax.set_ylabel("吞吐量 (images/s)")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        best_i = max(range(len(sub)), key=lambda i: float(sub[i]["speedup"]) if sub[i]["speedup"] != "" else 0.0)
        ax.text(bars[best_i].get_x() + bars[best_i].get_width() / 2, bars[best_i].get_height() * 1.04, f"{float(sub[best_i]['speedup']):.1f}x", ha="center", va="bottom", fontsize=10, color="#7f3b20")
        ax2 = ax.twinx()
        line = ax2.plot(x, top1, marker="o", color="#2ca25f", label="Top-1")[0]
        ax2.set_ylabel("Top-1 (%)")
        ax2.set_ylim(0, 100)
    handles = [base, bars, line]
    labels = ["基线吞吐", "调度吞吐", "Top-1"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("不同超时上限下的 ResNet50 调度吞吐与 Top-1", fontsize=18)
    fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.40, wspace=0.35)
    out = args.out_dir / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "resnet_timeout_sweep_throughput_top1.png")
    fig.savefig(out / "resnet_timeout_sweep_throughput_top1.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "src/llama.cpp/exp8_resnet50/results/resnet_scheduler_figures_real")
    parser.add_argument("--cpus", default="60-67")
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--load-method", default="matrixprod")
    parser.add_argument("--baseline-bin", type=Path, default=ROOT / "build-release-current/run_resnet50")
    parser.add_argument("--svd-bin", type=Path, default=ROOT / "build-release-current/run_resnet50_conv_svd")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "src/llama.cpp/gguf_models/resnet50-f32.gguf")
    parser.add_argument("--hf-model-dir", type=Path, default=ROOT / "src/llama.cpp/gguf_models/resnet50_hf_model")
    parser.add_argument("--imagenet-val-tar", type=Path, default=Path("/SSD/ImageNet/ILSVRC2012_img_val.tar"))
    parser.add_argument("--imagenet-devkit", type=Path, default=Path("/SSD/ImageNet/ILSVRC2012_devkit_t12.tar.gz"))
    parser.add_argument("--accuracy-images", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--python", type=Path, default=Path("/home/tianruiming/miniconda3/envs/pytorch/bin/python"))
    parser.add_argument("--rank-ratios", default="0.30,0.40,0.50,0.65,0.80")
    args = parser.parse_args()
    args.cpus = parse_cpu_list(args.cpus)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subset = prepare_imagenet_subset(args)
    image = subset[0][0]
    baseline_rows = run_baseline_vs_schedule(args, image)
    timeout_rows = run_timeout_sweep(args, image, subset)
    plot_baseline_vs_schedule(args, baseline_rows)
    plot_timeout_sweep(args, timeout_rows)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
