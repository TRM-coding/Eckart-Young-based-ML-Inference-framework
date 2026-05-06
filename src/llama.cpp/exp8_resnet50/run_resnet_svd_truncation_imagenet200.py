#!/usr/bin/env python3
"""Run ResNet50 SVD truncation sweep through the llama.cpp CPU path."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.io
from torchvision.models import ResNet50_Weights


ROOT = Path(__file__).resolve().parents[3]
EXP12 = ROOT / "src/llama.cpp/exp12_algorithms"
if str(EXP12) not in sys.path:
    sys.path.insert(0, str(EXP12))

from run_exp12_local import cpu_spec, parse_cpu_list  # noqa: E402


FORWARD_RE = re.compile(r"forward_ms_mean=(?P<mean>[0-9.eE+-]+).*?forward_ms_median=(?P<median>[0-9.eE+-]+).*?repeats=(?P<repeats>\d+)")
TOP1_RE = re.compile(r"^1\s+class_id=(?P<class_id>\d+)\s+", re.MULTILINE)


def read_val_ground_truth(devkit_tar: Path) -> dict[int, int]:
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
                raise RuntimeError(f"failed to map ImageNet class words to class id: {idx} {word_list}")
            idx_to_class_id[int(idx)] = matched

        gt_member = tf.getmember("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
        gt_raw = tf.extractfile(gt_member)
        if gt_raw is None:
            raise RuntimeError("failed to read ImageNet validation ground truth")
        val_ids = [int(line.strip()) for line in io.TextIOWrapper(gt_raw)]
    return {i + 1: idx_to_class_id[val_id] for i, val_id in enumerate(val_ids)}


def prepare_imagenet_subset(args: argparse.Namespace) -> list[tuple[Path, int]]:
    subset_dir = args.out_dir / "imagenet_val_subset_200"
    subset_dir.mkdir(parents=True, exist_ok=True)
    gt = read_val_ground_truth(args.imagenet_devkit)
    selected: list[tuple[str, int]] = []
    with tarfile.open(args.imagenet_val_tar, "r") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile() and m.name.endswith(".JPEG"))
        for name in names:
            idx = int(Path(name).stem.split("_")[-1])
            selected.append((name, gt[idx]))
            if len(selected) >= args.images:
                break
        existing = {p.name for p in subset_dir.glob("*.JPEG")}
        need = [name for name, _label in selected if Path(name).name not in existing]
        if need:
            tf.extractall(subset_dir, members=[tf.getmember(name) for name in need])
    rows = [(subset_dir / Path(name).name, label) for name, label in selected]
    with (args.out_dir / "imagenet200_subset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label0"])
        writer.writeheader()
        for image, label in rows:
            writer.writerow({"image": image, "label0": label})
    return rows


def run_cmd(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'build-release-current/bin'}:{env.get('LD_LIBRARY_PATH', '')}"
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        timeout=timeout_s,
    )


def inference_prefix(args: argparse.Namespace) -> list[str]:
    if args.no_taskset:
        return []
    return ["taskset", "-c", cpu_spec(args.cpus)]


def convert_svd_model(args: argparse.Namespace, truncation: float) -> Path:
    if truncation <= 0:
        return args.baseline_model
    keep_ratio = 1.0 - truncation
    out = args.out_dir / "models" / f"resnet50-svd-trunc{int(round(truncation * 100)):02d}-keep{int(round(keep_ratio * 100)):03d}-f32.gguf"
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
        f"{keep_ratio:.6g}",
        "--conv-svd-dtype",
        "f32",
    ]
    print(f"[convert] trunc={truncation:.1f} keep={keep_ratio:.1f} -> {out}", flush=True)
    run_cmd(cmd, args.convert_timeout_s).check_returncode()
    return out


def binary_for(args: argparse.Namespace, truncation: float) -> Path:
    return args.baseline_bin if truncation <= 0 else args.svd_bin


def model_for(models: dict[float, Path], truncation: float) -> Path:
    return models[truncation]


def classify_one(args: argparse.Namespace, image: Path, truncation: float, model: Path) -> tuple[str, int | None, str]:
    cmd = inference_prefix(args) + [
        str(binary_for(args, truncation)),
        "--model",
        str(model),
    ]
    if truncation > 0:
        cmd += ["--mode", args.svd_mode]
    cmd += ["--image", str(image), "--threads", str(args.threads), "--top-k", "1"]
    try:
        proc = run_cmd(cmd, args.classify_timeout_s)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return "timeout", None, "\n".join(out.splitlines()[-8:])
    match = TOP1_RE.search(proc.stdout)
    status = "ok" if proc.returncode == 0 and match else f"rc_{proc.returncode}"
    pred = int(match.group("class_id")) if match else None
    return status, pred, "\n".join(proc.stdout.splitlines()[-8:])


def evaluate_accuracy(args: argparse.Namespace, subset: list[tuple[Path, int]], truncation: float, model: Path) -> dict[str, Any]:
    correct = 0
    total = 0
    bad = 0
    t0 = time.time()
    pred_path = args.out_dir / f"predictions_trunc{int(round(truncation * 100)):02d}.csv"
    with pred_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label0", "pred", "correct", "status"])
        writer.writeheader()
        for i, (image, label) in enumerate(subset, start=1):
            status, pred, _tail = classify_one(args, image, truncation, model)
            ok = status == "ok" and pred is not None
            total += int(ok)
            bad += int(not ok)
            hit = ok and pred == label
            correct += int(hit)
            writer.writerow({"image": image, "label0": label, "pred": "" if pred is None else pred, "correct": int(hit), "status": status})
            if i % args.progress_every == 0 or i == len(subset):
                print(f"[acc] trunc={truncation:.1f} {i}/{len(subset)} correct={correct} valid={total} bad={bad}", flush=True)
    elapsed = time.time() - t0
    return {
        "truncation": truncation,
        "keep_ratio": 1.0 - truncation,
        "correct": correct,
        "total": total,
        "bad": bad,
        "top1": correct / total if total else 0.0,
        "elapsed_s": elapsed,
        "predictions": str(pred_path),
    }


def benchmark_speed(args: argparse.Namespace, truncation: float, model: Path, image: Path) -> dict[str, Any]:
    cmd = inference_prefix(args) + [
        str(binary_for(args, truncation)),
        "--model",
        str(model),
    ]
    if truncation > 0:
        cmd += ["--mode", args.svd_mode]
    cmd += [
        "--image",
        str(image),
        "--threads",
        str(args.threads),
        "--top-k",
        "5",
        "--benchmark",
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    try:
        proc = run_cmd(cmd, args.benchmark_timeout_s)
        text = proc.stdout
        match = FORWARD_RE.search(text)
        status = "ok" if proc.returncode == 0 and match else f"rc_{proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        text = exc.stdout or ""
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        match = None
        status = "timeout"
    row: dict[str, Any] = {
        "truncation": truncation,
        "keep_ratio": 1.0 - truncation,
        "status": status,
        "output_tail": "\n".join(text.splitlines()[-8:]),
    }
    if match:
        mean_ms = float(match.group("mean"))
        median_ms = float(match.group("median"))
        row.update(
            {
                "forward_ms_mean": mean_ms,
                "forward_ms_median": median_ms,
                "repeat": int(match.group("repeats")),
                "images_s": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            }
        )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_results(args: argparse.Namespace, merged_rows: list[dict[str, Any]]) -> tuple[Path, Path, Path, Path]:
    trunc = [float(r["truncation"]) for r in merged_rows]
    speedup = [float(r["speedup"]) for r in merged_rows]
    images_s = [float(r["images_s"]) for r in merged_rows]
    top1 = [float(r["top1"]) * 100.0 for r in merged_rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax2 = ax.twinx()
    l1 = ax.plot(trunc, speedup, marker="o", color="#1f77b4", linewidth=2.2, label="Speedup")
    l2 = ax2.plot(trunc, images_s, marker="s", color="#4d4d4d", linewidth=2.0, label="Throughput")
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xlabel("SVD truncation rate")
    ax.set_ylabel("Relative speedup")
    ax2.set_ylabel("Throughput (images/s)")
    ax.set_xticks(trunc)
    ax.set_ylim(0, max(speedup) * 1.18)
    ax2.set_ylim(0, max(images_s) * 1.18)
    ax.grid(axis="y", alpha=0.30)
    handles = l1 + l2
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper left")
    ax.set_title(f"ResNet50 llama.cpp CPU SVD speed on ImageNet-{args.images}")
    fig.tight_layout()
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    speed_png = fig_dir / "resnet50_svd_truncation_speed_imagenet200_llamacpp.png"
    speed_pdf = fig_dir / "resnet50_svd_truncation_speed_imagenet200_llamacpp.pdf"
    fig.savefig(speed_png, dpi=220)
    fig.savefig(speed_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rel_top1 = [v / max(top1[0], 1e-9) for v in top1]
    l1 = ax.plot(trunc, top1, marker="o", color="#2ca25f", linewidth=2.2, label="Top-1")
    ax2 = ax.twinx()
    l2 = ax2.plot(trunc, rel_top1, marker="^", color="#7b3294", linestyle="--", linewidth=1.8, label="Top-1 / baseline")
    ax.set_xlabel("SVD truncation rate")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax2.set_ylabel("Relative Top-1")
    ax.set_xticks(trunc)
    ax.set_ylim(0, max(top1) * 1.18 if top1 else 1)
    ax2.set_ylim(0, max(rel_top1) * 1.18 if rel_top1 else 1)
    ax.grid(axis="y", alpha=0.30)
    handles = l1 + l2
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right")
    ax.set_title(f"ResNet50 llama.cpp CPU SVD Top-1 on ImageNet-{args.images}")
    fig.tight_layout()
    acc_png = fig_dir / "resnet50_svd_truncation_accuracy_imagenet200_llamacpp.png"
    acc_pdf = fig_dir / "resnet50_svd_truncation_accuracy_imagenet200_llamacpp.pdf"
    fig.savefig(acc_png, dpi=220)
    fig.savefig(acc_pdf)
    plt.close(fig)
    return speed_png, speed_pdf, acc_png, acc_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "src/llama.cpp/exp8_resnet50/results/resnet_svd_truncation_imagenet200_llamacpp_20260506_r1")
    parser.add_argument("--cpus", default="0-7")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-taskset", action="store_true", help="Run CPU inference without setting affinity.")
    parser.add_argument("--baseline-bin", type=Path, default=ROOT / "build-release-current/run_resnet50")
    parser.add_argument("--svd-bin", type=Path, default=ROOT / "build-release-current/run_resnet50_conv_svd")
    parser.add_argument("--baseline-model", type=Path, default=ROOT / "src/llama.cpp/gguf_models/resnet50-f32.gguf")
    parser.add_argument("--hf-model-dir", type=Path, default=ROOT / "src/llama.cpp/gguf_models/resnet50_hf_model")
    parser.add_argument("--imagenet-val-tar", type=Path, default=Path("/SSD/ImageNet/ILSVRC2012_img_val.tar"))
    parser.add_argument("--imagenet-devkit", type=Path, default=Path("/SSD/ImageNet/ILSVRC2012_devkit_t12.tar.gz"))
    parser.add_argument("--python", type=Path, default=Path("/home/tianruiming/miniconda3/envs/pytorch/bin/python"))
    parser.add_argument("--truncations", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--svd-mode", choices=["fold", "im2col"], default="fold")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--classify-timeout-s", type=float, default=60.0)
    parser.add_argument("--benchmark-timeout-s", type=float, default=240.0)
    parser.add_argument("--convert-timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    args.cpus = parse_cpu_list(args.cpus)
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    truncations = [float(x) for x in args.truncations.split(",") if x.strip()]
    metadata = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "cpus": args.cpus,
        "taskset": not args.no_taskset,
        "threads": args.threads,
        "truncations": truncations,
        "images": args.images,
        "path": "llama.cpp CPU binaries only; no PyTorch/GPU inference is used for metrics",
        "baseline_bin": str(args.baseline_bin),
        "svd_bin": str(args.svd_bin),
        "baseline_model": str(args.baseline_model),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    subset = prepare_imagenet_subset(args)
    models = {trunc: convert_svd_model(args, trunc) for trunc in truncations}

    speed_rows: list[dict[str, Any]] = []
    acc_rows: list[dict[str, Any]] = []
    bench_image = subset[0][0]
    for trunc in truncations:
        model = model_for(models, trunc)
        print(f"[bench] trunc={trunc:.1f} model={model}", flush=True)
        speed_rows.append(benchmark_speed(args, trunc, model, bench_image))
        write_csv(args.out_dir / "speed_raw.csv", speed_rows)
        print(f"[accuracy] trunc={trunc:.1f}", flush=True)
        acc_rows.append(evaluate_accuracy(args, subset, trunc, model))
        write_csv(args.out_dir / "accuracy_raw.csv", acc_rows)

    bad_speed = [r for r in speed_rows if r.get("status") != "ok" or "images_s" not in r]
    if bad_speed:
        print("[error] speed benchmark failed; see speed_raw.csv for output_tail", file=sys.stderr)
        for row in bad_speed:
            print(f"  trunc={row['truncation']} status={row['status']}", file=sys.stderr)
        return 2
    base_images_s = next(float(r["images_s"]) for r in speed_rows if float(r["truncation"]) == 0.0)
    merged_rows: list[dict[str, Any]] = []
    acc_by_trunc = {float(r["truncation"]): r for r in acc_rows}
    for row in speed_rows:
        trunc = float(row["truncation"])
        acc = acc_by_trunc[trunc]
        merged = {
            "truncation": trunc,
            "keep_ratio": 1.0 - trunc,
            "status": row["status"],
            "forward_ms_mean": row.get("forward_ms_mean", ""),
            "forward_ms_median": row.get("forward_ms_median", ""),
            "images_s": row.get("images_s", 0.0),
            "speedup": float(row.get("images_s", 0.0)) / base_images_s if base_images_s else 0.0,
            "correct": acc["correct"],
            "total": acc["total"],
            "bad": acc["bad"],
            "top1": acc["top1"],
        }
        merged_rows.append(merged)
    write_csv(args.out_dir / "truncation_sweep_summary.csv", merged_rows)
    speed_png, speed_pdf, acc_png, acc_pdf = plot_results(args, merged_rows)
    print(args.out_dir)
    print(speed_png)
    print(speed_pdf)
    print(acc_png)
    print(acc_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
