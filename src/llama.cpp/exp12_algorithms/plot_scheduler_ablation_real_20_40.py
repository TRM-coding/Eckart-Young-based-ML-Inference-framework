#!/usr/bin/env python3
"""Plot the verified 20-40 ms real scheduler ablation data."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXP12 = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = EXP12 / "results/scheduler_ablation_real_6067_20260506_r1"
TIMEOUTS = [20, 30, 40]
SCENARIOS = [
    ("2occ_minor_hot", "2 occupied cores"),
    ("4occ_tail_hot", "4 occupied cores"),
    ("6occ_tail_hot", "6 occupied cores"),
    ("8occ_high", "8 occupied cores"),
]
ABLATIONS = [
    ("A1_random_input", "A1: random mini-batch input"),
    ("A2_random_pruning", "A2: random per-layer pruning"),
    ("A3_average_timeout", "A3: average timeout allocation"),
    ("A4_no_scheduler", "A4: no scheduler"),
]
POLICY_LABELS = {
    "scheduler": "Scheduler",
    "A1_random_input": "Random input",
    "A2_random_pruning": "Random pruning",
    "A3_average_timeout": "Average timeout",
    "A4_no_scheduler": "No scheduler",
}
POLICY_NOTES = {
    "A1_random_input": "random mini-batch input, DP pruning and weighted timeout kept",
    "A2_random_pruning": "constructed mini-batch input, randomized per-layer pruning, weighted timeout kept",
    "A3_average_timeout": "constructed mini-batch input and DP pruning kept, timeout evenly allocated",
    "A4_no_scheduler": "random input, random pruning, and average timeout allocation",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def require_ok(rows: list[dict[str, str]], name: str) -> None:
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        sample = bad[0]
        raise SystemExit(f"{name} contains non-ok rows, first bad row: {sample}")


def load_data(result_dir: Path) -> tuple[dict[tuple[str, int], float], dict[tuple[str, str, int], dict[str, Any]]]:
    ppl_rows = [r for r in read_csv(result_dir / "ppl_raw.csv") if int(r["timeout_ms"]) in TIMEOUTS]
    decode_rows = [r for r in read_csv(result_dir / "decode_raw.csv") if int(r["timeout_ms"]) in TIMEOUTS]
    require_ok(ppl_rows, "ppl_raw.csv")
    require_ok(decode_rows, "decode_raw.csv")

    expected_ppl = 5 * len(TIMEOUTS)
    expected_decode = 5 * len(TIMEOUTS) * len(SCENARIOS)
    if len(ppl_rows) != expected_ppl:
        raise SystemExit(f"Expected {expected_ppl} verified PPL rows for 20-40 ms, got {len(ppl_rows)}")
    if len(decode_rows) != expected_decode:
        raise SystemExit(f"Expected {expected_decode} verified decode rows for 20-40 ms, got {len(decode_rows)}")

    ppl: dict[tuple[str, int], float] = {}
    for row in ppl_rows:
        key = (row["policy"], int(row["timeout_ms"]))
        ppl[key] = float(row["ppl"])

    decode: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in decode_rows:
        key = (row["scenario"], row["policy"], int(row["timeout_ms"]))
        decode[key] = {
            "tok_s": float(row["decode_tok_s"]),
            "generation_decode_ms": float(row["generation_decode_ms"]),
            "occupied_count": int(row["occupied_count"]),
            "loads": row["loads"],
            "workers": row["workers"],
            "log": row["log"],
        }
    return ppl, decode


def write_long_csv(result_dir: Path, ppl: dict[tuple[str, int], float], decode: dict[tuple[str, str, int], dict[str, Any]]) -> Path:
    out = result_dir / "ablation_real_20_40_long.csv"
    fields = [
        "timeout_ms",
        "scenario",
        "occupied_count",
        "policy",
        "policy_label",
        "decode_tok_s",
        "generation_decode_ms",
        "ppl",
        "loads",
        "workers",
        "log",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for timeout in TIMEOUTS:
            for scenario, _ in SCENARIOS:
                for policy in ["scheduler"] + [p for p, _ in ABLATIONS]:
                    row = decode[(scenario, policy, timeout)]
                    writer.writerow(
                        {
                            "timeout_ms": timeout,
                            "scenario": scenario,
                            "occupied_count": row["occupied_count"],
                            "policy": policy,
                            "policy_label": POLICY_LABELS[policy],
                            "decode_tok_s": f"{row['tok_s']:.6g}",
                            "generation_decode_ms": f"{row['generation_decode_ms']:.6g}",
                            "ppl": f"{ppl[(policy, timeout)]:.6g}",
                            "loads": row["loads"],
                            "workers": row["workers"],
                            "log": row["log"],
                        }
                    )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def write_summary_csv(result_dir: Path, ppl: dict[tuple[str, int], float], decode: dict[tuple[str, str, int], dict[str, Any]]) -> Path:
    out = result_dir / "ablation_real_20_40_summary.csv"
    fields = [
        "ablation",
        "scenario",
        "occupied_count",
        "scheduler_tok_s_mean",
        "ablation_tok_s_mean",
        "tok_s_gap_mean",
        "scheduler_ppl_mean",
        "ablation_ppl_mean",
        "ppl_gap_mean",
        "scheduler_ppl_max",
        "ablation_ppl_min",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ablation, _ in ABLATIONS:
            for scenario, _title in SCENARIOS:
                occupied = decode[(scenario, "scheduler", TIMEOUTS[0])]["occupied_count"]
                sched_tok = [decode[(scenario, "scheduler", t)]["tok_s"] for t in TIMEOUTS]
                abl_tok = [decode[(scenario, ablation, t)]["tok_s"] for t in TIMEOUTS]
                sched_ppl = [ppl[("scheduler", t)] for t in TIMEOUTS]
                abl_ppl = [ppl[(ablation, t)] for t in TIMEOUTS]
                writer.writerow(
                    {
                        "ablation": ablation,
                        "scenario": scenario,
                        "occupied_count": occupied,
                        "scheduler_tok_s_mean": f"{mean(sched_tok):.6g}",
                        "ablation_tok_s_mean": f"{mean(abl_tok):.6g}",
                        "tok_s_gap_mean": f"{mean(sched_tok) - mean(abl_tok):.6g}",
                        "scheduler_ppl_mean": f"{mean(sched_ppl):.6g}",
                        "ablation_ppl_mean": f"{mean(abl_ppl):.6g}",
                        "ppl_gap_mean": f"{mean(abl_ppl) - mean(sched_ppl):.6g}",
                        "scheduler_ppl_max": f"{max(sched_ppl):.6g}",
                        "ablation_ppl_min": f"{min(abl_ppl):.6g}",
                    }
                )
    return out


def style_axes(ax: plt.Axes, ax2: plt.Axes) -> None:
    ax.set_xticks(TIMEOUTS)
    ax.set_xlabel("Timeout budget (ms)")
    ax.set_ylabel("Decode throughput (tok/s, log)")
    ax2.set_ylabel("PPL")
    ax.grid(True, axis="both", color="#d7dde5", linewidth=0.8, alpha=0.8)
    ax.set_yscale("log")
    ax.tick_params(labelsize=9)
    ax2.tick_params(labelsize=9)


def plot_one_ablation(
    result_dir: Path,
    figures_dir: Path,
    ablation: str,
    ablation_title: str,
    ppl: dict[tuple[str, int], float],
    decode: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    axes_flat = list(axes.ravel())

    sched_color = "#2ca25f"
    abl_color = "#756bb1"
    for ax, (scenario, scenario_title) in zip(axes_flat, SCENARIOS):
        sched_tok = [decode[(scenario, "scheduler", t)]["tok_s"] for t in TIMEOUTS]
        abl_tok = [decode[(scenario, ablation, t)]["tok_s"] for t in TIMEOUTS]
        sched_ppl = [ppl[("scheduler", t)] for t in TIMEOUTS]
        abl_ppl = [ppl[(ablation, t)] for t in TIMEOUTS]

        ax.plot(TIMEOUTS, sched_tok, marker="o", color=sched_color, linewidth=2.2)
        ax.plot(TIMEOUTS, abl_tok, marker="s", color=abl_color, linewidth=2.2)
        ax2 = ax.twinx()
        ax2.plot(TIMEOUTS, sched_ppl, marker="o", color=sched_color, linestyle="--", linewidth=1.8)
        ax2.plot(TIMEOUTS, abl_ppl, marker="s", color=abl_color, linestyle="--", linewidth=1.8)

        ax.set_title(scenario_title, fontsize=11, fontweight="bold")
        ymin = min(sched_tok + abl_tok)
        ymax = max(sched_tok + abl_tok)
        ax.set_ylim(max(ymin * 0.65, 0.05), ymax * 1.55)
        ax2.set_ylim(0, max(sched_ppl + abl_ppl) * 1.28)
        style_axes(ax, ax2)

    handles = [
        Line2D([0], [0], color=sched_color, marker="o", linewidth=2.2, linestyle="-", label="Scheduler throughput (left axis)"),
        Line2D([0], [0], color=abl_color, marker="s", linewidth=2.2, linestyle="-", label=f"{POLICY_LABELS[ablation]} throughput (left axis)"),
        Line2D([0], [0], color=sched_color, marker="o", linewidth=1.8, linestyle="--", label="Scheduler PPL (right axis)"),
        Line2D([0], [0], color=abl_color, marker="s", linewidth=1.8, linestyle="--", label=f"{POLICY_LABELS[ablation]} PPL (right axis)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"Scheduler Ablation ({ablation_title}), verified real runs at 20-40 ms", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03, "Solid lines use the left y-axis for throughput; dashed lines use the right y-axis for PPL. Source rows: status=ok only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.93))

    png = figures_dir / f"{ablation}_real_20_40.png"
    pdf = figures_dir / f"{ablation}_real_20_40.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_overview(result_dir: Path, figures_dir: Path, ppl: dict[tuple[str, int], float], decode: dict[tuple[str, str, int], dict[str, Any]]) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {
        "scheduler": "#2ca25f",
        "A1_random_input": "#3182bd",
        "A2_random_pruning": "#e6550d",
        "A3_average_timeout": "#756bb1",
        "A4_no_scheduler": "#636363",
    }
    policies = ["scheduler"] + [p for p, _ in ABLATIONS]

    overview_handles = []
    for policy in policies:
        avg_tok = []
        for timeout in TIMEOUTS:
            vals = [decode[(scenario, policy, timeout)]["tok_s"] for scenario, _ in SCENARIOS]
            avg_tok.append(mean(vals))
        line = axes[0].plot(TIMEOUTS, avg_tok, marker="o", linewidth=2.1, color=colors[policy], label=POLICY_LABELS[policy])[0]
        axes[1].plot(TIMEOUTS, [ppl[(policy, t)] for t in TIMEOUTS], marker="o", linewidth=2.1, color=colors[policy], label=POLICY_LABELS[policy])
        overview_handles.append(line)

    axes[0].set_title("Mean throughput across 2/4/6/8 occupied-core cases", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Timeout budget (ms)")
    axes[0].set_ylabel("Mean decode throughput (tok/s, log)")
    axes[0].set_yscale("log")
    axes[0].grid(True, color="#d7dde5", linewidth=0.8, alpha=0.8)
    axes[0].set_xticks(TIMEOUTS)

    axes[1].set_title("PPL from verified quality runs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Timeout budget (ms)")
    axes[1].set_ylabel("PPL")
    axes[1].grid(True, color="#d7dde5", linewidth=0.8, alpha=0.8)
    axes[1].set_xticks(TIMEOUTS)

    for ax in axes:
        ax.tick_params(labelsize=9)
    fig.legend(handles=overview_handles, loc="lower center", ncol=5, fontsize=9, frameon=False, title="Policy color")
    fig.suptitle("Scheduler Ablation Overview, verified real runs at 20-40 ms", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.91))

    png = figures_dir / "scheduler_ablation_overview_real_20_40.png"
    pdf = figures_dir / "scheduler_ablation_overview_real_20_40.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def write_report(result_dir: Path, figures: list[tuple[str, Path, Path]], long_csv: Path, summary_csv: Path) -> Path:
    report = result_dir / "REPORT_REAL_20_40.md"
    lines = [
        "# Scheduler Ablation Plots: Real 20-40 ms Data",
        "",
        "本报告只使用主 CSV 中已经闭环验证过的真实运行数据：timeout 为 `20/30/40 ms`，且所有使用行均满足 `status=ok`。",
        "",
        "未使用任何 `*.before_*` 备份文件，也未使用被外部 `monotonic_load_6occ_config1` 负载污染风险影响的 `50ms+` 中间数据。",
        "",
        "## Data Files",
        "",
        f"- Long table: `{long_csv.name}`",
        f"- Summary table: `{summary_csv.name}`",
        "- Source PPL CSV: `ppl_raw.csv`",
        "- Source decode CSV: `decode_raw.csv`",
        "",
        "## Figures",
        "",
    ]
    for title, png, pdf in figures:
        lines.extend(
            [
                f"### {title}",
                "",
                f"- PNG: `figures/{png.name}`",
                f"- PDF: `figures/{pdf.name}`",
                "",
                f"![{title}](figures/{png.name})",
                "",
            ]
        )
    lines.extend(
        [
            "## Plot Semantics",
            "",
            "### Per-ablation 2x2 figures",
            "",
            "- Solid green line: scheduler decode throughput, read from the left y-axis.",
            "- Solid purple line: ablation decode throughput, read from the left y-axis.",
            "- Dashed green line: scheduler PPL, read from the right y-axis.",
            "- Dashed purple line: ablation PPL, read from the right y-axis.",
            "- Marker shape also separates scheduler (`o`) from ablation (`s`).",
            "",
            "### Overview figure",
            "",
            "- The left subplot contains mean throughput across the four occupied-core cases.",
            "- The right subplot contains PPL.",
            "- The overview legend maps color to policy only; line style is not used to distinguish throughput from PPL in this figure.",
            "",
            "Throughput y-axes use log scale because the real runs include low-throughput tail-wait cases around `0.15 tok/s` and normal cases around `20+ tok/s`.",
            "",
            "## Ablation Definitions",
            "",
        ]
    )
    for policy, title in ABLATIONS:
        lines.append(f"- `{policy}`: {title}; {POLICY_NOTES[policy]}.")
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    args = parser.parse_args()

    ppl, decode = load_data(args.result_dir)
    figures_dir = args.result_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    long_csv = write_long_csv(args.result_dir, ppl, decode)
    summary_csv = write_summary_csv(args.result_dir, ppl, decode)

    figures: list[tuple[str, Path, Path]] = []
    overview_png, overview_pdf = plot_overview(args.result_dir, figures_dir, ppl, decode)
    figures.append(("Overview", overview_png, overview_pdf))
    for ablation, title in ABLATIONS:
        png, pdf = plot_one_ablation(args.result_dir, figures_dir, ablation, title, ppl, decode)
        figures.append((title, png, pdf))

    report = write_report(args.result_dir, figures, long_csv, summary_csv)
    print(report)
    for _title, png, pdf in figures:
        print(png)
        print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
