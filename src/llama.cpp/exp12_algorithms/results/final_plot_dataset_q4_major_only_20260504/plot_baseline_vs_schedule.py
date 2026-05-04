#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "plot_data.csv"
OUT = ROOT / "figures"


def compact_label(row: dict[str, str]) -> str:
    loads = [int(float(item)) for item in row["loads"].split(",")]
    workers = [int(float(item)) for item in row["workers"].split(",")]
    occupied = int(row["occupied_count"])
    load_part = "/".join(str(item) for item in loads[:occupied])
    worker_part = "/".join(str(item) for item in workers[:occupied])
    if any(item != 1 for item in workers[:occupied]):
        return f"L:{load_part}\nW:{worker_part}\n{row['method']}"
    return f"L:{load_part}\n{row['method']}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(DATA.open()))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    manifest: list[dict[str, object]] = []
    for occupied in [2, 4, 6, 8]:
        subset = [row for row in rows if int(row["occupied_count"]) == occupied]
        subset = sorted(subset, key=lambda row: row["plot_id"])
        labels = [compact_label(row) for row in subset]
        baseline = np.array([float(row["baseline_tok_s"]) for row in subset])
        schedule = np.array([float(row["selected_tok_s"]) for row in subset])
        speedup = np.array([float(row["speedup"]) for row in subset])

        x = np.arange(len(subset))
        width = 0.36

        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.bar(
            x - width / 2,
            baseline,
            width,
            label="Baseline",
            color="#8da0cb",
            edgecolor="#34495e",
            linewidth=0.6,
        )
        bars = ax.bar(
            x + width / 2,
            schedule,
            width,
            label="Schedule",
            color="#fc8d62",
            edgecolor="#7f3b20",
            linewidth=0.6,
        )

        ymax = max(schedule.max(), baseline.max())
        ax.set_title(f"8 Available Cores, {occupied} Occupied Cores")
        ax.set_ylabel("Throughput (tok/s)")
        ax.set_xlabel("Background load configuration")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylim(0, ymax * 1.18)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", frameon=False, ncols=2)

        for rect, ratio in zip(bars, speedup):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + ymax * 0.025,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7f3b20",
            )

        fig.tight_layout()
        png = OUT / f"occupied_{occupied}c_baseline_vs_schedule.png"
        pdf = OUT / f"occupied_{occupied}c_baseline_vs_schedule.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        manifest.append(
            {
                "occupied_count": occupied,
                "png": str(png),
                "pdf": str(pdf),
                "baseline_mean": baseline.mean(),
                "schedule_mean": schedule.mean(),
                "speedup_mean": speedup.mean(),
            }
        )

    with (OUT / "figure_manifest.csv").open("w", newline="") as f:
        fieldnames = ["occupied_count", "png", "pdf", "baseline_mean", "schedule_mean", "speedup_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
