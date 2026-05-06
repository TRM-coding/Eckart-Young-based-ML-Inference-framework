#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "plot_data.csv"
OUT = ROOT / "figures"


def compact_label(row: dict[str, str]) -> str:
    loads = [int(float(item)) for item in row["loads"].split(",")]
    occupied = int(row["occupied_count"])
    load_part = "/".join(str(item) for item in loads[:occupied])
    return f"{load_part}"


def strategy_label(row: dict[str, str]) -> str:
    return "No-Offloading"


def configure_chinese_font() -> None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    configure_chinese_font()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(DATA.open()))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
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
        subset = sorted(
            subset,
            key=lambda row: (0 if row["source"] == "low_load_additions" else 1, row["plot_id"]),
        )
        config_names = [f"Config{i + 1}" for i in range(len(subset))]
        config_loads = [compact_label(row) for row in subset]
        strategies = [strategy_label(row) for row in subset]
        baseline = np.array([float(row["baseline_tok_s"]) for row in subset])
        schedule = np.array([float(row["selected_tok_s"]) for row in subset])
        speedup = np.array([float(row["speedup"]) for row in subset])

        x = np.arange(len(subset))
        width = 0.36

        fig = plt.figure(figsize=(8.4, 4.28))
        grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.85, 0.95], hspace=0.34)
        ax = fig.add_subplot(grid[0])
        config_ax = fig.add_subplot(grid[1])
        ax.bar(
            x - width / 2,
            baseline,
            width,
            label="基线",
            color="#8da0cb",
            edgecolor="#34495e",
            linewidth=0.6,
        )
        bars = ax.bar(
            x + width / 2,
            schedule,
            width,
            label="调度策略",
            color="#fc8d62",
            edgecolor="#7f3b20",
            linewidth=0.6,
        )

        ymax = max(schedule.max(), baseline.max())
        ax.set_title(f"8 核可用，{occupied} 核有占用")
        ax.set_ylabel("吞吐量 (tok/s)")
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, rotation=0, ha="center")
        ax.tick_params(axis="x", pad=3)
        ax.set_ylim(0, ymax * 1.24)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", frameon=False, ncols=2)

        for rect, ratio, strategy in zip(bars, speedup, strategies):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + ymax * 0.025,
                f"{strategy}\n{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#7f3b20",
            )

        config_ax.axis("off")
        config_ax.set_xlim(0, 1)
        config_ax.set_ylim(0, 1)
        config_ax.text(
            0.0,
            0.98,
            "各占用核心负载（%）：",
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )
        # Two columns keep the labels readable while using less vertical space.
        for idx, (name, loads_text) in enumerate(zip(config_names, config_loads)):
            col = idx // 4
            row = idx % 4
            xpos = 0.0 + col * 0.50
            ypos = 0.75 - row * 0.21
            config_ax.text(
                xpos,
                ypos,
                f"{name}: {loads_text}",
                ha="left",
                va="center",
                fontsize=8.6,
            )

        fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.045)
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
