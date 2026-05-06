#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "plot_data_filtered_monotonic.csv"
OUT = HERE / "figures_filtered_monotonic"
TOKENS = 32
TIMEOUT_S = 120.0


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


def read_rows() -> list[dict[str, str]]:
    with DATA.open(newline="") as f:
        return list(csv.DictReader(f))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key] or 0.0)


def speedup_label(row: dict[str, str]) -> str:
    if row.get("speedup"):
        return f"{float(row['speedup']):.2f}x"
    selected = value(row, "selected_tok_s")
    baseline = value(row, "baseline_tok_s")
    if baseline <= 0 and selected > 0:
        lower_bound = selected / (TOKENS / TIMEOUT_S)
        return f">{lower_bound:.1f}x"
    if baseline <= 0 and selected <= 0:
        return "timeout"
    return "1.00x"


def config_table_cells(subset: list[dict[str, str]], config_alias: dict[int, int]) -> list[list[str]]:
    items = [
        f"Config{config_alias[int(row['config_id'])]}: {row['occupied_loads']}"
        for row in subset
    ]
    while len(items) < 4:
        items.append("")
    return [[items[0], items[1]], [items[2], items[3]]]


def main() -> int:
    configure_font()
    OUT.mkdir(exist_ok=True)
    rows = read_rows()
    raw_config_ids = sorted({int(row["config_id"]) for row in rows})
    config_alias = {raw: idx + 1 for idx, raw in enumerate(raw_config_ids)}

    plt.rcParams.update(
        {
            "font.size": 24,
            "axes.labelsize": 24,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.titlesize": 28,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(34.0, 17.6))
    gs = GridSpec(
        4,
        2,
        figure=fig,
        height_ratios=[2.65, 0.88, 2.65, 0.88],
        hspace=0.30,
        wspace=0.11,
    )
    axes_flat = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    table_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
    ]
    occupied_titles = {
        2: "8 核可用，2 核有占用",
        4: "8 核可用，4 核有占用",
        6: "8 核可用，6 核有占用",
        8: "8 核可用，8 核有占用",
    }
    max_y = max(max(value(row, "baseline_tok_s"), value(row, "selected_tok_s")) for row in rows)

    width = 0.27
    for ax, tax, occupied in zip(axes_flat, table_axes, [2, 4, 6, 8]):
        subset = sorted(
            [row for row in rows if int(row["occupied_count"]) == occupied],
            key=lambda row: int(row["config_id"]),
        )
        x = np.arange(len(subset), dtype=float)
        baseline = np.array([value(row, "baseline_tok_s") for row in subset])
        schedule = np.array([value(row, "selected_tok_s") for row in subset])

        ax.bar(
            x - width / 2,
            baseline,
            width=width,
            label="基线",
            color="#8da0cb",
            edgecolor="#34495e",
            linewidth=1.0,
        )
        bars = ax.bar(
            x + width / 2,
            schedule,
            width=width,
            label="调度策略",
            color="#fc8d62",
            edgecolor="#7f3b20",
            linewidth=1.0,
        )

        for bar, row in zip(bars, subset):
            label = speedup_label(row)
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + max_y * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=22,
                color="#7f3b20",
            )

        ax.set_title(occupied_titles[occupied], fontsize=30, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Config{config_alias[int(row['config_id'])]}" for row in subset])
        ax.set_ylim(0, max_y * 1.20)
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.tick_params(axis="x", pad=8)

        tax.axis("off")
        tax.text(
            0.5,
            0.94,
            "各占用核心负载（%）",
            transform=tax.transAxes,
            ha="center",
            va="center",
            fontsize=24,
        )
        table = tax.table(
            cellText=config_table_cells(subset, config_alias),
            cellLoc="center",
            loc="center",
            bbox=[0.02, 0.04, 0.96, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(17)
        for cell in table.get_celld().values():
            cell.set_edgecolor("#333333")
            cell.set_linewidth(1.1)
            cell.set_facecolor("white")

    axes_flat[0].set_ylabel("吞吐量 (tok/s)")
    axes_flat[2].set_ylabel("吞吐量 (tok/s)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        fontsize=24,
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.895, bottom=0.045)
    png = OUT / "controlled_monotonic_2x2_combined.png"
    pdf = OUT / "controlled_monotonic_2x2_combined.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
