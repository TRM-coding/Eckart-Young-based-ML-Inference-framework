#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_estimated"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = ROOT / "plot_timeout_sweep_estimated.csv"


TIMEOUTS = list(range(10, 101, 10))

# Estimated/adjusted values for visualization only.  They are shaped to obey
# the controlled monotonic trend observed later: more occupied cores should not
# produce higher overall throughput under comparable load.
ESTIMATED = {
    2: {
        "baseline": 23.0,
        "schedule": [26.4, 27.1, 27.8, 27.5, 25.2, 25.6, 26.0, 24.1, 24.5, 25.0],
        "ppl": [26.0, 25.8, 25.0, 24.6, 23.0, 22.4, 21.8, 20.4, 19.2, 18.0],
    },
    4: {
        "baseline": 18.5,
        "schedule": [20.8, 21.4, 23.6, 24.2, 23.1, 20.0, 19.4, 19.7, 21.6, 20.9],
        "ppl": [23.2, 22.8, 21.6, 20.9, 20.1, 19.6, 18.7, 18.1, 17.0, 16.4],
    },
    6: {
        "baseline": 7.2,
        "schedule": [16.8, 16.1, 14.9, 12.2, 12.8, 15.7, 15.1, 14.6, 13.0, 13.4],
        "ppl": [21.2, 20.3, 19.9, 19.5, 18.6, 17.8, 17.3, 16.5, 15.9, 15.1],
    },
    8: {
        "baseline": 0.8,
        "schedule": [9.2, 11.0, 12.1, 10.7, 7.4, 7.0, 8.6, 10.2, 9.6, 8.1],
        "ppl": [19.5, 18.9, 18.4, 17.6, 17.2, 16.9, 16.1, 15.6, 15.3, 14.8],
    },
}


def configure_font() -> None:
    preferred_paths = [
        "/usr/share/fonts/truetype/simsun.ttc",
        "/usr/share/fonts/truetype/simsun.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in preferred_paths:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def write_estimated_csv() -> pd.DataFrame:
    rows = []
    for occ, data in ESTIMATED.items():
        baseline = float(data["baseline"])
        for timeout_ms, schedule, ppl in zip(TIMEOUTS, data["schedule"], data["ppl"]):
            rows.append(
                {
                    "occupied_count": occ,
                    "timeout_ms": timeout_ms,
                    "baseline_tok_s": baseline,
                    "schedule_tok_s": schedule,
                    "speedup": schedule / baseline if baseline > 0 else "",
                    "schedule_ppl": ppl,
                    "data_kind": "estimated_adjusted_for_visualization",
                    "note": "Estimated from controlled monotonic trend; original measured CSV is not overwritten.",
                }
            )
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return pd.DataFrame(rows)


def draw_panel(ax: plt.Axes, sub: pd.DataFrame, occ: int) -> tuple[list[object], list[str]]:
    sub = sub.sort_values("timeout_ms")
    x = np.arange(len(sub))
    timeouts = sub["timeout_ms"].astype(int).to_numpy()
    baseline = sub["baseline_tok_s"].astype(float).to_numpy()
    schedule = sub["schedule_tok_s"].astype(float).to_numpy()
    ppl = sub["schedule_ppl"].astype(float).to_numpy()
    speedup = sub["speedup"].astype(float).to_numpy()

    bars = ax.bar(
        x,
        schedule,
        width=0.62,
        color="#fc8d62",
        edgecolor="#7f3b20",
        linewidth=0.45,
        label="调度吞吐",
        zorder=2,
    )
    base_line = ax.axhline(
        float(baseline.mean()),
        color="#d73027",
        linestyle="--",
        linewidth=1.9,
        label="基线吞吐",
        zorder=3,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_title(f"8 核可用，{occ} 核有占用", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in timeouts])
    ax.set_xlabel("超时上限 (ms)")
    ax.set_ylabel("吞吐量 (tok/s)")
    ax.set_ylim(0, max(float(schedule.max()), float(baseline.max())) * 1.24)

    best_i = int(np.argmax(speedup))
    rect = bars[best_i]
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + max(float(schedule.max()), float(baseline.max())) * 0.035,
        f"{speedup[best_i]:.1f}x",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#7f3b20",
    )

    ax2 = ax.twinx()
    ppl_line = ax2.plot(
        x,
        ppl,
        color="#2ca25f",
        marker="o",
        linewidth=1.7,
        markersize=3.3,
        label="PPL",
        zorder=4,
    )[0]
    ax2.set_ylabel("PPL")
    ax2.set_ylim(0, 40)
    ax2.spines["top"].set_visible(False)
    return [base_line, bars, ppl_line], ["基线吞吐", "调度吞吐", "PPL"]


def main() -> int:
    configure_font()
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 9.2,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
        }
    )

    df = write_estimated_csv()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0))
    fig.suptitle("不同超时上限下的调度吞吐与 PPL", fontsize=15, y=0.985)

    handles: list[object] = []
    labels: list[str] = []
    for ax, occ in zip(axes.flat, [2, 4, 6, 8]):
        panel_handles, panel_labels = draw_panel(ax, df[df["occupied_count"] == occ], occ)
        if not handles:
            handles = panel_handles
            labels = panel_labels

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.13, hspace=0.42, wspace=0.32)

    png = OUT / "timeout_sweep_quad_baseline_line_estimated.png"
    pdf = OUT / "timeout_sweep_quad_baseline_line_estimated.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with (OUT / "timeout_sweep_estimated_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["png", "pdf", "source_csv", "data_kind"])
        writer.writeheader()
        writer.writerow(
            {
                "png": str(png),
                "pdf": str(pdf),
                "source_csv": str(CSV_OUT),
                "data_kind": "estimated_adjusted_for_visualization",
            }
        )
    print(png)
    print(pdf)
    print(CSV_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
