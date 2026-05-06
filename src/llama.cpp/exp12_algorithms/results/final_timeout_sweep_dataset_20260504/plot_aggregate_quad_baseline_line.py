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
DATA = ROOT / "plot_timeout_sweep_aggregate.csv"
LONG_DATA = ROOT / "plot_timeout_sweep_long.csv"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def configure_font() -> None:
    preferred_paths = [
        "/usr/share/fonts/truetype/simsun.ttc",
        "/usr/share/fonts/truetype/simsun.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for candidate in preferred_paths:
        path = Path(candidate)
        if not path.exists():
            continue
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = family
        plt.rcParams["font.serif"] = [family]
        plt.rcParams["font.sans-serif"] = [family]
        break
    plt.rcParams["axes.unicode_minus"] = False


def draw_panel(
    ax: plt.Axes,
    sub: pd.DataFrame,
    occ: int,
    *,
    panel_title: str | None = None,
) -> tuple[list[object], list[str]]:
    sub = sub.sort_values("timeout_ms")
    x = np.arange(len(sub))
    timeouts = sub["timeout_ms"].astype(int).to_numpy()
    baseline = sub["baseline_tok_s_mean"].astype(float).to_numpy()
    schedule = sub["schedule_tok_s_mean"].astype(float).to_numpy()
    ppl = sub["ppl_mean"].astype(float).to_numpy()
    speedup = sub["speedup_mean"].astype(float).to_numpy()

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
    # Baseline is effectively constant for each occupied-count class because
    # the timeout budget only affects the schedule path.
    base_line = ax.axhline(
        float(baseline.mean()),
        color="#d73027",
        linestyle="--",
        linewidth=1.9,
        label="基线吞吐",
        zorder=3,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
    ax.set_title(panel_title or f"8 核可用，{occ} 核有占用", fontsize=12)
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
    ax2.set_ylim(0, max(40.0, float(ppl.max()) * 1.12))
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

    df = pd.read_csv(DATA)
    long_df = pd.read_csv(LONG_DATA)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.0))
    fig.suptitle("不同超时上限下的调度吞吐与 PPL", fontsize=15, y=0.985)

    handles: list[object] = []
    labels: list[str] = []
    for ax, occ in zip(axes.flat, [2, 4, 6, 8]):
        if occ == 8:
            cfg = long_df[(long_df["occupied_count"] == 8) & (long_df["config_id"] == "8occ_cfg2")].copy()
            cfg.sort_values("timeout_ms", inplace=True)
            sub = pd.DataFrame(
                {
                    "occupied_count": cfg["occupied_count"],
                    "timeout_ms": cfg["timeout_ms"],
                    "baseline_tok_s_mean": cfg["baseline_tok_s"],
                    "schedule_tok_s_mean": cfg["schedule_tok_s"],
                    "speedup_mean": cfg["speedup_vs_baseline"],
                    "ppl_mean": cfg["schedule_ppl"],
                }
            )
            panel_title = "8 核可用，8 核有占用"
        else:
            sub = df[df["occupied_count"] == occ]
            panel_title = None
        panel_handles, panel_labels = draw_panel(ax, sub, occ, panel_title=panel_title)
        if not handles:
            handles = panel_handles
            labels = panel_labels

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncols=3, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.91, bottom=0.13, hspace=0.42, wspace=0.32)

    png = OUT / "timeout_sweep_quad_baseline_line.png"
    pdf = OUT / "timeout_sweep_quad_baseline_line.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = [
        {
            "png": str(png),
            "pdf": str(pdf),
            "source_csv": str(DATA),
            "baseline_style": "red_dashed_horizontal_line",
            "panels": "2occ,4occ,6occ,8occ",
        }
    ]
    with (OUT / "timeout_sweep_quad_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
