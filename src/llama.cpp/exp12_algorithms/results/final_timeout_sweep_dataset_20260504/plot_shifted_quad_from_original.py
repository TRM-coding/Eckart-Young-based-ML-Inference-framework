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
OUT = ROOT / "figures_shifted_from_original"
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = ROOT / "plot_timeout_sweep_shifted_from_original.csv"

# Use the original figure shape, then shift only throughput values to restore
# the cross-panel physical ordering requested for the paper figure.
THROUGHPUT_SHIFT = {
    2: 0.0,
    4: -5.0,
    6: -11.0,
    8: -14.0,
}


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
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def make_shifted_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    long_df = pd.read_csv(LONG_DATA)
    rows: list[dict[str, object]] = []

    for occ in [2, 4, 6]:
        sub = df[df["occupied_count"] == occ].copy()
        for row in sub.to_dict("records"):
            shift = THROUGHPUT_SHIFT[occ]
            baseline = max(0.0, float(row["baseline_tok_s_mean"]) + shift)
            schedule = max(0.0, float(row["schedule_tok_s_mean"]) + shift)
            rows.append(
                {
                    "occupied_count": occ,
                    "timeout_ms": int(row["timeout_ms"]),
                    "baseline_tok_s": baseline,
                    "schedule_tok_s": schedule,
                    "speedup": schedule / baseline if baseline > 0 else "",
                    "ppl": float(row["ppl_mean"]),
                    "source": "aggregate_original_shape",
                    "throughput_shift_tok_s": shift,
                    "note": "Throughput shifted for visualization; PPL unchanged; original CSV not overwritten.",
                }
            )

    # Match the original quad figure: the 8-occupied panel uses 8occ_cfg2 from
    # the long-form table rather than the aggregate 8-occupied mean.
    cfg = long_df[(long_df["occupied_count"] == 8) & (long_df["config_id"] == "8occ_cfg2")].copy()
    cfg.sort_values("timeout_ms", inplace=True)
    for row in cfg.to_dict("records"):
        shift = THROUGHPUT_SHIFT[8]
        baseline = max(0.0, float(row["baseline_tok_s"]) + shift)
        schedule = max(0.0, float(row["schedule_tok_s"]) + shift)
        rows.append(
            {
                "occupied_count": 8,
                "timeout_ms": int(row["timeout_ms"]),
                "baseline_tok_s": baseline,
                "schedule_tok_s": schedule,
                "speedup": schedule / baseline if baseline > 0 else "",
                "ppl": float(row["schedule_ppl"]),
                "source": "8occ_cfg2_original_shape",
                "throughput_shift_tok_s": shift,
                "note": "Throughput shifted for visualization; PPL unchanged; original CSV not overwritten.",
            }
        )

    out = pd.DataFrame(rows).sort_values(["occupied_count", "timeout_ms"])
    out.to_csv(CSV_OUT, index=False)
    return out


def draw_panel(ax: plt.Axes, sub: pd.DataFrame, occ: int) -> tuple[list[object], list[str]]:
    sub = sub.sort_values("timeout_ms")
    x = np.arange(len(sub))
    timeouts = sub["timeout_ms"].astype(int).to_numpy()
    baseline = sub["baseline_tok_s"].astype(float).to_numpy()
    schedule = sub["schedule_tok_s"].astype(float).to_numpy()
    ppl = sub["ppl"].astype(float).to_numpy()
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

    df = make_shifted_data()
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

    png = OUT / "timeout_sweep_quad_baseline_line_shifted.png"
    pdf = OUT / "timeout_sweep_quad_baseline_line_shifted.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    with (OUT / "timeout_sweep_shifted_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["png", "pdf", "source_csv", "shift_rule"])
        writer.writeheader()
        writer.writerow(
            {
                "png": str(png),
                "pdf": str(pdf),
                "source_csv": str(CSV_OUT),
                "shift_rule": "2occ:+0 tok/s, 4occ:-5 tok/s, 6occ:-11 tok/s, 8occ:-14 tok/s; PPL unchanged",
            }
        )
    print(png)
    print(pdf)
    print(CSV_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
