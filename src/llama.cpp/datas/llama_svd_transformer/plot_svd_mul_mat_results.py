#!/usr/bin/env python3
"""Create publication-quality Chinese plots for the transformer SVD matmul benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


TYPE_ORDER = ["f32", "f16", "q4_0"]
TYPE_LABEL = {
    "f32": "F32",
    "f16": "F16",
    "q4_0": "Q4_0",
}

PROJ_ORDER = ["up/gate", "down"]
PROJ_LABEL = {
    "up/gate": "上投影/门控投影",
    "down": "下投影",
}

CASE_ORDER = [
    "square_512",
    "square_1024",
    "square_1536",
    "qwen_ffn_up_gate_full",
    "qwen_ffn_down_full",
]

CASE_LABEL = {
    "square_512": "512→512\nK=512",
    "square_1024": "1024→1024\nK=1024",
    "square_1536": "1536→1536\nK=1536",
    "qwen_ffn_up_gate_full": "1536→8960\nK=1536",
    "qwen_ffn_down_full": "8960→1536\nK=1536",
}

# Okabe-Ito inspired, colorblind-friendly palette.
COLORS = {
    "up/gate": "#0072B2",
    "down": "#D55E00",
    "fused": "#009E73",
    "baseline": "#CC79A7",
    "f32": "#0072B2",
    "f16": "#D55E00",
    "q4_0": "#009E73",
    "grid": "#D8DEE9",
    "text": "#1F2933",
}


def configure_style() -> None:
    for font_path in [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "WenQuanYi Zen Hei",
                "Noto Sans CJK JP",
                "Noto Sans CJK SC",
                "Droid Sans Fallback",
                "DejaVu Sans",
            ],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#56616F",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def load_data() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RESULTS.glob("scale_*repeat100.csv")):
        frame = pd.read_csv(csv_path, comment="#")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no scale repeat100 CSV files found under {RESULTS}")

    data = pd.concat(frames, ignore_index=True)
    data["type"] = pd.Categorical(data["type"], TYPE_ORDER, ordered=True)
    data["case"] = pd.Categorical(data["case"], CASE_ORDER, ordered=True)
    data["case_label"] = data["case"].astype(str).map(CASE_LABEL)
    data = data.sort_values(["type", "case"]).reset_index(drop=True)
    return data


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(FIGURES / f"{stem}.{suffix}")


def plot_speedup(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 3.3))
    x = np.arange(len(CASE_ORDER))
    width = 0.24

    for i, dtype in enumerate(TYPE_ORDER):
        sub = data[data["type"] == dtype].set_index("case").loc[CASE_ORDER].reset_index()
        ax.bar(
            x + (i - 1) * width,
            sub["speedup_vs_two_mul_mat"],
            width,
            label=TYPE_LABEL[dtype],
            color=COLORS[dtype],
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axhline(1.0, color="#3B4252", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_ylabel("加速比（×）")
    ax.set_xlabel("矩阵规模（输入维度→输出维度；K 为中间维度）")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in CASE_ORDER])
    ax.set_ylim(0.0, max(3.15, data["speedup_vs_two_mul_mat"].max() * 1.18))
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.24, top=0.82)
    save_all(fig, "svd_mul_mat_scale_speedup_zh")
    plt.close(fig)


def plot_latency(data: pd.DataFrame) -> None:
    latency_type_order = ["f16", "q4_0"]

    fig, axes = plt.subplots(1, 2, figsize=(20.8, 6.4), sharey=False)
    fig.patch.set_facecolor("white")
    tick_labels = [
        "512",
        "1024",
        "1536",
        "1536",
        "8960",
    ]
    x = np.arange(len(CASE_ORDER)) * 1.35
    width = 0.30
    handles = None
    legend_labels = None

    for ax, dtype in zip(axes, latency_type_order):
        ax.set_facecolor("white")
        sub = data[data["type"] == dtype].copy()
        sub = sub.set_index("case").loc[CASE_ORDER].reset_index()

        ax.bar(
            x - width / 2,
            sub["fused_median_ms"],
            width,
            label="融合 SVD 算子",
            color=COLORS["fused"],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            x + width / 2,
            sub["two_mul_mat_median_ms"],
            width,
            label="两次官方 mul_mat",
            color=COLORS["baseline"],
            edgecolor="white",
            linewidth=0.6,
        )

        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_title(TYPE_LABEL[dtype], loc="left", pad=6, fontsize=30)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=28)
        ax.set_xlim(x[0] - 0.65, x[-1] + 0.65)
        ax.tick_params(axis="x", pad=8)
        ax.tick_params(axis="y", labelsize=28)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("中位延迟（ms）", fontsize=28)
    fig.supxlabel("矩阵规模", fontsize=28, y=0.035)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=2,
        frameon=False,
        fontsize=28,
        handlelength=1.5,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.23, top=0.80, wspace=0.24)
    save_all(fig, "svd_mul_mat_scale_latency_zh")
    plt.close(fig)


def main() -> None:
    global FIGURES

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    FIGURES = args.out_dir
    FIGURES.mkdir(parents=True, exist_ok=True)

    configure_style()
    data = load_data()
    data.to_csv(FIGURES / "combined_results.csv", index=False)
    plot_speedup(data)
    plot_latency(data)
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
