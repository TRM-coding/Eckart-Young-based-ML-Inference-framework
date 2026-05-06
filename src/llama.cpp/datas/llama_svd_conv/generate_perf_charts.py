#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


OUT_DIR = Path("/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/datas/llama_svd_conv")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
FONT_PROP = font_manager.FontProperties(fname=FONT_PATH, size=18)

font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = FONT_PROP.get_name()
plt.rcParams["font.size"] = 18
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "ggml": "#3B5B92",
    "pytorch": "#2A9D8F",
    "onednn": "#E76F51",
    "im2col_svd": "#9C755F",
    "fold_svd": "#F28E2B",
    "speedup": "#2F6F73",
}


def font_prop(size, weight=None):
    return font_manager.FontProperties(fname=FONT_PATH, size=size, weight=weight)


def annotate_grouped(ax, xs, values, dx):
    for offset, series in zip(dx, values):
        for x, value in zip(xs + offset, series):
            ax.text(
                x,
                value + max(series) * 0.03,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=14,
                color="#333333",
                fontproperties=FONT_PROP,
            )


def base_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(axis="y", linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)


def make_operator_chart():
    categories = ["7x7 Stem", "3x3 Block", "1x1 Bottleneck"]
    ggml = np.array([9.43745, 3.61984, 1.23041])
    pytorch = np.array([1.415657, 0.704772, 0.449512])
    onednn = np.array([0.683422, 0.629792, 0.294949])

    x = np.arange(len(categories))
    width = 0.24
    offsets = np.array([-width, 0.0, width])

    fig, ax = plt.subplots(figsize=(12.5, 8.2), dpi=180)
    fig.patch.set_facecolor("#FBFBF8")
    ax.set_facecolor("#FBFBF8")

    ax.bar(x + offsets[0], ggml, width=width, color=COLORS["ggml"], label="llama.cpp 自带 im2col Conv")
    ax.bar(x + offsets[1], pytorch, width=width, color=COLORS["pytorch"], label="PyTorch 官方 Conv 算子")
    ax.bar(x + offsets[2], onednn, width=width, color=COLORS["onednn"], label="oneDNN Conv 完整调用")

    ax.set_title("卷积算子调用耗时对比图", pad=16, weight="bold", fontproperties=FONT_PROP)
    ax.set_ylabel("单个卷积算子耗时 / ms", fontproperties=FONT_PROP)
    ax.set_xlabel("典型卷积尺寸", fontproperties=FONT_PROP)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontproperties=FONT_PROP)
    ax.set_ylim(0, max(ggml.max(), pytorch.max(), onednn.max()) * 1.28)
    base_style(ax)
    annotate_grouped(ax, x, [ggml, pytorch, onednn], offsets)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=13,
        prop=FONT_PROP,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.78)
    fig.savefig(OUT_DIR / "operator_performance_comparison.png", bbox_inches="tight")
    plt.close(fig)


def make_model_chart():
    labels = ["llama.cpp 自带 im2col 推理", "PyTorch 前向传播", "结合 oneDNN 的前向传播"]
    values = np.array([112.322, 29.158629151061177, 39.6674])
    colors = [COLORS["ggml"], COLORS["pytorch"], COLORS["onednn"]]

    fig, ax = plt.subplots(figsize=(12.0, 7.2), dpi=180)
    fig.patch.set_facecolor("#FBFBF8")
    ax.set_facecolor("#FBFBF8")

    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.58, color=colors)

    ax.set_title("模型推理速度对比图", pad=16, weight="bold", fontproperties=FONT_PROP)
    ax.set_ylabel("单次前向传播耗时 / ms", fontproperties=FONT_PROP)
    ax.set_xlabel("推理实现", fontproperties=FONT_PROP)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=FONT_PROP)
    ax.set_ylim(0, values.max() * 1.20)
    base_style(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + values.max() * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=15,
            color="#333333",
            fontproperties=FONT_PROP,
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_inference_comparison.png", bbox_inches="tight")
    plt.close(fig)


def make_operator_model_combined_chart():
    op_categories = ["7x7", "3x3", "1x1"]
    ggml = np.array([9.43745, 3.61984, 1.23041])
    pytorch = np.array([1.415657, 0.704772, 0.449512])
    onednn = np.array([0.683422, 0.629792, 0.294949])

    model_labels = ["llama.cpp", "PyTorch", "oneDNN"]
    model_values = np.array([112.322, 29.158629151061177, 39.6674])

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.8), dpi=220)
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.set_box_aspect(0.78)

    ax = axes[0]
    x = np.arange(len(op_categories))
    width = 0.18
    offsets = np.array([-0.26, 0.0, 0.26])

    ax.bar(x + offsets[0], ggml, width=width, color=COLORS["ggml"], label="llama.cpp im2col Conv")
    ax.bar(x + offsets[1], pytorch, width=width, color=COLORS["pytorch"], label="PyTorch Conv")
    ax.bar(x + offsets[2], onednn, width=width, color=COLORS["onednn"], label="oneDNN Conv")

    ax.set_title("(a) 卷积算子调用耗时", pad=10, fontproperties=font_prop(30, "bold"))
    ax.set_ylabel("耗时 / ms", fontproperties=font_prop(28))
    ax.set_xlabel("典型卷积尺寸", fontproperties=font_prop(28))
    ax.set_xticks(x)
    ax.set_xticklabels(op_categories, fontproperties=font_prop(28))
    ax.set_ylim(0, 10.6)
    base_style(ax)
    ax.tick_params(axis="both", labelsize=28, width=1.0, length=5)

    for offset, series in zip(offsets, [ggml, pytorch, onednn]):
        for xpos, value in zip(x + offset, series):
            ax.text(
                xpos,
                value + 0.18,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=16,
                color="#2F2F2F",
                fontproperties=font_prop(16),
            )

    ax = axes[1]
    x = np.arange(len(model_labels))
    bars = ax.bar(
        x,
        model_values,
        width=0.56,
        color=[COLORS["ggml"], COLORS["pytorch"], COLORS["onednn"]],
    )

    ax.set_title("(b) 模型推理速度", pad=10, fontproperties=font_prop(30, "bold"))
    ax.set_ylabel("单次前向传播耗时 / ms", fontproperties=font_prop(28))
    ax.set_xlabel("推理实现", fontproperties=font_prop(28))
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontproperties=font_prop(28))
    ax.set_ylim(0, 124)
    base_style(ax)
    ax.tick_params(axis="both", labelsize=28, width=1.0, length=5)

    for bar, value in zip(bars, model_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 2.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#2F2F2F",
            fontproperties=font_prop(16),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=26,
        prop=font_prop(26),
        columnspacing=1.1,
        handlelength=1.2,
    )

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.76, wspace=0.14)
    fig.savefig(OUT_DIR / "operator_model_combined_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "operator_model_combined_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fold_unfold_chart():
    categories = ["1x1", "3x3", "5x5", "7x7"]
    im2col_svd = np.array([1.979688, 6.782098, 16.966614, 34.799834])
    fold_svd = np.array([0.107094, 0.708539, 1.829269, 3.774640])
    speedup = im2col_svd / fold_svd

    x = np.arange(len(categories))
    width = 0.34

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=180)
    fig.patch.set_facecolor("#FBFBF8")
    ax.set_facecolor("#FBFBF8")

    ax.bar(x - width / 2.0, im2col_svd, width=width, color=COLORS["im2col_svd"], label="不使用 fold-unfold：im2col SVD")
    ax.bar(x + width / 2.0, fold_svd, width=width, color=COLORS["fold_svd"], label="使用 fold-unfold：fold SVD")

    ax2 = ax.twinx()
    ax2.plot(x, speedup, color=COLORS["speedup"], marker="o", linewidth=2.6, markersize=7, label="fold-unfold 加速比")
    ax2.set_ylabel("加速比 / x", fontproperties=FONT_PROP)
    ax2.set_ylim(0, speedup.max() * 1.22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(1.2)
    ax2.tick_params(axis="y", colors=COLORS["speedup"])

    ax.set_title("不同卷积核尺寸下 fold-unfold SVD 卷积加速对比", pad=16, weight="bold", fontproperties=FONT_PROP)
    ax.set_ylabel("单个 SVD 卷积算子耗时 / ms", fontproperties=FONT_PROP)
    ax.set_xlabel("卷积核尺寸", fontproperties=FONT_PROP)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontproperties=FONT_PROP)
    ax.set_ylim(0, im2col_svd.max() * 1.20)
    base_style(ax)

    for xpos, value in zip(x - width / 2.0, im2col_svd):
        ax.text(xpos, value + im2col_svd.max() * 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=13, fontproperties=FONT_PROP)
    for xpos, value in zip(x + width / 2.0, fold_svd):
        ax.text(xpos, value + im2col_svd.max() * 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=13, fontproperties=FONT_PROP)
    for xpos, value in zip(x, speedup):
        ax2.text(xpos, value + speedup.max() * 0.035, f"{value:.2f}x", ha="center", va="bottom", fontsize=13, color=COLORS["speedup"], fontproperties=FONT_PROP)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=13,
        prop=FONT_PROP,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.12, top=0.78)
    fig.savefig(OUT_DIR / "fold_unfold_svd_speedup_by_kernel.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_operator_chart()
    make_model_chart()
    make_operator_model_combined_chart()
    make_fold_unfold_chart()


if __name__ == "__main__":
    main()
