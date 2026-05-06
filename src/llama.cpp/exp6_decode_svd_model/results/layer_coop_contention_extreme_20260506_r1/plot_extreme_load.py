#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw.csv"
FIG_DIR = HERE / "figures"


def setup_font() -> None:
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


def load_rows() -> list[dict[str, str]]:
    with RAW.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok" and r["tok_s"]]
    return rows


def display_name(row: dict[str, str]) -> str:
    policy = row["policy"]
    if policy == "pc_full":
        return "PC基线"
    if policy.startswith("major_only_"):
        cores = policy.removeprefix("major_only_").removesuffix("c")
        return f"No-Offloading\n{cores}核"
    if policy == "split":
        return f"协同推理\nM={row['split_m']}"
    return policy


def main() -> int:
    setup_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    order = ["pc_full", "major_only_2c", "major_only_4c", "major_only_6c", "split:2", "split:4"]
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row["policy"] if row["policy"] != "split" else f"split:{row['split_m']}"
        by_key[key] = row
    plot_rows = [by_key[k] for k in order if k in by_key]

    labels = [display_name(r) for r in plot_rows]
    tok_s = [float(r["tok_s"]) for r in plot_rows]
    baseline = float(by_key["pc_full"]["tok_s"])
    best_major = max(float(r["tok_s"]) for r in rows if r["policy"].startswith("major_only_"))
    best_split = max((r for r in rows if r["policy"] == "split"), key=lambda r: float(r["tok_s"]))
    best_split_tok = float(best_split["tok_s"])

    colors = []
    for row in plot_rows:
        if row["policy"] == "pc_full":
            colors.append("#d62728")
        elif row["policy"].startswith("major_only_"):
            colors.append("#4c78a8")
        elif row["policy"] == "split" and row["split_m"] == best_split["split_m"]:
            colors.append("#f58518")
        else:
            colors.append("#72b7b2")

    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=180)
    x = list(range(len(plot_rows)))
    bars = ax.bar(x, tok_s, width=0.64, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.axhline(baseline, color="#d62728", linestyle="--", linewidth=1.8, label=f"PC基线 {baseline:.2f} tok/s")
    ax.axhline(best_major, color="#4c78a8", linestyle=":", linewidth=1.8, label=f"最优No-Offloading {best_major:.2f} tok/s")

    for bar, value in zip(bars, tok_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.annotate(
        f"相对PC基线 {best_split_tok / baseline:.1f}x\n相对No-Offloading {best_split_tok / best_major:.2f}x",
        xy=(labels.index(f"协同推理\nM={best_split['split_m']}"), best_split_tok),
        xytext=(3.55, 14.7),
        ha="center",
        fontsize=9.5,
        arrowprops={"arrowstyle": "->", "color": "#8c3b12", "lw": 1.2},
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff4e8", "ec": "#f58518", "lw": 0.8},
    )

    ax.set_title("极端负载下真实协同推理性能", fontsize=15, pad=12)
    ax.set_ylabel("吞吐量 (tok/s)", fontsize=12)
    ax.set_xlabel("推理策略", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(tok_s) * 1.38)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()

    png = FIG_DIR / "extreme_load_coop_vs_no_offloading.png"
    pdf = FIG_DIR / "extreme_load_coop_vs_no_offloading.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    # Also export the exact plotting rows for paper scripts.
    csv_out = FIG_DIR / "extreme_load_plot_data.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["display_name", "policy", "split_m", "tok_s", "baseline_tok_s", "best_major_only_tok_s"])
        for row, label, value in zip(plot_rows, labels, tok_s):
            w.writerow([label.replace("\n", " "), row["policy"], row["split_m"], value, baseline, best_major])

    print(png)
    print(pdf)
    print(csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
