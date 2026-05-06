#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'plot_timeout_sweep_long.csv'
OUT = ROOT / 'figures_top3_grouped'
OUT.mkdir(parents=True, exist_ok=True)


def configure_font() -> None:
    for candidate in [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False


def f(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def config_rank_key(rows: list[dict[str, str]]) -> tuple[float, float]:
    # Rank by average speedup; use max speedup as tiebreaker.
    speeds = [f(row['speedup_vs_baseline']) for row in rows]
    return (float(np.mean(speeds)), max(speeds))


def main() -> int:
    configure_font()
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    rows = list(csv.DictReader(DATA.open()))
    manifest: list[dict[str, object]] = []
    colors = ['#fc8d62', '#66c2a5', '#e78ac3']
    baseline_color = '#b8c4dc'

    for occ in [2, 4, 6, 8]:
        occ_rows = [row for row in rows if int(row['occupied_count']) == occ]
        by_cfg: dict[str, list[dict[str, str]]] = {}
        for row in occ_rows:
            by_cfg.setdefault(row['config_id'], []).append(row)
        ranked = sorted(by_cfg.items(), key=lambda item: config_rank_key(item[1]), reverse=True)
        selected = ranked[:3]

        timeouts = sorted({int(float(row['timeout_ms'])) for row in occ_rows})
        x = np.arange(len(timeouts))
        # 3 configs, each has baseline + schedule. Keep bars compact.
        pair_width = 0.22
        bar_width = pair_width / 2.0
        offsets = np.array([-pair_width, 0.0, pair_width])

        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ymax = 0.0
        legend_handles = []
        legend_labels = []

        for cfg_idx, (cfg, cfg_rows) in enumerate(selected):
            by_timeout = {int(float(row['timeout_ms'])): row for row in cfg_rows}
            baseline = np.array([f(by_timeout[t]['baseline_tok_s']) for t in timeouts])
            schedule = np.array([f(by_timeout[t]['schedule_tok_s']) for t in timeouts])
            speedup = np.array([f(by_timeout[t]['speedup_vs_baseline']) for t in timeouts])
            ymax = max(ymax, float(baseline.max()), float(schedule.max()))

            center = x + offsets[cfg_idx]
            b_base = ax.bar(
                center - bar_width / 2,
                baseline,
                bar_width,
                color=baseline_color,
                edgecolor='#34495e',
                linewidth=0.45,
                hatch='//',
                alpha=0.95,
            )
            b_sched = ax.bar(
                center + bar_width / 2,
                schedule,
                bar_width,
                color=colors[cfg_idx],
                edgecolor='#4b2b1a',
                linewidth=0.45,
                alpha=0.95,
            )
            legend_handles.extend([b_base[0], b_sched[0]])
            legend_labels.extend([f'{cfg} 基线', f'{cfg} 调度'])

            # Label only the best timeout per config to reduce clutter.
            best_i = int(np.argmax(speedup))
            rect = b_sched[best_i]
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(1.0, ymax) * 0.025,
                f'{speedup[best_i]:.1f}x',
                ha='center',
                va='bottom',
                fontsize=8,
                color='#7f3b20',
            )

        ax.set_title(f'8 核可用，{occ} 核有占用：Top-3 提速配置')
        ax.set_xlabel('单 token 允许超时时间 (ms)')
        ax.set_ylabel('吞吐量 (tok/s)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in timeouts])
        ax.set_ylim(0, ymax * 1.20)
        ax.grid(axis='y', linestyle='--', alpha=0.28)
        # Put legend below chart in multiple columns.
        ax.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.20),
            ncols=3,
            frameon=False,
        )
        fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.29)

        png = OUT / f'timeout_grouped_top3_{occ}occ.png'
        pdf = OUT / f'timeout_grouped_top3_{occ}occ.pdf'
        fig.savefig(png, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)

        manifest.append({
            'occupied_count': occ,
            'selected_configs': ','.join(cfg for cfg, _ in selected),
            'png': str(png),
            'pdf': str(pdf),
            'mean_speedups': ','.join(f'{config_rank_key(cfg_rows)[0]:.6f}' for _, cfg_rows in selected),
        })

    with (OUT / 'top3_grouped_manifest.csv').open('w', newline='') as fcsv:
        fieldnames = ['occupied_count', 'selected_configs', 'mean_speedups', 'png', 'pdf']
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(OUT)
    for item in manifest:
        print(item)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
