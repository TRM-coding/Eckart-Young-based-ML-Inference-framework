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
OUT = ROOT / 'figures_top3_subplots'
OUT.mkdir(parents=True, exist_ok=True)
MIN_BASELINE_TOK_S = 15.0


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


def mean_speed(rows: list[dict[str, str]]) -> float:
    return float(np.mean([f(row['speedup_vs_baseline']) for row in rows]))


def max_speed(rows: list[dict[str, str]]) -> float:
    return max(f(row['speedup_vs_baseline']) for row in rows)


def short_loads(loads: str, occ: int) -> str:
    return '/'.join(loads.split(',')[:occ])


def main() -> int:
    configure_font()
    plt.rcParams.update({
        'font.size': 9.5,
        'axes.titlesize': 10.5,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'axes.spines.top': False,
    })

    rows = list(csv.DictReader(DATA.open()))
    manifest: list[dict[str, object]] = []

    for occ in [2, 4, 6, 8]:
        occ_rows = [row for row in rows if int(row['occupied_count']) == occ]
        by_cfg: dict[str, list[dict[str, str]]] = {}
        for row in occ_rows:
            by_cfg.setdefault(row['config_id'], []).append(row)
        valid_cfgs = [
            item for item in by_cfg.items()
            if min(f(row['baseline_tok_s']) for row in item[1]) >= MIN_BASELINE_TOK_S
        ]
        selected = sorted(valid_cfgs, key=lambda item: (mean_speed(item[1]), max_speed(item[1])), reverse=True)[:3]
        if len(selected) < 3:
            raise RuntimeError(f'occupied_count={occ} only has {len(selected)} configs with baseline >= {MIN_BASELINE_TOK_S}')

        fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharex=True)
        fig.suptitle(f'8 核可用，{occ} 核有占用：不同超时上限下的吞吐与 PPL', fontsize=14, y=0.98)

        all_handles = []
        all_labels = []
        for ax, (cfg, cfg_rows) in zip(axes, selected):
            cfg_rows = sorted(cfg_rows, key=lambda row: int(float(row['timeout_ms'])))
            timeouts = np.array([int(float(row['timeout_ms'])) for row in cfg_rows])
            x = np.arange(len(timeouts))
            schedule = np.array([f(row['schedule_tok_s']) for row in cfg_rows])
            baseline = f(cfg_rows[0]['baseline_tok_s'])
            ppl = np.array([f(row['schedule_ppl']) for row in cfg_rows])
            speed = np.array([f(row['speedup_vs_baseline']) for row in cfg_rows])

            bars = ax.bar(x, schedule, width=0.62, color='#fc8d62', edgecolor='#7f3b20', linewidth=0.5, label='调度吞吐')
            base_line = ax.axhline(baseline, color='#d73027', linestyle='--', linewidth=1.8, label='基线吞吐')
            ax.set_ylim(0, max(float(schedule.max()), baseline) * 1.22)
            ax.grid(axis='y', linestyle='--', alpha=0.25)
            ax.set_title(f'{cfg}\n负载 {short_loads(cfg_rows[0]["loads"], occ)}')
            ax.set_xticks(x)
            ax.set_xticklabels([str(t) for t in timeouts], rotation=0)
            ax.set_xlabel('超时 (ms)')

            ax2 = ax.twinx()
            ppl_line = ax2.plot(x, ppl, color='#2ca25f', marker='o', linewidth=1.7, markersize=3.4, label='PPL')[0]
            ax2.set_ylim(0, max(40.0, float(ppl.max()) * 1.15))
            ax2.spines['top'].set_visible(False)

            # Annotate only the best speedup point for each config.
            best_i = int(np.argmax(speed))
            rect = bars[best_i]
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(float(schedule.max()), baseline) * 0.035,
                f'{speed[best_i]:.1f}x',
                ha='center',
                va='bottom',
                fontsize=8,
                color='#7f3b20',
            )

            if ax is axes[0]:
                ax.set_ylabel('吞吐量 (tok/s)')
            if ax is axes[-1]:
                ax2.set_ylabel('PPL')
            else:
                ax2.set_yticklabels([])

            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            if not all_handles:
                all_handles = handles + handles2
                all_labels = labels + labels2

        fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncols=3, frameon=False)
        fig.subplots_adjust(left=0.065, right=0.94, top=0.78, bottom=0.25, wspace=0.28)

        png = OUT / f'timeout_top3_subplots_{occ}occ.png'
        pdf = OUT / f'timeout_top3_subplots_{occ}occ.pdf'
        fig.savefig(png, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        manifest.append({
            'occupied_count': occ,
            'min_baseline_tok_s': MIN_BASELINE_TOK_S,
            'selected_configs': ','.join(cfg for cfg, _ in selected),
            'mean_speedups': ','.join(f'{mean_speed(cfg_rows):.6f}' for _, cfg_rows in selected),
            'baselines_tok_s': ','.join(f'{f(cfg_rows[0]["baseline_tok_s"]):.6f}' for _, cfg_rows in selected),
            'png': str(png),
            'pdf': str(pdf),
        })

    with (OUT / 'top3_subplots_manifest.csv').open('w', newline='') as fcsv:
        fieldnames = ['occupied_count', 'min_baseline_tok_s', 'selected_configs', 'mean_speedups', 'baselines_tok_s', 'png', 'pdf']
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    print(OUT)
    for item in manifest:
        print(item)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
