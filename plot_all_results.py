"""
複数実験の各メトリクスをターン単位で重ねてプロットする。
CivRealm 論文 Figure 9 相当。

使い方:
  python3 plot_all_results.py                       # 全実験
  python3 plot_all_results.py 2026-06-15            # 指定日のみ
  python3 plot_all_results.py 2026-06-14 2026-06-15 # 複数日指定
"""
import json
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = 'results'
SCORES_DIR  = 'game_scores'
OUT_DIR     = 'plot_all_results'

# プロットするメトリクスと表示名
METRICS = {
    'score':          'Score',
    'population':     'Population',
    'cities':         'Cities',
    'military_units': 'Military Units',
    'researched_techs': 'Researched Techs',
    'economics':      'Economics',
    'production':     'Production',
    'gold':           'Gold',
}

AGENT_COLORS = {
    'BaselineLanguageAgent': '#1f77b4',
    'QwenMastabaAgent':      '#ff7f0e',
}
AGENT_LABELS = {
    'BaselineLanguageAgent': 'Baseline',
    'QwenMastabaAgent':      'Qwen Mastaba',
}

# ------------------------------------------------------------------
# データ読み込み
# ------------------------------------------------------------------

def find_score_dir(timestamp: str):
    """result の timestamp に対応する game_scores ディレクトリを返す。"""
    prefix = timestamp.replace(' ', '-')[:16]
    for d in os.listdir(SCORES_DIR):
        if d[:16] == prefix:
            return os.path.join(SCORES_DIR, d)
    return None


def load_runs(dates=None):
    """
    agent 種別・メトリクスごとに score 配列のリストを返す。
    dates: ['2026-06-15', ...] のように指定すると該当日のみ読み込む。None なら全て。
    {agent_name: {metric: [array, array, ...]}}
    """
    runs = {}

    for rpath in sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json'))):
        with open(rpath) as f:
            result = json.load(f)

        agent = result.get('agent', 'unknown')
        ts    = result.get('timestamp', '')
        my_id = str(result.get('my_player_id', 0))

        # 日付フィルタ: ts は '2026-06-15 09:16:18' 形式
        if dates is not None and ts[:10] not in dates:
            continue

        score_dir = find_score_dir(ts)
        if score_dir is None:
            print(f'[skip] no game_scores match for {os.path.basename(rpath)}')
            continue

        eval_path = os.path.join(score_dir, 'evaluations.json')
        if not os.path.exists(eval_path):
            print(f'[skip] evaluations.json not found in {score_dir}')
            continue

        with open(eval_path) as f:
            evals = json.load(f)

        if agent not in runs:
            runs[agent] = {m: [] for m in METRICS}

        for metric in METRICS:
            values = evals.get(metric, {}).get(my_id)
            if values is not None:
                runs[agent][metric].append(np.array(values, dtype=float))

    return runs


# ------------------------------------------------------------------
# プロット
# ------------------------------------------------------------------

def plot_metric(ax, agent_runs, metric):
    """1つの ax にメトリクスの全 agent を描画する。"""
    legend_patches = []

    for agent, metric_dict in agent_runs.items():
        series_list = metric_dict.get(metric, [])
        if not series_list:
            continue

        color = AGENT_COLORS.get(agent, '#888888')
        label = AGENT_LABELS.get(agent, agent)
        n     = len(series_list)

        max_len = max(len(s) for s in series_list)
        mat = np.full((n, max_len), np.nan)
        for i, s in enumerate(series_list):
            mat[i, :len(s)] = s

        turns = np.arange(1, max_len + 1)

        for i in range(n):
            ax.plot(turns, mat[i], color=color, alpha=0.2, linewidth=0.8)

        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        ax.plot(turns, mean, color=color, linewidth=2.0)
        ax.fill_between(turns, mean - std, mean + std, color=color, alpha=0.15)

        legend_patches.append(
            mpatches.Patch(color=color, label=f'{label} (n={n})')
        )

    ax.set_ylabel(METRICS[metric], fontsize=11)
    ax.legend(handles=legend_patches, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_all(runs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n_metrics = len(METRICS)
    ncols = 2
    nrows = (n_metrics + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        plot_metric(ax, runs, metric)
        ax.set_title(METRICS[metric], fontsize=12)
        if idx >= n_metrics - ncols:
            ax.set_xlabel('Turn', fontsize=11)

    # 余った axes を非表示
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Game Metrics over Turns (all runs)', fontsize=14, y=1.01)
    plt.tight_layout()

    combined_path = os.path.join(out_dir, 'all_metrics.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {combined_path}')
    plt.show()

    # メトリクスごとに個別保存
    for metric in METRICS:
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        plot_metric(ax2, runs, metric)
        ax2.set_xlabel('Turn', fontsize=12)
        ax2.set_title(f'{METRICS[metric]} over Turns', fontsize=13)
        plt.tight_layout()
        path = os.path.join(out_dir, f'{metric}.png')
        plt.savefig(path, dpi=150)
        plt.close(fig2)
        print(f'Saved: {path}')


# ------------------------------------------------------------------

if __name__ == '__main__':
    dates = sys.argv[1:] if len(sys.argv) > 1 else None
    if dates:
        print(f'Filter: {dates}')
    runs = load_runs(dates)
    for agent, metric_dict in runs.items():
        n = len(next(iter(metric_dict.values())))
        print(f'{agent}: {n} runs')
    if not any(len(next(iter(m.values()))) > 0 for m in runs.values()):
        print('No runs found. Check the date format (YYYY-MM-DD).')
    else:
        plot_all(runs, OUT_DIR)
