from typing import List
import argparse
import csv
import os
import sys
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr, kendalltau

import_dir = '/'.join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, import_dir)
from utils.embedding_layer_metrics import mean_cossim_across_last_n_layers


RENAME_MAP = {
    'gpt2': 'GPT2\n(0.124B)',
    'gpt2-medium': 'GPT2-medium\n(0.355B)',
    'gpt2-large': 'GPT2-large\n(0.774B)',
    'gpt2-xl': 'GPT2-xl\n(1.5B)',
    'albert-base-v2': 'ALBERT-base-v2',
    'albert-large-v2': 'ALBERT-large-v2',
    'albert-xlarge-v2': 'ALBERT-xlarge-v2',
    'albert-xxlarge-v2': 'ALBERT-xxlarge-v2',
    'openai-community-gpt2': 'GPT2\n(0.124B)',
    'openai-community-gpt2-medium': 'GPT2-medium\n(0.335B)',
    'openai-community-gpt2-large': 'GPT2-large\n(0.774B)',
    'openai-community-gpt2-xl': 'GPT2-xl\n(1.5B)',
    'Qwen-Qwen-1_8B': 'Qwen-1.8B',
    'Qwen-Qwen-7B': 'Qwen-7B',
    'Qwen-Qwen-14B': 'Qwen-14B',
    'Qwen-Qwen-72B': 'Qwen-72B',
    'Qwen-Qwen2.5-0.5B': 'Qwen2.5-0.5B',
    'Qwen-Qwen2.5-1.5B': 'Qwen2.5-1.5B',
    'Qwen-Qwen2.5-3B': 'Qwen2.5-3B',
    'Qwen-Qwen2.5-7B': 'Qwen2.5-7B',
    'Qwen-Qwen2.5-14B': 'Qwen2.5-14B',
    'Qwen-Qwen2.5-32B': 'Qwen2.5-32B',
    'Qwen-Qwen2.5-72B': 'Qwen2.5-72B',
    'Qwen-Qwen2.5-Math-1.5B': 'Qwen2.5-Math-1.5B',
    'Qwen-Qwen2.5-Math-7B': 'Qwen2.5-Math-7B',
    'Qwen-Qwen2.5-Math-14B': 'Qwen2.5-Math-14B',
    'Qwen-Qwen2.5-Math-32B': 'Qwen2.5-Math-32B',
    'Qwen-Qwen3-0.6B': 'Qwen3-0.6B',
    'Qwen-Qwen3-1.7B': 'Qwen3-1.7B',
    'Qwen-Qwen3-4B': 'Qwen3-4B',
    'Qwen-Qwen3-8B': 'Qwen3-8B',
    'Qwen-Qwen3-14B': 'Qwen3-14B',
    'Qwen-Qwen3-32B': 'Qwen3-32B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-R1-Distill-\nQwen2.5-Math-1.5B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-7B': 'DeepSeek-R1-Distill-\nQwen2.5-Math-7B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-14B': 'DeepSeek-R1-Distill-\nQwen2.5-14B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-32B': 'DeepSeek-R1-Distill-\nQwen2.5-32B',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B': 'DeepSeek-R1-Distill-\nLlama-8B',
    'meta-llama-Llama-3.1-8B': 'Llama-3.1-8B',
    'meta-llama-Llama-3.2-1B': 'Llama-3.2-1B',
    'meta-llama-Llama-3.2-3B': 'Llama-3.2-3B',
    'allenai-OLMo-2-0425-1B': 'Olmo2-1B',
    'allenai-OLMo-2-1124-7B': 'Olmo2-7B',
    'allenai-OLMo-2-1124-13B': 'Olmo2-13B',
    'allenai-OLMo-2-0325-32B': 'Olmo2-32B',
    'bigscience-bloom-560m': 'Bloom-0.56B',
    'bigscience-bloom-1b1': 'Bloom-1.1B',
    'bigscience-bloom-1b7': 'Bloom-1.7B',
    'bigscience-bloom-3b': 'Bloom-3B',
    'bigscience-bloom-7b1': 'Bloom-7.1B',
}


def display_name(model_id: str) -> str:
    """Map Hugging Face ids (with '/') to RENAME_MAP keys (with '-')."""
    k = '-'.join(model_id.split('/'))
    return RENAME_MAP.get(k, k)


def plot_condensation_trend(model_id_list: List[str],
                            cossim_matrix_list: List[np.array],
                            spearman_corr_list: List[float],
                            kendall_tau_list: List[float],
                            mean_last_n_by_model: List[np.ndarray] = None,
                            paired: bool = False,
                            bins: int = 128,
                            save_path: str = None):

    plt.rcParams['font.family'] = 'sans-serif'
    include_last_n_panel = mean_last_n_by_model is not None
    if paired:
        num_models_each = len(model_id_list) // 2
        width_ratios = [1.8, 0.05] + [1] * num_models_each
        if include_last_n_panel:
            fig = plt.figure(figsize=(9 * np.sum(width_ratios), 22))
            gs = gridspec.GridSpec(
                3, num_models_each + 2, width_ratios=width_ratios, height_ratios=[1, 1, 0.75]
            )
        else:
            fig = plt.figure(figsize=(9 * np.sum(width_ratios), 18))
            gs = gridspec.GridSpec(2, num_models_each + 2, width_ratios=width_ratios)
    else:
        width_ratios = [1, 0.05] + [1] * len(model_id_list)
        if include_last_n_panel:
            fig = plt.figure(figsize=(8.5 * np.sum(width_ratios), 11))
            gs = gridspec.GridSpec(
                2, len(model_id_list) + 2, width_ratios=width_ratios, height_ratios=[2.2, 1.0]
            )
        else:
            fig = plt.figure(figsize=(8.5 * np.sum(width_ratios), 8))
            gs = gridspec.GridSpec(1, len(model_id_list) + 2, width_ratios=width_ratios)

    for model_idx in range(len(model_id_list)):
        if paired:
            ax = fig.add_subplot(gs[int(model_idx >= num_models_each), 2 + model_idx % num_models_each])
        else:
            ax = fig.add_subplot(gs[0, 2 + model_idx])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        layer_indices, hist_data = [], []
        n_layers = len(cossim_matrix_list[model_idx])
        denom = max(n_layers - 1, 1)
        for layer_idx in range(n_layers):
            cossim_arr = cossim_matrix_list[model_idx][layer_idx].flatten()
            hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
            hist_data.append(hist)
            layer_indices.append(layer_idx / denom)
        hist_matrix = np.array(hist_data)

        im = ax.imshow(hist_matrix.T, aspect="auto", origin="lower", cmap='Reds',
                       extent=[0, layer_indices[-1], -1, 1], vmin=0, vmax=10)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.set_title(rf'{display_name(model_id_list[model_idx])}', pad=24,
                     fontfamily='monospace', fontname='cmtt10', fontsize=36)
        ax.set_xlabel('Layer Fraction', fontsize=36)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_ylim([-0.25, 1])
        ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
        if model_idx == 0 or (paired and model_idx == num_models_each):
            ax.set_ylabel('Cosine Similarity', fontsize=36)

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(axis='both', which='major', labelsize=26)
        cbar.ax.set_title('Probability\nDensity', fontsize=20, pad=20)

    if paired:
        plot_trend_metrics_paired(gs[0:2, 0] if include_last_n_panel else gs[:, 0],
                                  fig, model_id_list, spearman_corr_list, kendall_tau_list)
        if include_last_n_panel:
            plot_mean_last_n_panel(fig, gs[2, :], model_id_list, mean_last_n_by_model)
    else:
        plot_trend_metrics(gs[0, 0], fig, model_id_list, spearman_corr_list, kendall_tau_list)
        if include_last_n_panel:
            plot_mean_last_n_panel(fig, gs[1, :], model_id_list, mean_last_n_by_model)

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_mean_last_n_panel(fig, subplotspec, model_id_list, mean_last_n_by_model):
    ax = fig.add_subplot(subplotspec)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(model_id_list), 1)))
    for i, model_id in enumerate(model_id_list):
        y = mean_last_n_by_model[i]
        if y.size == 0:
            continue
        n_axis = np.arange(1, len(y) + 1)
        ax.plot(n_axis, y, color=colors[i % len(colors)], linewidth=2.5, marker='o', markersize=4,
                label=display_name(model_id).replace('\n', ' '))
    ax.set_xlabel(r'$N$ (last layers)', fontsize=22)
    ax.set_ylabel('Mean cossim\n(avg. over last $N$ layers)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.set_title('Mean layer-mean cosine similarity over last $N$ hidden layers', fontsize=20, pad=8)
    return ax


def plot_trend_metrics(subplotspec, fig, model_id_list, spearman_corr_list, kendall_tau_list):
    ax = fig.add_subplot(subplotspec)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(model_id_list)))
    for i in range(len(model_id_list) - 1):
        ax.plot(np.arange(len(model_id_list))[i : i+2], spearman_corr_list[i : i+2], color=colors[i], linewidth=4)
    ax.scatter(np.arange(len(model_id_list)), spearman_corr_list, color=colors, s=120)
    ax.set_ylabel('Spearman Correlation', labelpad=12, fontsize=30, color=colors[-1])
    ax.set_ylim([-1, 1.05])
    ax.set_xlim([-0.3, len(model_id_list) - 0.7])
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xticks(np.arange(len(model_id_list)))
    ax.set_xticklabels([display_name(model_id) for model_id in model_id_list], fontsize=24, rotation=45, ha='right')

    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(model_id_list)))
    for i in range(len(model_id_list) - 1):
        ax2.plot(np.arange(len(model_id_list))[i : i+2], kendall_tau_list[i : i+2], color=colors[i], linewidth=4)
    ax2.scatter(np.arange(len(model_id_list)), kendall_tau_list, color=colors, s=120)
    ax2.set_ylabel("Kendall's Tau", labelpad=36, fontsize=30, rotation=270, color=colors[-1])
    ax2.set_ylim([-1, 1.05])
    ax2.tick_params(axis='both', which='major', labelsize=26)
    return ax

def plot_trend_metrics_paired(subplotspec, fig, model_id_list, spearman_corr_list, kendall_tau_list):
    assert len(model_id_list) % 2 == 0, 'Paired plotting require even number of models!'
    ax = fig.add_subplot(subplotspec)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    num_models_each = len(model_id_list) // 2
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, num_models_each))
    for i in range(num_models_each - 1):
        ax.plot(np.arange(num_models_each)[i : i+2], spearman_corr_list[i : i+2], color=colors[i], linewidth=4)
    for i in range(num_models_each - 1):
        ax.plot(num_models_each + np.arange(num_models_each)[i : i+2],
                spearman_corr_list[num_models_each + i : num_models_each + i+2], color=colors[i], linewidth=4)
    for i in range(num_models_each):
        ax.plot([i, num_models_each + i], [spearman_corr_list[i], spearman_corr_list[num_models_each + i]],
                color=colors[i], linewidth=2, linestyle='--')
    ax.scatter(np.arange(num_models_each), spearman_corr_list[:num_models_each], color=colors, s=120)
    ax.scatter(num_models_each + np.arange(num_models_each), spearman_corr_list[num_models_each:], color=colors, s=120)
    ax.set_ylabel('Spearman Correlation', labelpad=12, fontsize=30, color=colors[-1])
    ax.set_ylim([-1, 1.05])
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xticks(np.arange(len(model_id_list)))
    ax.set_xticklabels([display_name(model_id) for model_id in model_id_list], fontsize=24, rotation=45, ha='right')

    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, num_models_each))
    for i in range(num_models_each - 1):
        ax2.plot(np.arange(num_models_each)[i : i+2], kendall_tau_list[i : i+2], color=colors[i], linewidth=4)
    for i in range(num_models_each - 1):
        ax2.plot(num_models_each + np.arange(num_models_each)[i : i+2],
                 kendall_tau_list[num_models_each + i : num_models_each + i+2], color=colors[i], linewidth=4)
    for i in range(num_models_each):
        ax2.plot([i, num_models_each + i], [kendall_tau_list[i], kendall_tau_list[num_models_each + i]],
                 color=colors[i], linewidth=2, linestyle='--')
    ax2.scatter(np.arange(num_models_each), kendall_tau_list[:num_models_each], color=colors, s=120)
    ax2.scatter(num_models_each + np.arange(num_models_each), kendall_tau_list[num_models_each:], color=colors, s=120)
    ax2.set_ylabel("Kendall's Tau", labelpad=36, fontsize=30, rotation=270, color=colors[-1])
    ax2.set_ylim([-1, 1.05])
    ax2.tick_params(axis='both', which='major', labelsize=26)
    return ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--model-id', type=str, default='albert-base-v2', nargs='+')
    parser.add_argument('--model-family', type=str, default='albert')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument(
        '--output-tag',
        type=str,
        default=None,
        help='Optional tag matching results_cossim_{dataset}_{tag}.npz from compute runs.',
    )
    parser.add_argument(
        '--table-only',
        action='store_true',
        help='Skip figure; only print metrics and write CSV.',
    )
    parser.add_argument(
        '--include-last-n-panel',
        action='store_true',
        help='Add a last-N layer mean panel to the trend figure.',
    )
    parser.add_argument(
        '--write-csv',
        action='store_true',
        help='Write last-N metrics to CSV.',
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Override CSV path (default: ../visualization/transformer/_trend/{family}_last_n_metrics.csv).',
    )
    args = parser.parse_args()

    out_tag = (args.output_tag or '').strip().replace('/', '_').replace(' ', '_')
    tag_suffix = f'_{out_tag}' if out_tag else ''

    model_id_list, cossim_matrix_list = [], []
    spearman_corr_list, kendall_tau_list = [], []
    mean_last_n_by_model: List[np.ndarray] = []
    rows_csv = []

    for model_id in args.model_id:
        model_name_cleaned = '-'.join(model_id.split('/'))
        npz_cossim = (
            f'../visualization/transformer/{model_name_cleaned}/'
            f'results_cossim_{args.dataset}{tag_suffix}.npz'
        )
        cossim_matrix_by_layer = np.load(npz_cossim)['cossim_matrix_by_layer']
        mean_cossim_by_layer = cossim_matrix_by_layer.mean(axis=(1, 2))

        spearman_corr, _ = spearmanr(mean_cossim_by_layer, np.arange(len(mean_cossim_by_layer)))
        kendell_tau, _ = kendalltau(mean_cossim_by_layer, np.arange(len(mean_cossim_by_layer)))
        mean_last_n = mean_cossim_across_last_n_layers(mean_cossim_by_layer)

        model_id_list.append(model_id)
        cossim_matrix_list.append(cossim_matrix_by_layer)
        spearman_corr_list.append(spearman_corr)
        kendall_tau_list.append(kendell_tau)
        mean_last_n_by_model.append(mean_last_n)

        for n, val in enumerate(mean_last_n, start=1):
            rows_csv.append(
                {
                    'model_id': model_id,
                    'N': n,
                    'mean_cossim_last_n_layers': val,
                    'spearman_vs_layer_index': spearman_corr,
                    'kendall_tau_vs_layer_index': kendell_tau,
                }
            )
        if args.table_only or args.include_last_n_panel or args.write_csv or args.csv_path:
            print(f'\n=== {model_id} ===')
            print('Spearman(layer_mean_cossim, layer_index):', spearman_corr)
            print("Kendall's tau:", kendell_tau)
            print('Mean cossim averaged over last N layers (N=1..L):')
            for n, val in enumerate(mean_last_n, start=1):
                print(f'  N={n}: {val:.6f}')

    trend_dir = '../visualization/transformer/_trend'
    should_write_csv = args.write_csv or args.csv_path is not None or args.table_only
    if should_write_csv and rows_csv:
        csv_path = args.csv_path or os.path.join(
            trend_dir, f'{args.model_family}_last_n_metrics{tag_suffix}.csv'
        )
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        print(f'\nWrote {csv_path}')

    if not args.table_only:
        plot_condensation_trend(
            model_id_list=model_id_list,
            cossim_matrix_list=cossim_matrix_list,
            spearman_corr_list=spearman_corr_list,
            kendall_tau_list=kendall_tau_list,
            mean_last_n_by_model=mean_last_n_by_model if args.include_last_n_panel else None,
            paired=args.paired,
            save_path=os.path.join(trend_dir, f'{args.model_family}{tag_suffix}.png'),
        )
