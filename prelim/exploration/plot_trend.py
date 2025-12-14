from typing import List
import argparse
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr, kendalltau


RENAME_MAP = {
    'albert-base-v2': 'ALBERT-base-v2',
    'albert-large-v2': 'ALBERT-large-v2',
    'albert-xlarge-v2': 'ALBERT-xlarge-v2',
    'albert-xxlarge-v2': 'ALBERT-xxlarge-v2',
    'openai-community-gpt2': 'GPT2',
    'openai-community-gpt2-medium': 'GPT2-medium',
    'openai-community-gpt2-large': 'GPT2-large',
    'openai-community-gpt2-xl': 'GPT2-xl',
    'Qwen-Qwen-1_8B': 'Qwen-1.8B',
    'Qwen-Qwen-7B': 'Qwen-7B',
    'Qwen-Qwen-14B': 'Qwen-14B',
    'Qwen-Qwen2.5-0.5B': 'Qwen2.5-0.5B',
    'Qwen-Qwen2.5-1.5B': 'Qwen2.5-1.5B',
    'Qwen-Qwen2.5-3B': 'Qwen2.5-3B',
    'Qwen-Qwen2.5-7B': 'Qwen2.5-7B',
    'Qwen-Qwen2.5-Math-1.5B': 'Qwen2.5-Math-1.5B',
    'Qwen-Qwen2.5-Math-7B': 'Qwen2.5-Math-7B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B': 'DeepSeek-R1-Distill-\nQwen2.5-Math-1.5B',
    'deepseek-ai-DeepSeek-R1-Distill-Qwen-7B': 'DeepSeek-R1-Distill-\nQwen2.5-Math-7B',
}

def plot_condensation_trend(model_id_list: List[str],
                            cossim_matrix_list: List[np.array],
                            spearman_corr_list: List[float],
                            kendall_tau_list: List[float],
                            bins: int = 128,
                            save_path: str = None):

    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(10 * (len(model_id_list) + 1), 8))
    gs = gridspec.GridSpec(1, len(model_id_list) + 2, width_ratios=[1] * len(model_id_list) + [0.05, 0.9])

    for model_idx in range(len(model_id_list)):
        ax = fig.add_subplot(gs[0, model_idx])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        layer_indices, hist_data = [], []
        for layer_idx in range(len(cossim_matrix_list[model_idx])):
            cossim_arr = cossim_matrix_list[model_idx][layer_idx].flatten()
            hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
            hist_data.append(hist)
            layer_indices.append(layer_idx / (len(cossim_matrix_list[model_idx]) - 1))
        hist_matrix = np.array(hist_data)

        im = ax.imshow(hist_matrix.T, aspect="auto", origin="lower", cmap='Reds',
                       extent=[0, layer_indices[-1], -1, 1], vmin=0, vmax=10)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.set_title(rf'{RENAME_MAP[model_id_list[model_idx]]}', pad=24,
                     fontfamily='monospace', fontname='cmtt10', fontsize=36)
        ax.set_xlabel('Layer Fraction', fontsize=36)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        if model_idx == 0:
            ax.set_ylabel('Cosine Similarity', fontsize=36)

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(axis='both', which='major', labelsize=26)
        cbar.ax.set_title('Probability\nDensity', fontsize=20, pad=20)

    ax = fig.add_subplot(gs[0, -1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(model_id_list)))
    for i in range(len(model_id_list) - 1):
        ax.plot(np.arange(len(model_id_list))[i : i+2], spearman_corr_list[i : i+2], color=colors[i], linewidth=4)
    ax.scatter(np.arange(len(model_id_list)), spearman_corr_list, color=colors, s=120)
    ax.set_ylabel('Spearman Correlation', labelpad=12, fontsize=30, color=colors[-1])
    ax.set_ylim([-1, 1.02])
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xticks(np.arange(len(model_id_list)))
    ax.set_xticklabels([RENAME_MAP[model_id] for model_id in model_id_list], fontsize=18, rotation=30, ha='right')

    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(model_id_list)))
    for i in range(len(model_id_list) - 1):
        ax2.plot(np.arange(len(model_id_list))[i : i+2], kendall_tau_list[i : i+2], color=colors[i], linewidth=4)
    ax2.scatter(np.arange(len(model_id_list)), kendall_tau_list, color=colors, s=120)
    ax2.set_ylabel("Kendall's Tau", labelpad=36, fontsize=30, rotation=270, color=colors[-1])
    ax2.set_ylim([-1, 1.02])
    ax2.tick_params(axis='both', which='major', labelsize=26)

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--model-id', type=str, default='albert-base-v2', nargs='+')
    parser.add_argument('--model-family', type=str, default='albert')
    parser.add_argument('--dataset', type=str, default='wikipedia')
    args = parser.parse_args()

    model_id_list, cossim_matrix_list, spearman_corr_list, kendall_tau_list = [], [], [], []
    for model_id in args.model_id:
        model_name_cleaned = '-'.join(model_id.split('/'))
        npz_cossim = f'../visualization/transformer/{model_name_cleaned}/results_cossim_{args.dataset}.npz'
        cossim_matrix_by_layer = np.load(npz_cossim)['cossim_matrix_by_layer']
        mean_cossim_by_layer = cossim_matrix_by_layer.mean(axis=(1,2))

        spearman_corr, _ = spearmanr(mean_cossim_by_layer, np.arange(len(mean_cossim_by_layer)))
        kendell_tau, _ = kendalltau(mean_cossim_by_layer, np.arange(len(mean_cossim_by_layer)))

        model_id_list.append(model_id)
        cossim_matrix_list.append(cossim_matrix_by_layer)
        spearman_corr_list.append(spearman_corr)
        kendall_tau_list.append(kendell_tau)

    plot_condensation_trend(model_id_list=model_id_list,
                            cossim_matrix_list=cossim_matrix_list,
                            spearman_corr_list=spearman_corr_list,
                            kendall_tau_list=kendall_tau_list,
                            save_path=f'../visualization/transformer/_trend/{args.model_family}.png')
