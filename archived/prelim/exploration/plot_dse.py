import os
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    save_path = '../visualization/transformer/dse_observation.png'

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = 2

    fig = plt.figure(figsize=(18, 4))

    model_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    # model_names = ['albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2']
    for subplot_idx, model_name in enumerate(model_names):
        ax = fig.add_subplot(1, len(model_names), subplot_idx + 1)
        load_path = f'../visualization/transformer/{model_name}/results_DSE.npz'
        if not os.path.isfile(load_path):
            continue
        DSE_by_layer = np.load(load_path)['DSE_by_layer']
        repetitions, layers = DSE_by_layer.shape

        ax.scatter(np.arange(layers) / layers, DSE_by_layer.mean(axis=0), color='black', facecolor='skyblue', zorder=2)
        ax.errorbar(np.arange(layers) / layers, DSE_by_layer.mean(axis=0), yerr=DSE_by_layer.std(axis=0),
                    linestyle='none', capsize=4, color='black', zorder=1)

        # Fit a first-degree polynomial (linear fit)
        x = np.arange(layers) / layers
        y = DSE_by_layer.mean(axis=0)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        ax.plot(x, y_fit, color='firebrick', linestyle='--', linewidth=2, zorder=3, label=fr'slope = $\mathbf{{{slope:.3f}}}$')

        ax.legend(loc='upper right')
        ax.set_xlabel('Normalized Layer Index')
        ax.set_xlim([-0.1, 1.1])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels([0, 0.5, 1])
        ax.set_ylim([0, 10])
        if subplot_idx == 0:
            ax.set_ylabel('Diffusion Spectral Entropy')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(save_path, dpi=300)
