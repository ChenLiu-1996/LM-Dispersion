from typing import List
import argparse
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import login
import matplotlib.pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, import_dir)
from utils.text_data import get_random_long_text
from dse.dse import diffusion_spectral_entropy
from utils.embedding_layer_metrics import (
    hfc_lfc_ratio,
    log_hfc_frobenius_relative,
    pairwise_inner_products,
    per_layer_hfc_lfc_ratio,
    per_layer_inner_products,
    per_layer_log_hfc_frobenius,
    per_layer_singular_value_entropy_and_mev,
    singular_value_entropy_and_mev,
)


def organize_embeddings(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in embeddings:
        z = z.squeeze(0).float().cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer

def compute_cosine_similarities(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in embeddings:
        z = normalize(z, axis=1)
        cossim_matrix = np.matmul(z, z.T).clip(-1, 1)  # Clipping to correct occasional rounding error.
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def plot_similarity_histograms(cossim_matrix_by_layer: List[np.ndarray],
                               save_path: str = None,
                               step: int = 1,
                               bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    num_plots = len(selected)

    # Auto-determine layout (rows x cols) to be roughly square
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    # Global y-axis limit for consistent scaling
    max_density = max(
        np.histogram(data, bins=bins, density=True)[0].max() for _, data in selected
    )

    for ax, (i, cossim_matrix) in zip(axes, selected):
        cossim_arr = cossim_matrix.flatten()
        IQR = np.percentile(cossim_arr, 75) - np.percentile(cossim_arr, 25)
        bin_width = 2 * IQR / len(cossim_arr) ** (1 / 3)
        optimal_bins = max(10, int((max(cossim_arr) - min(cossim_arr)) / bin_width))

        ax.hist(cossim_arr, bins=optimal_bins, density=True, histtype='step', color='#d62728', linewidth=1.5)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(f'Layer {i}', fontsize=32)
        ax.set_xlim([-0.4, 1.1])
        ax.set_ylim(0, max_density * 1.1)

    # Turn off unused axes
    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_similarity_heatmap(cossim_matrix_by_layer: List[np.ndarray],
                            save_path: str = None,
                            step: int = 1,
                            bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]

    layer_indices, hist_data = [], []
    for (layer_idx, cossim_matrix) in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx)
    hist_matrix = np.array(hist_data)

    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    im = ax.imshow(hist_matrix.T, aspect="auto", origin="lower", cmap='Reds',
                   extent=[0, layer_indices[-1], -1, 1], vmin=0, vmax=10)

    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.set_xlabel('Layer', fontsize=36)
    ax.set_ylabel('Cosine Similarity', fontsize=36)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(axis='both', which='major', labelsize=26)
    cbar.ax.set_title('Probability\nDensity', fontsize=20, pad=20)

    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_probability(cossim_matrix_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(10, 8))
    for subplot_idx, threshold in enumerate([0.9, 0.95, 0.99, 1.0]):
        curr_prob = [(cossim_matrix.flatten() > threshold).sum() / len(cossim_matrix.flatten())
                     for cossim_matrix in cossim_matrix_by_layer]
        ax = fig.add_subplot(2, 2, subplot_idx + 1)
        ax.plot(curr_prob, marker='o', linewidth=2, color='#2ca02c')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(fr'Cosine Similarity $\geq$ {threshold}', fontsize=18)
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(r'P(cossim(Embedding)$\approx$1) per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_entropy(cossim_matrix_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(12, 6))
    for subplot_idx, entropy_type in enumerate(['Shannon', 'von Neumann']):
        ax = fig.add_subplot(1, 2, subplot_idx + 1)
        if entropy_type == 'Shannon':
            cmap = plt.get_cmap('Greens')
            for num_bins, cmap_idx in zip([64, 256, 1024, 4096], [0.4, 0.6, 0.8, 1.0]):
                entropy_arr = [compute_entropy(cossim_matrix, entropy_type=entropy_type, num_bins=num_bins)
                               for cossim_matrix in cossim_matrix_by_layer]
                ax.plot(entropy_arr, marker='o', linewidth=2, color=cmap(cmap_idx), label=f'{num_bins} bins')
            ax.legend(loc='lower right')
        else:
            entropy_arr = [compute_entropy(cossim_matrix, entropy_type=entropy_type)
                           for cossim_matrix in cossim_matrix_by_layer]
            ax.plot(entropy_arr, marker='o', linewidth=2, color='#2ca02c')

        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f'{entropy_type} Entropy', fontsize=18)
        ax.set_xlabel('Layer', fontsize=14)
        ax.set_ylabel('Entropy', fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('cossim(Embedding) Entropy per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_DSE(embeddings_by_layer: List[np.ndarray], save_path: str = None):
    fig = plt.figure(figsize=(16, 8))
    for subplot_idx, l2_normalize in enumerate([False, True]):
        if l2_normalize:
            normalize_func = normalize
        else:
            normalize_func = lambda x: x
        ax = fig.add_subplot(1, 2, subplot_idx + 1)
        for sigma, color_base in zip([1, 5, 10], ['Blues', 'Reds', 'Greens']):
            cmap = plt.get_cmap(color_base)
            for diffusion_t, cmap_idx in zip([1, 2, 5, 10], [0.4, 0.6, 0.8, 1.0]):
                entropy_arr = [diffusion_spectral_entropy(normalize_func(embeddings), gaussian_kernel_sigma=sigma, t=diffusion_t)
                               for embeddings in embeddings_by_layer]
                ax.plot(entropy_arr, marker='o', linewidth=2, color=cmap(cmap_idx), label=f'$\sigma$ = {sigma}, t = {diffusion_t}')
        ax.legend(loc='upper right', ncols=3)

        ax.set_ylim([0, ax.get_ylim()[1] * 1.2])
        ax.tick_params(axis='both', which='major', labelsize=18)
        if l2_normalize:
            ax.set_title('With L2 normalization', fontsize=18)
        else:
            ax.set_title('Without L2 normalization', fontsize=18)
        ax.set_xlabel('Layer', fontsize=18)
        ax.set_ylabel('Entropy', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Embedding DSE per Layer', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def compute_entropy(matrix: np.ndarray, entropy_type: str, num_bins: int = 256):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    if entropy_type == 'Shannon':
        vec = matrix.flatten()
        # Min-Max scale.
        vec = (vec - np.min(vec)) / (np.max(vec) - np.min(vec))
        # Binning.
        bins = np.linspace(0, 1, num_bins + 1)[:-1]
        vec_binned = np.digitize(vec, bins=bins)
        # Count probability.
        counts = np.unique(vec_binned, axis=0, return_counts=True)[1]
        prob = counts / np.sum(counts)
        # Compute entropy.
        prob = prob + np.finfo(float).eps
        entropy = -np.sum(prob * np.log2(prob))

    elif entropy_type == 'von Neumann':
        '''
        NOTE: This can be still considered DSE, if we set the kernel function as dot product instead of Gaussian kernel.
        '''
        # Ensure Hermitian
        assert np.allclose(matrix, matrix.conj().T)
        # Eigen-decomposition
        eigvals = np.linalg.eigvalsh(matrix)
        # Clip small negative eigenvalues due to numerical errors
        eigvals = np.clip(eigvals, 0, np.inf)
        # Normalize to ensure trace = 1
        eigvals = eigvals / eigvals.sum()
        # Count probability.
        prob = eigvals[eigvals > 0]
        # Compute entropy.
        prob = prob + np.finfo(float).eps
        entropy = -np.sum(prob * np.log2(prob))

    return entropy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--tokenizer-id', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='/home/cl2482/palmer_scratch/.cache/')
    parser.add_argument('--model-id', type=str, default='albert-base-v2')
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--huggingface-token', type=str, default=None)
    parser.add_argument('--num-attention-heads', type=int, default=None)
    parser.add_argument('--num-hidden-layers', type=int, default=None)
    parser.add_argument('--plot-all', action='store_true')
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument(
        '--metrics',
        type=str,
        choices=('all', 'cossim'),
        default='all',
        help=(
            'all: original behavior, hidden-state cos-sim plus DSE outputs. '
            'cossim: cos-sim only. Add --include-logits-layer to append logits cos-sim.'
        ),
    )
    parser.add_argument(
        '--include-logits-layer',
        action='store_true',
        help=(
            'Append row-normalized logits cos-sim. This loads AutoModelForCausalLM first.'
        ),
    )
    parser.add_argument(
        '--save-layer-metrics',
        action='store_true',
        help='Also save inner products, HFC/LFC, singular-value entropy, MEV, and log-HFC metrics.',
    )
    parser.add_argument(
        '--device-map',
        type=str,
        default=None,
        help="Optional Hugging Face device_map, e.g. 'auto' for multi-GPU cluster runs.",
    )
    parser.add_argument(
        '--output-tag',
        type=str,
        default=None,
        help=(
            'Optional tag appended to output basenames (npz, png, csv) so runs do not '
            'overwrite previous files, e.g. Slurm job id. Unsafe characters are replaced.'
        ),
    )
    args = parser.parse_args()

    compute_dse = args.metrics == 'all'
    save_layer_metrics = args.save_layer_metrics

    if args.huggingface_token is not None:
        login(token=args.huggingface_token)

    config_kwargs = {'cache_dir': args.cache_dir}
    if args.num_attention_heads is not None:
        # NOTE: We cannot change number of attention heads for most models.
        # ALBERT is an exception.
        config_kwargs['num_attention_heads'] = args.num_attention_heads

    if args.num_hidden_layers is not None:
        # NOTE: We cannot change number of layeres for most models.
        # ALBERT model's attention blocks are identical,
        # so it can be further stacked without a problem.
        config_kwargs['num_hidden_layers'] = args.num_hidden_layers

    if args.tokenizer_id is None:
        args.tokenizer_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, cache_dir=args.cache_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_id, **config_kwargs, trust_remote_code=True)
    model_kwargs = {
        'cache_dir': args.cache_dir,
        'trust_remote_code': True,
    }
    if args.device_map is not None:
        model_kwargs['device_map'] = args.device_map

    if args.include_logits_layer:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                config=config,
                **model_kwargs,
            ).eval()
        except Exception as e:
            print(f"Unable to load as CausalLM: {args.model_id}. Falling back to AutoModel. Error: {e}.")
            model = AutoModel.from_pretrained(
                args.model_id,
                config=config,
                **model_kwargs,
            ).eval()
    else:
        try:
            model = AutoModel.from_pretrained(
                args.model_id,
                config=config,
                **model_kwargs,
            ).eval()
        except Exception as e:
            print(f"Unable to process model: {args.model_id}. Error occurred: {e}.")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                **model_kwargs,
            ).eval()

    # Extracting the cosine similarity by layer, and average over repetitions.
    cossim_matrix_by_layer = []
    inner_matrix_by_layer = []
    DSE_by_layer = []
    hfc_lfc_ratio_by_rep = None
    singular_value_entropy_by_rep = None
    mev_by_rep = None
    log_hfc_frobenius_by_rep = None
    embeddings_by_layer_last = None  # for optional plot_DSE when DSE is computed

    for random_seed in tqdm(range(args.repetitions)):
        torch.manual_seed(random_seed)

        # Run model on a random long input.
        text = get_random_long_text(args.dataset, random_seed=random_seed, min_word_count=1024, max_word_count=1280)
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            output = model(**tokens, output_hidden_states=True)
            embeddings_by_layer = organize_embeddings(output.hidden_states)
            curr_cossim_matrix_by_layer = compute_cosine_similarities(embeddings_by_layer)

            if args.include_logits_layer and hasattr(output, 'logits'):
                logits = torch.nn.functional.normalize(output.logits.squeeze(0).float(), dim=1)
                logits_cossim = torch.matmul(logits, logits.T).clamp(-1, 1).cpu().numpy()
                curr_cossim_matrix_by_layer.append(logits_cossim)

            if save_layer_metrics:
                curr_inner_by_layer = per_layer_inner_products(embeddings_by_layer)
                curr_hfc_lfc = per_layer_hfc_lfc_ratio(embeddings_by_layer)
                curr_sv_entropy, curr_mev = per_layer_singular_value_entropy_and_mev(embeddings_by_layer)
                curr_log_hfc = per_layer_log_hfc_frobenius(embeddings_by_layer)
                if args.include_logits_layer and hasattr(output, 'logits'):
                    logits = torch.nn.functional.normalize(output.logits.squeeze(0).float(), dim=1)
                    logits_np = logits.cpu().numpy()
                    x0 = embeddings_by_layer[0]
                    curr_inner_by_layer.append(pairwise_inner_products(logits_np))
                    curr_hfc_lfc = np.append(curr_hfc_lfc, hfc_lfc_ratio(logits_np))
                    ent_l, mev_l = singular_value_entropy_and_mev(logits_np)
                    curr_sv_entropy = np.append(curr_sv_entropy, ent_l)
                    curr_mev = np.append(curr_mev, mev_l)
                    curr_log_hfc = np.append(
                        curr_log_hfc, log_hfc_frobenius_relative(logits_np, x0)
                    )
            else:
                curr_inner_by_layer = None
                curr_hfc_lfc = None
                curr_sv_entropy = None
                curr_mev = None
                curr_log_hfc = None

            if compute_dse:
                curr_DSE_per_layer = np.array([
                    diffusion_spectral_entropy(embeddings, gaussian_kernel_sigma=10, t=10)
                    for embeddings in embeddings_by_layer])
                embeddings_by_layer_last = embeddings_by_layer
            else:
                curr_DSE_per_layer = None

        if random_seed == 0:
            cossim_matrix_by_layer = [
                curr_cossim_matrix_by_layer[i][None, ...].clip(-1, 1)
                for i in range(len(curr_cossim_matrix_by_layer))
            ]
            if save_layer_metrics:
                inner_matrix_by_layer = [
                    curr_inner_by_layer[i][None, ...] for i in range(len(curr_inner_by_layer))
                ]
                hfc_lfc_ratio_by_rep = curr_hfc_lfc[None, ...]
                singular_value_entropy_by_rep = curr_sv_entropy[None, ...]
                mev_by_rep = curr_mev[None, ...]
                log_hfc_frobenius_by_rep = curr_log_hfc[None, ...]
            if compute_dse:
                DSE_by_layer = curr_DSE_per_layer[None, ...]
        else:
            for i in range(len(cossim_matrix_by_layer)):
                cossim_matrix_by_layer[i] = np.concatenate(
                    (
                        cossim_matrix_by_layer[i],
                        curr_cossim_matrix_by_layer[i][None, ...],
                    ),
                    axis=0,
                )
            if save_layer_metrics:
                for i in range(len(inner_matrix_by_layer)):
                    inner_matrix_by_layer[i] = np.concatenate(
                        (inner_matrix_by_layer[i], curr_inner_by_layer[i][None, ...]), axis=0
                    )
                hfc_lfc_ratio_by_rep = np.concatenate(
                    (hfc_lfc_ratio_by_rep, curr_hfc_lfc[None, ...]), axis=0
                )
                singular_value_entropy_by_rep = np.concatenate(
                    (singular_value_entropy_by_rep, curr_sv_entropy[None, ...]), axis=0
                )
                mev_by_rep = np.concatenate((mev_by_rep, curr_mev[None, ...]), axis=0)
                log_hfc_frobenius_by_rep = np.concatenate(
                    (log_hfc_frobenius_by_rep, curr_log_hfc[None, ...]), axis=0
                )
            if compute_dse:
                DSE_by_layer = np.concatenate((DSE_by_layer, curr_DSE_per_layer[None, ...]), axis=0)

    for i in range(len(cossim_matrix_by_layer)):
        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)
    if save_layer_metrics:
        for i in range(len(inner_matrix_by_layer)):
            inner_matrix_by_layer[i] = inner_matrix_by_layer[i].mean(axis=0)

    # Plot and save histograms.
    model_name_cleaned = '-'.join(args.model_id.split('/'))
    out_tag = (args.output_tag or '').strip().replace('/', '_').replace(' ', '_')
    tag_suffix = f'_{out_tag}' if out_tag else ''
    plot_similarity_heatmap(
        cossim_matrix_by_layer,
        save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_cossim_heatmap_{model_name_cleaned}_{args.dataset}{tag_suffix}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')

    # Save results.
    cossim_matrix_by_layer = np.array(cossim_matrix_by_layer)
    npz_cossim = f'../visualization/transformer/{model_name_cleaned}/results_cossim_{args.dataset}{tag_suffix}.npz'
    np.savez(npz_cossim, cossim_matrix_by_layer=cossim_matrix_by_layer)

    if compute_dse:
        npz_DSE = f'../visualization/transformer/{model_name_cleaned}/results_DSE_{args.dataset}{tag_suffix}.npz'
        np.savez(npz_DSE, DSE_by_layer=DSE_by_layer)

        csv_DSE = f'../visualization/transformer/{model_name_cleaned}/results_DSE_{args.dataset}{tag_suffix}.csv'
        columns = [
            'model_name',
            'first_layer_mean', 'first_layer_std',
            'mean_mean', 'mean_std',
            'last_layer_mean', 'last_layer_std'
        ]
        new_data = {
            'model_name': model_name_cleaned,
            'first_layer_mean': DSE_by_layer.mean(axis=0)[0],
            'first_layer_std': DSE_by_layer.std(axis=0)[0],
            'mean_mean': DSE_by_layer.mean(axis=0).mean(),
            'mean_std': DSE_by_layer.mean(axis=0).std(),
            'last_layer_mean': DSE_by_layer.mean(axis=0)[-1],
            'last_layer_std': DSE_by_layer.std(axis=0)[-1],
        }
        if os.path.exists(csv_DSE):
            df = pd.read_csv(csv_DSE)
        else:
            df = pd.DataFrame(columns=columns)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        df.to_csv(csv_DSE, index=False)

    if save_layer_metrics:
        inner_product_matrix_by_layer = np.array(inner_matrix_by_layer)
        npz_metrics = f'../visualization/transformer/{model_name_cleaned}/results_layer_metrics_{args.dataset}{tag_suffix}.npz'
        np.savez(
            npz_metrics,
            inner_product_matrix_by_layer=inner_product_matrix_by_layer,
            hfc_lfc_ratio_by_rep=hfc_lfc_ratio_by_rep,
            singular_value_entropy_by_rep=singular_value_entropy_by_rep,
            mev_by_rep=mev_by_rep,
            log_hfc_frobenius_by_rep=log_hfc_frobenius_by_rep,
        )

    if args.plot_all:
        plot_similarity_histograms(
            cossim_matrix_by_layer,
            save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_cossim_histogram_{model_name_cleaned}_{args.dataset}{tag_suffix}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')
        plot_probability(
            cossim_matrix_by_layer,
            save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_cossim_probability_{model_name_cleaned}_{args.dataset}{tag_suffix}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')
        plot_entropy(
            cossim_matrix_by_layer,
            save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_cossim_entropy_{model_name_cleaned}_{args.dataset}{tag_suffix}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')
        if compute_dse and embeddings_by_layer_last is not None:
            plot_DSE(
                embeddings_by_layer_last,
                save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_DSE_{model_name_cleaned}_{args.dataset}{tag_suffix}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')
