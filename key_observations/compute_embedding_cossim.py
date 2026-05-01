from typing import List
import argparse
import os
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from huggingface_hub import login
import matplotlib.pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, import_dir)
from utils.text_data import get_random_long_text


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

def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--tokenizer-id', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default='/home/cl2482/palmer_scratch/.cache/')
    parser.add_argument('--model-id', type=str, default='albert-base-v2')
    parser.add_argument('--dataset', type=str, default='wikipedia')
    parser.add_argument('--huggingface-token', type=str, default=None)
    parser.add_argument('--num-attention-heads', type=int, default=None)
    parser.add_argument('--num-hidden-layers', type=int, default=None)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--include-logits-layer', action='store_true')
    args = parser.parse_args()

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

        if random_seed == 0:
            cossim_matrix_by_layer = [
                curr_cossim_matrix_by_layer[i][None, ...].clip(-1, 1)
                for i in range(len(curr_cossim_matrix_by_layer))
            ]

        else:
            for i in range(len(cossim_matrix_by_layer)):
                cossim_matrix_by_layer[i] = np.concatenate(
                    (
                        cossim_matrix_by_layer[i],
                        curr_cossim_matrix_by_layer[i][None, ...],
                    ),
                    axis=0,
                )

    for i in range(len(cossim_matrix_by_layer)):
        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)

    # Plot and save histograms.
    model_name_cleaned = '-'.join(args.model_id.split('/'))
    plot_similarity_heatmap(
        cossim_matrix_by_layer,
        save_path=f'../visualization/transformer/{model_name_cleaned}/embedding_cossim_heatmap_{model_name_cleaned}_{args.dataset}_layers_{config.num_hidden_layers}_heads_{config.num_attention_heads}.png')

    # Save results.
    cossim_matrix_by_layer = np.array(cossim_matrix_by_layer)
    npz_cossim = f'../visualization/transformer/{model_name_cleaned}/results_cossim_{args.dataset}.npz'
    np.savez(npz_cossim, cossim_matrix_by_layer=cossim_matrix_by_layer)
