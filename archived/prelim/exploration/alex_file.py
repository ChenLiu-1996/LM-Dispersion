'''
File to compute Diffusion Spectral Entropy (DSE) and von Neumann Entropy (VNE) for w following models:
- Meta-Llama-3-8b
- Google/gemma-7b

NOTE: This code was adapted from the code found in the file evalute_transformer_metrics.py (Coded by Chen Liu)
'''


from typing import List
import argparse
import os
import sys
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer,
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AlbertConfig, AlbertTokenizer, AlbertModel
)
import matplotlib.pyplot as plt
from tqdm import tqdm

from huggingface_hub import login
login("login_token_here")  # Replace with your Hugging Face token

# Imports from your local utilities
from transformer_embedding_dse import organize_embeddings, plot_DSE
from transformer_embedding_histogram import (
    compute_entropy, plot_entropy,
    plot_probability, plot_similarity_histograms,
    compute_cosine_similarities
)

from dse.dse import diffusion_spectral_entropy

# Utility
def get_random_long_text_input(dataset_name, tokenizer, min_length: int = 300, max_length: int = 512):
    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        key = 'text'
    elif dataset_name == 'pubmed_qa':
        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        key = 'long_answer'
    elif dataset_name == 'imdb':
        dataset = load_dataset("imdb")
        key = 'text'
    elif dataset_name == 'squad':
        dataset = load_dataset("squad")
        key = 'context'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    while True:
        idx = torch.randint(len(dataset['train']), (1,)).item()
        text = dataset['train'][idx][key]
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        if tokens['input_ids'].shape[1] > min_length:
            return tokens

# Model inference wrapper
def load_model_correctly(model_name: str):
    # Handle causal vs encoder-decoder
    if "llama" in model_name or "gemma" in model_name:
        model_class = AutoModel

    config = AutoConfig.from_pretrained(model_name, use_auth_token=True)
    model = model_class.from_pretrained(model_name, config=config, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    return tokenizer, config, model

def alex_plot_DSE(embeddings_by_layer: List[np.ndarray], cossim_matrix_by_layer: List[np.ndarray],  save_path: str = None):
    '''
    Function to generate the plots that we want for DSE w/ the VNE, per Smita's request

    This code is adapted from the original code found in the file evalute_transformer_metrics.py (Coded by Chen Liu)
    '''
    fig = plt.figure(figsize=(16, 8))
    for subplot_idx, plot in enumerate(['DSE', 'VNE']):
        if plot == 'DSE':
            normalize_func = lambda x: x
            ax = fig.add_subplot(1, 2, 1)
            for sigma, color_base in zip([1, 5, 10], ['Blues', 'Reds', 'Greens']):
                cmap = plt.get_cmap(color_base)
                for diffusion_t, cmap_idx in zip([5, 20, 35, 50], [0.4, 0.6, 0.8, 1.0]): # NOTE: The first list in this line controls the "t" for the diffusion
                    entropy_arr = [diffusion_spectral_entropy(normalize_func(embeddings), gaussian_kernel_sigma=sigma, t=diffusion_t)
                                for embeddings in embeddings_by_layer]
                    ax.plot(entropy_arr, marker='o', linewidth=2, color=cmap(cmap_idx), label=f'$\sigma$ = {sigma}, t = {diffusion_t}')
            ax.legend(loc='upper right', ncols=3)

            ax.set_ylim([0, ax.get_ylim()[1] * 1.2])
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_title('DSE by layer Without L2 normalization', fontsize=18)
            ax.set_xlabel('Layer', fontsize=18)
            ax.set_ylabel('Entropy', fontsize=18)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            for subplot_idx, entropy_type in enumerate(['von Neumann']):
                ax = fig.add_subplot(1, 2, 2)
            
                entropy_arr = [compute_entropy(cossim_matrix, entropy_type=entropy_type)
                            for cossim_matrix in cossim_matrix_by_layer]
                ax.plot(entropy_arr, marker='o', linewidth=2, color='#2ca02c')

                ax.set_ylim([0, ax.get_ylim()[1]])
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.set_title(f'{entropy_type} Entropy by layer', fontsize=18)
                ax.set_xlabel('Layer', fontsize=14)
                ax.set_ylabel('Entropy', fontsize=14)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

    fig.suptitle('Embedding DSE per Layer, and Von Neuman Entropy', fontsize=24)
    fig.tight_layout(pad=2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--random-seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_list = [
        # "meta-llama/Meta-Llama-3-8B"
        "google/gemma-7b"
        # 'albert-xlarge-v2'
    ]
    dataset_list = ['wikitext', 'imdb', 'squad']

    for model_name in model_list:
        print(f"\n>>> Processing model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name) # , num_hidden_layers=48, num_attention_heads=1 for the albert model
            model = AutoModel.from_pretrained(model_name, config=config)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        for dataset_name in dataset_list:
            print(f"  > Dataset: {dataset_name}")
            try:
                tokens = get_random_long_text_input(dataset_name, tokenizer)
                # tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.no_grad():
                    output = model(**tokens, output_hidden_states=True)
                    hidden_states = output.hidden_states

                # 1. DSE
                cossim_matrix_by_layer = compute_cosine_similarities(hidden_states)
                embeddings_by_layer = organize_embeddings(hidden_states)
                model_tag = model_name.replace("/", "_")
                dse_path = f'../../visualization/transformer/embedding_DSE_{model_tag}_on_{dataset_name}.png'
                alex_plot_DSE(embeddings_by_layer, cossim_matrix_by_layer, save_path=dse_path)
                # plot_DSE(embeddings_by_layer, save_path=dse_path)

                # 2. Cosine Similarity
                

                # prob_path = f'../../visualization/transformer/embedding_cossim_probability_{model_tag}_on_{dataset_name}.png'
                # plot_probability(cossim_matrix_by_layer, save_path=prob_path)

                # entropy_path = f'../../visualization/transformer/embedding_cossim_entropy_{model_tag}_on_{dataset_name}.png'
                # plot_entropy(cossim_matrix_by_layer, save_path=entropy_path)

                # hist_path = f'../../visualization/transformer/embedding_cossim_histogram_{model_tag}_on_{dataset_name}.png'
                # plot_similarity_histograms(cossim_matrix_by_layer, save_path=hist_path)

            except Exception as e:
                print(f"Failed during evaluation on dataset {dataset_name} with model {model_name}: {e}")
                continue