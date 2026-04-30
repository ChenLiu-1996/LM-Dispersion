from typing import List
import argparse
import os
import sys
import numpy as np
import torch
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
import matplotlib.pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from diffusion.diffusion_condensation import diffusion_condensation
from utils.text_data import get_random_long_text


def compute_matrix_ranks(embeddings: List[np.ndarray]) -> List[int]:
    ranks = []
    for z in tqdm(embeddings):
        rank = np.linalg.matrix_rank(z).item()
        ranks.append(rank)
    return ranks

def plot_matrix_ranks(ranks: List[int], save_path: str = None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ranks, marker='o', linewidth=2, color='#2ca02c')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_title('Embedding Matrix Rank per Layer', fontsize=20)
    ax.set_xlabel('Layer', fontsize=18)
    ax.set_ylabel('Matrix Rank', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    parser.add_argument('--random-seed', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # Load dataset and model.
    tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    config = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model = AlbertModel.from_pretrained("albert-xlarge-v2", config=config)

    # Run model on a random long input.
    text = get_random_long_text('wikipedia')
    tokens = tokenizer(text, return_tensors='pt', truncation=True)

    # Compute the matrix rank of the embeddings.
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)

        # Effectively only take the first 2 layers.
        z_0 = output.hidden_states[0].squeeze(0)  # [seq_len, hidden_dim]
        z_0 = torch.nn.functional.normalize(z_0, dim=1, p=2).cpu().numpy()
        z_1 = output.hidden_states[1].squeeze(0)  # [seq_len, hidden_dim]
        z_1 = torch.nn.functional.normalize(z_1, dim=1, p=2).cpu().numpy()

        embeddings_by_layer = diffusion_condensation(X=z_1, random_seed=args.random_seed)
        embeddings_by_layer = [z_0] + embeddings_by_layer

        ranks = compute_matrix_ranks(embeddings_by_layer)

    plot_matrix_ranks(ranks, save_path='../../visualization/diffusion/embedding_matrix_rank_albert_xlarge_v2.png')
