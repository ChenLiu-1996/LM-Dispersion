from typing import List
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm
import tempfile

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from dse.dse import diffusion_spectral_entropy
from utils.text_data import get_random_long_text


def extract_embeddings(hidden_states: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in tqdm(hidden_states):
        z = z.squeeze(0).cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer

def compute_cosine_similarities(hidden_states: List[torch.Tensor]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in tqdm(hidden_states):
        z = torch.nn.functional.normalize(z.squeeze(0), dim=1)
        cossim_matrix = torch.matmul(z, z.T).cpu().numpy()
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def compute_matrix_ranks(hidden_states: List[torch.Tensor]) -> List[int]:
    ranks = []
    for z in tqdm(hidden_states):
        z = z.squeeze(0)  # [seq_len, hidden_dim]
        rank = torch.linalg.matrix_rank(z).item()
        ranks.append(rank)
    return ranks

def compute_DSE(embeddings_by_layer: List[np.ndarray]):
    '''
    Compute Diffusion Spectral Entropy (https://arxiv.org/abs/2312.04823).
    '''
    DSE_by_layer = [diffusion_spectral_entropy(normalize_numpy(embeddings), gaussian_kernel_sigma=1, t=1)
                    for embeddings in embeddings_by_layer]
    return DSE_by_layer

def compute_VNE(cossim_matrix_by_layer: List[np.ndarray]):
    '''
    Compute von Neumann Entropy.
    '''
    VNE_by_layer = [compute_entropy(cossim_matrix, entropy_type='von Neumann')
                    for cossim_matrix in cossim_matrix_by_layer]
    return VNE_by_layer


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

def normalize_numpy(x, p=2, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / (norm + eps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()

    # NOTE: The model performance is copied from
    # https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/
    model_zoo = pd.read_csv('model_performance.csv')

    output_csv = 'condensation_vs_performance.csv'
    output_df = pd.DataFrame(columns=model_zoo.keys().tolist() + \
        ['dataset', 'seed',
         'final_layer_matrix_rank', 'final_layer_DSE', 'final_layer_VNE',
         'matrix_rank_arr', 'DSE_arr', 'VNE_arr'])

    with torch.no_grad():

        for row_idx, row in model_zoo.iterrows():
            huggingface_id = row['huggingface_ID']

            with tempfile.TemporaryDirectory() as tmp_cache:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(huggingface_id, cache_dir=tmp_cache)
                    config = AutoConfig.from_pretrained(huggingface_id, cache_dir=tmp_cache)
                    model = AutoModel.from_pretrained(huggingface_id, config=config, cache_dir=tmp_cache)
                except Exception as e:
                    print(f"Unable to process model: {huggingface_id}. Error occurred: {e}.")
                    continue

                for repetition_idx in range(args.repeat):
                    torch.manual_seed(repetition_idx)

                    for dataset_name in ['wikipedia', 'pubmed', 'imdb', 'squad']:
                        text = get_random_long_text(dataset_name)
                        tokens = tokenizer(text, return_tensors='pt', truncation=True)
                        output = model(**tokens, output_hidden_states=True)

                        matrix_rank_by_layer = compute_matrix_ranks(output.hidden_states)

                        embeddings_by_layer = extract_embeddings(output.hidden_states)
                        DSE_by_layer = compute_DSE(embeddings_by_layer=embeddings_by_layer)

                        cossim_matrix_by_layer = compute_cosine_similarities(output.hidden_states)
                        VNE_by_layer = compute_VNE(cossim_matrix_by_layer=cossim_matrix_by_layer)

                        row['dataset'] = dataset_name
                        row['seed'] = repetition_idx
                        row['final_layer_matrix_rank'] = matrix_rank_by_layer[-1]
                        row['final_layer_DSE'] = DSE_by_layer[-1]
                        row['final_layer_VNE'] = VNE_by_layer[-1]
                        row['matrix_rank_arr'] = np.array(matrix_rank_by_layer)
                        row['DSE_arr'] = np.array(DSE_by_layer)
                        row['VNE_arr'] = np.array(VNE_by_layer)

                        output_df.loc[len(output_df)] = row
                        output_df.to_csv(output_csv, index=False)

