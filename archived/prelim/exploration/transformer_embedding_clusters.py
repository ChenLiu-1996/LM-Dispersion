from typing import List
import argparse
import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import phate
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)
from utils.text_data import get_random_long_text


def auto_kmeans(z_2d: np.ndarray, max_k: int = 10, random_state: int = 42):
    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init='auto', max_iter=100, random_state=random_state)
        labels = kmeans.fit_predict(z_2d)

        if len(np.unique(labels)) == 1:
            continue  # skip trivial clustering

        score = silhouette_score(z_2d, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    return best_k, best_labels


def compute_cosine_similarities(embeddings: List[torch.Tensor], eps: float = 1e-6) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in tqdm(embeddings):
        z = torch.nn.functional.normalize(z.squeeze(0), dim=1, p=2)
        cossim_matrix = torch.matmul(z, z.T).cpu().numpy()

        # Clipping to avoid some numerical error.
        assert cossim_matrix.min() > -1 - eps and cossim_matrix.max() < 1 + eps
        cossim_matrix = np.clip(cossim_matrix, -1, 1)

        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer


def compute_embedding_clusters(cossim_matrix_by_layer: List[np.ndarray],
                               step: int = 10,
                            #    max_k: int = 10,
                               method: str = 'phate',
                               random_seed: int = 42) -> List[dict]:
    cossim_matrix_selected_layers = [(i, cossim_matrix) for i, cossim_matrix in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_cluster_data = []

    for layer_idx, cossim_matrix in tqdm(cossim_matrix_selected_layers):
        angular_distance = np.arccos(cossim_matrix)
        if method == 'phate':
            phate_op = phate.PHATE(n_components=2, knn_dist='precomputed_distance', random_state=random_seed)
            z_2d = phate_op.fit_transform(angular_distance)
        elif method == 'tsne':
            raise NotImplementedError
            # tsne_op = TSNE(n_components=2, init='random', metric='precomputed', random_state=random_seed)
            # z_2d = tsne_op.fit_transform(angular_distance)

        if layer_idx == 0:
            # best_k, labels = auto_kmeans(z_2d, max_k=max_k, random_state=random_seed)
            # assert best_k == len(np.unique(labels))
            kmeans = KMeans(n_clusters=10, n_init='auto', max_iter=100, random_state=random_seed)
            labels = kmeans.fit_predict(z_2d)

        layer_cluster_data.append({'layer': layer_idx, 'points': z_2d, 'labels': labels})

    return layer_cluster_data

# def compute_embedding_clusters(embeddings: List[torch.Tensor],
#                                step: int = 10,
#                                max_k: int = 100,
#                                method: str = 'phate',
#                                random_seed: int = 42) -> List[dict]:
#     embeddings_selected_layers = [(i, cossim_matrix) for i, cossim_matrix in enumerate(embeddings) if i % step == 0]
#     layer_cluster_data = []

#     for layer_idx, embeddings in tqdm(embeddings_selected_layers):
#         z = torch.nn.functional.normalize(embeddings.squeeze(0), dim=1).cpu().numpy()  # [seq_len, dim]
#         if method == 'phate':
#             phate_op = phate.PHATE(n_components=2, knn_dist='cosine', mds_dist='cosine', random_state=random_seed, t=10)
#             z_2d = phate_op.fit_transform(z)
#         elif method == 'tsne':
#             tsne_op = TSNE(n_components=2, metric='cosine', random_state=random_seed)
#             z_2d = tsne_op.fit_transform(z)

#         best_k, labels = auto_kmeans(z_2d, max_k=max_k, random_state=random_seed)
#         assert best_k == len(np.unique(labels))

#         layer_cluster_data.append({'layer': layer_idx, 'points': z_2d, 'labels': labels, 'n_clusters': best_k})

#     return layer_cluster_data

def plot_embedding_cluster(cluster_data: List[dict], save_path: str = None, method: str = 'phate'):
    num_plots = len(cluster_data)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for ax, data in zip(axes, cluster_data):
        points, labels = data['points'], data['labels']
        for cluster_id in np.unique(labels):
            cluster_points = points[labels == cluster_id]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'C{cluster_id}', alpha=0.7)
        ax.set_title(f"Layer {data['layer']}", fontsize=18)
        # ax.axis('off')

    # for ax in axes[num_plots:]:
    #     ax.axis('off')

    if method == 'phate':
        method_name = 'PHATE'
    elif method == 'tsne':
        method_name = 't-SNE'

    fig.suptitle(f'{method_name} Embedding Clusters per Layer', fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=2)

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
    parser.add_argument('--method', type=str, default='phate')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    # Load dataset and model.
    tokenizer = AlbertTokenizer.from_pretrained('albert-xlarge-v2')
    config = AlbertConfig.from_pretrained("albert-xlarge-v2", num_hidden_layers=48, num_attention_heads=1)
    model = AlbertModel.from_pretrained("albert-xlarge-v2", config=config)

    # Run model on a random long input.
    text = get_random_long_text('wikipedia')
    tokens = tokenizer(text, return_tensors='pt', truncation=True)

    # Dimensionality-reduce and plot the embeddings in 2D.
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
        cossim_matrix_by_layer = compute_cosine_similarities(output.hidden_states)
        cluster_data = compute_embedding_clusters(
            cossim_matrix_by_layer=cossim_matrix_by_layer,
            step=2,
            method=args.method,
            random_seed=args.random_seed)

        # cluster_data = compute_embedding_clusters(
        #     embeddings=output.hidden_states,
        #     step=2,
        #     method=args.method,
        #     random_seed=args.random_seed)

    plot_embedding_cluster(
        cluster_data,
        save_path=f'../../visualization/transformer/embedding_clusters_{args.method}_albert_xlarge_v2.png',
        method=args.method)
