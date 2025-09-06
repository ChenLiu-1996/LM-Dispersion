from typing import List
import os
import sys
from glob import glob
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel
import tempfile

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, os.path.join(import_dir, 'prelim'))
from utils.text_data import get_random_long_text


def normalize(x, p=2, axis=1, eps=1e-3):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

def organize_embeddings(embeddings: List[torch.Tensor]) -> List[np.ndarray]:
    embeddings_by_layer = []
    for z in embeddings:
        z = z.squeeze(0).cpu().numpy()
        embeddings_by_layer.append(z)
    return embeddings_by_layer

def compute_cosine_similarities(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    cossim_matrix_by_layer = []
    for z in embeddings:
        z = normalize(z, axis=1)
        cossim_matrix = np.matmul(z, z.T).clip(-1, 1)
        cossim_matrix_by_layer.append(cossim_matrix)
    return cossim_matrix_by_layer

def build_hist_stack(cossim_matrix_by_layer: List[np.ndarray], step: int = 1, bins: int = 128):
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_indices, hist_data = [], []
    for (layer_idx, cossim_matrix) in selected:
        cossim_arr = cossim_matrix.flatten()
        hist, _ = np.histogram(cossim_arr, bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_indices.append(layer_idx)
    return np.array(hist_data), layer_indices

def parse_run_triplet(run_folder: str):
    dispersion = run_folder.split('disp-')[1].split('-')[0]
    dispersion_coeff = run_folder.split(f'{dispersion}-')[1].split('-')[0]
    dispersion_loc = run_folder.split(f'{dispersion_coeff}-')[1].split('_')[0]
    return dispersion, dispersion_coeff, dispersion_loc

def find_checkpoints(run_folder: str):
    ckpt_dirs = glob(os.path.join(run_folder, 'eval_ckpt_*_step*'))
    out = []
    for p in ckpt_dirs:
        name = os.path.basename(p)
        if 'step' in name:
            try:
                step = int(name.split('step')[-1])
                out.append((step, p))
            except:
                pass
    out.sort(key=lambda x: x[0])
    return out

def run_label(d, c, l):
    if str(d) == 'None':
        return 'None'
    return f'{d}-{c}-{l}'


if __name__ == '__main__':
    figure_save_prefix = './figures/embedding_heatmaps_grid'

    os.makedirs(os.path.dirname(figure_save_prefix), exist_ok=True)
    run_folder_list = sorted(glob(os.path.join('./results', 'midtrain_gpt2_*')))

    repetitions = 5
    max_length = 1024
    vmax = 10
    ckpt_stride = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    runs = []
    baseline_idx = None
    for run_folder in run_folder_list:
        d, c, l = parse_run_triplet(run_folder)
        # NOTE: Temporary hack.
        if float(c) > 1:
            continue
        if str(d).lower() == 'none' and baseline_idx is None:
            baseline_idx = len(runs)
        runs.append((run_folder, d, c, l))

    if baseline_idx is None:
        raise RuntimeError('No baseline run found with dispersion == None.')

    baseline_run = runs[baseline_idx]
    others = runs[:baseline_idx] + runs[baseline_idx+1:]
    order_disp = ['Covariance', 'Hinge', 'InfoNCE_l2', 'InfoNCE_cosine']

    def coeff_key(x):
        try:
            return float(x)
        except:
            return np.inf

    for disp in order_disp:
        group = [r for r in others if r[1] == disp]
        group.sort(key=lambda r: coeff_key(r[2]))
        if len(group) == 0:
            continue

        runs_in_fig = [baseline_run] + group

        ckpt_lists = []
        max_ckpts = 0
        for (run_folder, d, c, l) in runs_in_fig:
            ckpts = find_checkpoints(run_folder)
            ckpts = ckpts[::ckpt_stride]
            ckpt_lists.append(ckpts)
            if len(ckpts) > max_ckpts:
                max_ckpts = len(ckpts)

        fig = plt.figure(figsize=(4 * max_ckpts, 4 * len(runs_in_fig)))
        for row_idx, (run_folder, d, c, l) in enumerate(tqdm(runs_in_fig)):
            ckpts = ckpt_lists[row_idx]
            for col_idx in tqdm(range(max_ckpts)):
                ax = fig.add_subplot(len(runs_in_fig), max_ckpts, row_idx * max_ckpts + col_idx + 1)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                if col_idx >= len(ckpts):
                    ax.axis('off')
                    continue

                step, ckpt_path = ckpts[col_idx]

                with tempfile.TemporaryDirectory() as tmp_cache:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, cache_dir=tmp_cache)
                    except Exception:
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_fallback_id, cache_dir=tmp_cache)

                    try:
                        config = AutoConfig.from_pretrained(ckpt_path, cache_dir=tmp_cache)
                        model = AutoModel.from_pretrained(ckpt_path, config=config, cache_dir=tmp_cache)
                    except Exception:
                        ax.axis('off')
                        continue

                    model.to(device)
                    model.eval()

                    cossim_matrix_by_layer = None
                    for random_seed in range(repetitions):
                        torch.manual_seed(random_seed)
                        text = get_random_long_text('wikipedia',
                                                    random_seed=random_seed,
                                                    min_word_count=1024,
                                                    max_word_count=1280)
                        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
                        tokens = {k: v.to(device) for k, v in tokens.items()}

                        with torch.no_grad():
                            output = model(**tokens, output_hidden_states=True)
                            embeddings_by_layer = organize_embeddings(output.hidden_states)
                            curr = compute_cosine_similarities(embeddings_by_layer)

                        if cossim_matrix_by_layer is None:
                            cossim_matrix_by_layer = [m[None, ...] for m in curr]
                        else:
                            for i in range(len(cossim_matrix_by_layer)):
                                cossim_matrix_by_layer[i] = np.concatenate(
                                    (cossim_matrix_by_layer[i], curr[i][None, ...]), axis=0
                                )

                    for i in range(len(cossim_matrix_by_layer)):
                        cossim_matrix_by_layer[i] = cossim_matrix_by_layer[i].mean(axis=0)

                    hist_matrix, layer_indices = build_hist_stack(cossim_matrix_by_layer)
                    if hist_matrix.size == 0:
                        ax.axis('off')
                    else:
                        im = ax.imshow(
                            hist_matrix,
                            aspect='auto',
                            origin='lower',
                            cmap='Reds',
                            extent=[-1, 1, 0, layer_indices[-1]],
                            vmin=0,
                            vmax=vmax,
                        )

                    ax.set_title(f'{run_label(d, c, l)}\nstep {step}', fontsize=20)
                    ax.set_ylabel('Layer', fontsize=18)
                    ax.set_xlabel('Cosine Similarity', fontsize=18)

                    del model
                    if device == 'cuda':
                        torch.cuda.empty_cache()

                fig.tight_layout(pad=2)
                fig.savefig(f'{figure_save_prefix}_{disp}.png', dpi=300)

        plt.close(fig)

    print('\nDone.')
