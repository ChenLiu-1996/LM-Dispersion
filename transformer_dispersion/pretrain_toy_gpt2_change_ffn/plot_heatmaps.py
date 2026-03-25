"""Cosine-similarity histogram heatmaps from pretrain_toy_gpt2 checkpoints (folder names = that script's output_dir)."""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from glob import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, os.path.join(import_dir, 'prelim'))
from utils.text_data import get_random_long_text


def normalize_rows(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    row_norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return x / np.maximum(row_norms, eps)


def cosine_matrices_per_layer(hidden_states: tuple) -> list[np.ndarray]:
    return [np.matmul(z, z.T).clip(-1, 1) for z in (normalize_rows(t.squeeze(0).cpu().numpy()) for t in hidden_states)]


def histogram_stack_over_layers(cossim_by_layer: list[np.ndarray], bins: int = 128) -> tuple[np.ndarray, list[float]]:
    depth = max(1, len(cossim_by_layer) - 1)
    histograms, layer_fractions = [], []
    for layer_index, cossim_matrix in enumerate(cossim_by_layer):
        counts, _ = np.histogram(cossim_matrix.ravel(), bins=bins, density=True, range=(-1, 1))
        histograms.append(counts)
        layer_fractions.append(layer_index / depth)
    return np.array(histograms), layer_fractions


def dispersion_is_none_run(basename: str) -> bool:
    if "disp-" not in basename:
        return False
    segment = basename.split("disp-", 1)[1]
    if "-tau_cos-" in segment:
        dispersion_segment = segment.split("-tau_cos-", 1)[0]
    elif "_fewshot-" in segment:
        dispersion_segment = segment.split("_fewshot-", 1)[0]
    else:
        dispersion_segment = segment.split("_", 1)[0]
    return dispersion_segment.split("-")[0].lower() == "none"


def parse_ninner_and_seed(basename: str) -> tuple[int, int]:
    ninner_match = re.search(r"_ninner-(\d+)_", basename)
    if not ninner_match:
        raise ValueError(f"Missing ninner- in folder name: {basename}")
    n_inner = int(ninner_match.group(1))
    seed_match = re.search(r"_seed-(\d+)$", basename)
    seed = int(seed_match.group(1)) if seed_match else -1
    return n_inner, seed


def find_checkpoints(run_dir: str) -> list[tuple[int, str]]:
    checkpoint_paths = glob(os.path.join(run_dir, "eval_ckpt_*_step*"))
    rows = []
    for path in checkpoint_paths:
        try:
            step = int(os.path.basename(path).split("step")[-1])
            rows.append((step, path))
        except ValueError:
            pass
    return sorted(rows, key=lambda item: item[0])


def pick_one_folder_per_ninner(run_paths: list[str]) -> dict[int, str]:
    paths_grouped: dict[int, list[str]] = {}
    for path in run_paths:
        basename = os.path.basename(path.rstrip(os.sep))
        n_inner, _ = parse_ninner_and_seed(basename)
        paths_grouped.setdefault(n_inner, []).append(path)

    def sort_key(path: str) -> tuple:
        basename = os.path.basename(path.rstrip(os.sep))
        _, seed = parse_ninner_and_seed(basename)
        checkpoints = find_checkpoints(path)
        max_step = checkpoints[-1][0] if checkpoints else -1
        is_not_ce_only = not dispersion_is_none_run(basename)
        return (is_not_ce_only, -seed, -max_step, path)

    return {f: sorted(paths, key=sort_key)[0] for f, paths in paths_grouped.items()}


def load_model(checkpoint_path: str, cache_dir: str, device: str):
    config = AutoConfig.from_pretrained(checkpoint_path, cache_dir=cache_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config, cache_dir=cache_dir)
    except Exception:
        model = AutoModel.from_pretrained(checkpoint_path, config=config, cache_dir=cache_dir)
    return model.to(device).eval()


def draw_heatmap(fig, axis, cossim_by_layer: list[np.ndarray], title: str) -> None:
    hist_matrix, layer_fractions = histogram_stack_over_layers(cossim_by_layer)
    if hist_matrix.size == 0:
        axis.axis("off")
        return

    image = axis.imshow(hist_matrix.T, aspect="auto", origin="lower", cmap="Reds", extent=[0, layer_fractions[-1], -1, 1], vmin=0, vmax=10)
    axis.set_title(title, pad=8, fontsize=28)
    axis.set_xlabel("Layer Fraction", fontsize=36)
    axis.set_ylabel("Cosine Similarity", fontsize=36)
    axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axis.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    axis.set_ylim(-0.25, 1)
    axis.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1])
    axis.set_yticklabels([-0.25, 0, 0.25, 0.5, 0.75, 1])
    axis.tick_params(axis="both", which="major", labelsize=26)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    colorbar = fig.colorbar(image, ax=axis)
    colorbar.ax.tick_params(axis="both", which="major", labelsize=26)
    colorbar.ax.set_title("Probability\nDensity", fontsize=20, pad=12)


def glob_pretrain_ffn_run_directories(results_dir: str, model_name: str, dataset_name: str | None) -> tuple[list[str], str]:
    glob_tail = (
        f"pretrain_toy_ffn_{model_name}_nlayers-*_ninner-*_{'-'.join(dataset_name.split('/'))}_*"
        if dataset_name is not None
        else f"pretrain_toy_ffn_{model_name}_nlayers-*_ninner-*"
    )
    glob_pattern = os.path.join(results_dir, glob_tail)
    directories = sorted(path for path in glob(glob_pattern) if os.path.isdir(path))
    return directories, glob_pattern


def main(args) -> None:
    plt.rcParams["font.family"] = "sans-serif"

    dataset_filter = None if args.any_dataset else args.dataset_name
    run_directories, glob_pattern = glob_pretrain_ffn_run_directories(args.results_dir, args.model_name, dataset_filter)
    run_directories = [path for path in run_directories if find_checkpoints(path)]
    if not run_directories:
        raise RuntimeError(f"No eval checkpoints under {args.results_dir} matching {glob_pattern}")

    folder_by_ninner = pick_one_folder_per_ninner(run_directories)
    ninner_order = sorted(folder_by_ninner)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    output_directory = os.path.dirname(os.path.abspath(args.output))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    figure = plt.figure(figsize=(args.fig_width, args.fig_height_per_row * len(ninner_order)))
    for row_index, n_inner in enumerate(tqdm(ninner_order, desc="n_inner")):
        axis = figure.add_subplot(len(ninner_order), 1, row_index + 1)
        run_folder = folder_by_ninner[n_inner]
        checkpoints = find_checkpoints(run_folder)
        training_step, checkpoint_path = checkpoints[-1]

        with tempfile.TemporaryDirectory() as cache_dir:
            try:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, cache_dir=cache_dir)
                model = load_model(checkpoint_path, cache_dir, device)
            except Exception:
                axis.axis("off")
                continue

            stacked_per_layer = None
            for repetition_index in range(args.repetitions):
                torch.manual_seed(repetition_index)
                text = get_random_long_text("wikipedia", random_seed=repetition_index, min_word_count=1024, max_word_count=1280)
                batch = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.no_grad():
                    hidden_states = model(**batch, output_hidden_states=True).hidden_states
                    if hidden_states is None:
                        raise RuntimeError("Model returned no hidden_states")
                cossim_this_rep = cosine_matrices_per_layer(hidden_states)

                if stacked_per_layer is None:
                    stacked_per_layer = [arr[np.newaxis, ...] for arr in cossim_this_rep]
                else:
                    for layer_index, arr in enumerate(cossim_this_rep):
                        stacked_per_layer[layer_index] = np.concatenate((stacked_per_layer[layer_index], arr[np.newaxis, ...]), axis=0)

            mean_per_layer = [stack.mean(axis=0) for stack in stacked_per_layer]
            draw_heatmap(figure, axis, mean_per_layer, title=f"$F={n_inner}$, step {training_step}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    figure.tight_layout(pad=2)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Last-checkpoint heatmap per n_inner (pretrain_toy_gpt2 FFN sweep).")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_name", default="Salesforce/wikitext")
    parser.add_argument("--any_dataset", action="store_true")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--fig_width", type=float, default=9.5)
    parser.add_argument("--fig_height_per_row", type=float, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = os.path.join(script_dir, "results")
    if args.output is None:
        args.output = os.path.join(script_dir, "figures", f"embedding_heatmaps_pretrain_toy_ffn_{args.model_name}_last_ckpt_per_F.png")
    main(args)
