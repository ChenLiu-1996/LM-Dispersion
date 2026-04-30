"""Cosine-similarity histogram heatmaps from pretrain_toy_gpt2 checkpoints (folder names = that script's output_dir)."""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from glob import glob

import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, os.path.join(import_dir, 'prelim'))
from utils.text_data import get_random_long_text

PLOT_TREND_WIDTH_UNIT = 10.0
PLOT_TREND_FIGHEIGHT = 8.0


def normalize_rows(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    row_norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    return x / np.maximum(row_norms, eps)


def cosine_matrices_per_layer(hidden_states: tuple) -> list[np.ndarray]:
    return [np.matmul(z, z.T).clip(-1, 1) for z in (normalize_rows(t.squeeze(0).cpu().numpy()) for t in hidden_states)]


def mean_cosine_per_layer(cossim_by_layer: list[np.ndarray]) -> np.ndarray:
    """Scalar mean cosine per layer (same summary as plot_trend.py on cossim matrices)."""
    return np.array([float(np.asarray(m).mean()) for m in cossim_by_layer], dtype=np.float64)


def layer_depth_correlations(cossim_by_layer: list[np.ndarray]) -> tuple[float, float]:
    """Spearman and Kendall correlation between layer-mean cosine and layer index (trend vs depth)."""
    y = mean_cosine_per_layer(cossim_by_layer)
    if y.size < 2:
        return float("nan"), float("nan")
    x = np.arange(len(y), dtype=np.float64)
    sp, _ = spearmanr(y, x)
    kt, _ = kendalltau(y, x)
    return (float(sp) if np.isfinite(sp) else float("nan"), float(kt) if np.isfinite(kt) else float("nan"))


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


def sort_key(path: str) -> tuple:
    basename = os.path.basename(path.rstrip(os.sep))
    _, seed = parse_ninner_and_seed(basename)
    checkpoints = find_checkpoints(path)
    max_step = checkpoints[-1][0] if checkpoints else -1
    is_not_ce_only = not dispersion_is_none_run(basename)
    return (is_not_ce_only, -seed, -max_step, path)

def pick_one_folder_per_ninner(run_paths: list[str]) -> dict[int, str]:
    paths_grouped: dict[int, list[str]] = {}
    for path in run_paths:
        basename = os.path.basename(path.rstrip(os.sep))
        n_inner, _ = parse_ninner_and_seed(basename)
        paths_grouped.setdefault(n_inner, []).append(path)
    return {f: sorted(paths, key=sort_key)[-1] for f, paths in paths_grouped.items()}


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
    axis.set_title(title, fontsize=36, pad=24)
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
    colorbar.ax.set_title("Probability\nDensity", fontsize=20, pad=20)


def draw_trend_panel(
    fig,
    grid_spec: gridspec.GridSpec,
    row: int,
    f_values: list[int],
    spearman_corr_list: list[float],
    kendall_tau_list: list[float],
) -> None:
    """Dual-axis trend over F for one row (same labels for each row; data from that row's checkpoints)."""
    n = len(f_values)
    if n < 1:
        return
    ax = fig.add_subplot(grid_spec[row, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x = np.arange(n, dtype=np.float64)
    colors_b = plt.cm.Blues(np.linspace(0.3, 0.9, n))
    for i in range(n - 1):
        seg_x, seg_y = x[i : i + 2], [spearman_corr_list[i], spearman_corr_list[i + 1]]
        if np.isfinite(seg_y).all():
            ax.plot(seg_x, seg_y, color=colors_b[i], linewidth=4)
    for i in range(n):
        if np.isfinite(spearman_corr_list[i]):
            ax.scatter([i], [spearman_corr_list[i]], color=[colors_b[i]], s=120, zorder=3)
    ax.set_ylabel("Spearman correlation", labelpad=12, fontsize=30, color=colors_b[n - 1])
    ax.set_ylim(-1, 1.05)
    ax.set_xlim(-0.3, n - 0.7)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(f_values, fontsize=24)
    ax.set_xlabel("MLP dimension", fontsize=30)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    colors_r = plt.cm.Reds(np.linspace(0.3, 0.9, n))
    for i in range(n - 1):
        seg_x, seg_y = x[i : i + 2], [kendall_tau_list[i], kendall_tau_list[i + 1]]
        if np.isfinite(seg_y).all():
            ax2.plot(seg_x, seg_y, color=colors_r[i], linewidth=4)
    for i in range(n):
        if np.isfinite(kendall_tau_list[i]):
            ax2.scatter([i], [kendall_tau_list[i]], color=[colors_r[i]], s=120, zorder=3)
    ax2.set_ylabel("Kendall tau", labelpad=36, fontsize=30, rotation=270, color=colors_r[n - 1])
    ax2.set_ylim(-1, 1.05)
    ax2.tick_params(axis="y", which="major", labelsize=26)


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

    n_panels = len(ninner_order)
    spearman_first: list[float] = []
    kendall_first: list[float] = []
    spearman_last: list[float] = []
    kendall_last: list[float] = []

    if n_panels > 0:
        # trend | gap | heatmaps (row 0 = first ckpt, row 1 = last ckpt per F)
        width_ratios = [1.0, 0.05] + [1.0] * n_panels
        ratio_sum = float(sum(width_ratios))
        figure = plt.figure(
            figsize=(
                PLOT_TREND_WIDTH_UNIT * ratio_sum,
                PLOT_TREND_FIGHEIGHT * 2.0,
            )
        )
        grid_spec = gridspec.GridSpec(2, len(width_ratios), width_ratios=width_ratios, height_ratios=[1.0, 1.0])
    else:
        figure = plt.figure(figsize=(PLOT_TREND_WIDTH_UNIT, PLOT_TREND_FIGHEIGHT * 2.0))
        grid_spec = None

    def run_heatmap_for_checkpoint(axis, checkpoint_path: str, n_inner: int) -> tuple[float, float]:
        """Returns (spearman, kendall) from mean cossim layers for the trend panel on the same row."""
        with tempfile.TemporaryDirectory() as cache_dir:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, cache_dir=cache_dir)
            model = load_model(checkpoint_path, cache_dir, device)
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
            sp, kt = layer_depth_correlations(mean_per_layer)
            draw_heatmap(figure, axis, mean_per_layer, title=f"MLP dim $= {n_inner}$")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            return sp, kt

    for col_index, n_inner in enumerate(tqdm(ninner_order, desc="n_inner")):
        assert grid_spec is not None
        run_folder = folder_by_ninner[n_inner]
        checkpoints = find_checkpoints(run_folder)
        if not checkpoints:
            for row in range(2):
                ax = figure.add_subplot(grid_spec[row, col_index + 2])
                ax.axis("off")
            spearman_first.append(float("nan"))
            kendall_first.append(float("nan"))
            spearman_last.append(float("nan"))
            kendall_last.append(float("nan"))
            continue

        first_path = checkpoints[0][1]
        last_path = checkpoints[-1][1]
        rows_spec = [(0, first_path), (1, last_path)]

        for row_idx, ckpt_path in rows_spec:
            axis = figure.add_subplot(grid_spec[row_idx, col_index + 2])
            try:
                sp, kt = run_heatmap_for_checkpoint(axis, ckpt_path, n_inner)
                if row_idx == 0:
                    spearman_first.append(sp)
                    kendall_first.append(kt)
                else:
                    spearman_last.append(sp)
                    kendall_last.append(kt)
            except Exception:
                axis.axis("off")
                if row_idx == 0:
                    spearman_first.append(float("nan"))
                    kendall_first.append(float("nan"))
                else:
                    spearman_last.append(float("nan"))
                    kendall_last.append(float("nan"))

            figure.tight_layout(pad=2)
            figure.savefig(args.output, dpi=300)

    if grid_spec is not None and n_panels > 0:
        draw_trend_panel(figure, grid_spec, 0, ninner_order, spearman_first, kendall_first)
        draw_trend_panel(figure, grid_spec, 1, ninner_order, spearman_last, kendall_last)

    figure.tight_layout(pad=2)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="First- and last-checkpoint heatmaps per n_inner + Spearman/Kendall trend (FFN sweep).")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--dataset_name", default="codelion/fineweb-edu-1B")
    parser.add_argument("--any_dataset", action="store_true")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = os.path.join(script_dir, "results")
    if args.output is None:
        args.output = os.path.join(script_dir, "figures", f"embedding_heatmaps_pretrain_toy_ffn_{args.model_name}_first_last_ckpt_per_F.png")
    main(args)
