"""Embedding condensation measurement for Hugging Face transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from tqdm import tqdm

try:
    word_tokenize('Arbitrary sentence.')
except:
    import nltk
    nltk.download('punkt_tab')


__all__ = [
    "CondensationResult",
    "measure_embedding_condensation",
    "plot_similarity_heatmap",
]


@dataclass
class CondensationResult:
    """Per-layer token cosine-similarity matrices (L, S, S)."""
    cossim_by_layer: np.ndarray


def get_random_long_text(
    dataset_name: str,
    min_word_count: int = 1024,
    max_word_count: int = 1280,
    split: str = "train",
    random_seed: int = 0,
) -> str:
    if dataset_name == "wikipedia":
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        key = "text"
    elif dataset_name == "pubmed":
        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        key = "long_answer"
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb")
        key = "text"
    elif dataset_name == "squad":
        dataset = load_dataset("squad")
        key = "context"
    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            "Use wikipedia, pubmed, imdb, squad, or pass `texts=` directly."
        )

    text = ""
    rng = np.random.default_rng(seed=random_seed)
    idx = rng.integers(0, int(len(dataset["train"]) * 0.95)).item()
    while len(word_tokenize(text)) < min_word_count:
        text += dataset[split][idx][key]
        idx += 1
        if len(word_tokenize(text)) > max_word_count:
            break
    return text


def _normalize(x: np.ndarray, p: int = 2, axis: int = 1, eps: float = 1e-3) -> np.ndarray:
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def organize_embeddings(embeddings: Sequence[torch.Tensor]) -> List[np.ndarray]:
    return [z.squeeze(0).float().cpu().numpy() for z in embeddings]


def compute_cosine_similarities(embeddings: Sequence[np.ndarray]) -> List[np.ndarray]:
    out = []
    for z in embeddings:
        z = _normalize(z, axis=1)
        out.append(np.matmul(z, z.T).clip(-1, 1))
    return out


def plot_similarity_heatmap(
    cossim_matrix_by_layer: Sequence[np.ndarray],
    save_path: Optional[str] = None,
    step: int = 1,
    bins: int = 128,
):
    n_layers = len(cossim_matrix_by_layer)
    denom = max(n_layers - 1, 1)
    selected = [(i, data) for i, data in enumerate(cossim_matrix_by_layer) if i % step == 0]
    layer_fractions, hist_data = [], []
    for layer_idx, cossim_matrix in selected:
        hist, _ = np.histogram(cossim_matrix.flatten(), bins=bins, density=True, range=(-1, 1))
        hist_data.append(hist)
        layer_fractions.append(layer_idx / denom)
    hist_matrix = np.array(hist_data)

    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    im = ax.imshow(
        hist_matrix.T,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        extent=[0, layer_fractions[-1], -1, 1],
        vmin=0,
        vmax=10,
    )
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel("Layer Fraction", fontsize=36)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel("Cosine Similarity", fontsize=36)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(axis="both", which="major", labelsize=26)
    cbar.ax.set_title("Probability\nDensity", fontsize=20, pad=20)
    fig.tight_layout(pad=2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return fig


def measure_embedding_condensation(
    model: torch.nn.Module,
    tokenizer,
    *,
    texts: Optional[Sequence[str]] = None,
    dataset: str = "wikipedia",
    repetitions: int = 100,
    max_length: int = 512,
    min_word_count: int = 1024,
    max_word_count: int = 1280,
    include_logits_layer: bool = False,
    plot: bool = True,
    save_path: Optional[str] = "./test_embedding_condensation.png",
    show_progress: bool = True,
) -> CondensationResult:
    """
    Run the LM-Dispersion embedding condensation measurement pipeline.

    Pass `texts` to use fixed inputs. Otherwise samples
    random long text from a Hugging Face dataset each repetition.
    """
    model.eval()
    device = next(model.parameters()).device
    stacked: Optional[List[np.ndarray]] = None

    if texts is not None:
        if len(texts) == 0:
            raise ValueError("`texts` must be non-empty when provided.")
        rep_iter = range(repetitions)
        if show_progress:
            rep_iter = tqdm(rep_iter, desc="condensation")
        for r in rep_iter:
            torch.manual_seed(r)
            text = texts[r % len(texts)]
            curr = _forward_cossim(
                model, tokenizer, text, device, max_length, include_logits_layer
            )
            stacked = _stack_repetition(stacked, curr)
    else:
        rep_iter = range(repetitions)
        if show_progress:
            rep_iter = tqdm(rep_iter, desc="condensation")
        for random_seed in rep_iter:
            torch.manual_seed(random_seed)
            text = get_random_long_text(
                dataset,
                random_seed=random_seed,
                min_word_count=min_word_count,
                max_word_count=max_word_count,
            )
            curr = _forward_cossim(
                model, tokenizer, text, device, max_length, include_logits_layer
            )
            stacked = _stack_repetition(stacked, curr)

    assert stacked is not None
    averaged = [m.mean(axis=0) for m in stacked]
    cossim_arr = np.stack(averaged, axis=0)
    result = CondensationResult(
        cossim_by_layer=cossim_arr,
    )

    if plot or save_path:
        plot_similarity_heatmap(averaged, save_path=save_path)
    return result


def _forward_cossim(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int,
    include_logits_layer: bool,
) -> List[np.ndarray]:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
        curr = compute_cosine_similarities(organize_embeddings(output.hidden_states))
        if include_logits_layer and hasattr(output, "logits"):
            logits = torch.nn.functional.normalize(output.logits.squeeze(0).float(), dim=1)
            curr.append(torch.matmul(logits, logits.T).clamp(-1, 1).cpu().numpy())
    return curr


def _stack_repetition(
    stacked: Optional[List[np.ndarray]],
    curr: List[np.ndarray],
) -> List[np.ndarray]:
    clipped = [m.clip(-1, 1) for m in curr]
    if stacked is None:
        return [m[None, ...] for m in clipped]
    return [
        np.concatenate((stacked[i], clipped[i][None, ...]), axis=0)
        for i in range(len(stacked))
    ]
