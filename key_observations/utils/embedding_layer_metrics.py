"""Layer-wise metrics on token embedding matrices X in R^{n x d} (rows = tokens)."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def mean_cossim_across_last_n_layers(mean_cossim_by_layer: np.ndarray) -> np.ndarray:
    """For N=1..L, mean of per-layer mean cossim over the last N hidden layers."""
    layer_means = np.asarray(mean_cossim_by_layer, dtype=np.float64).ravel()
    num_layers = len(layer_means)
    if num_layers == 0:
        return np.array([])
    out = np.empty(num_layers, dtype=np.float64)
    for n in range(1, num_layers + 1):
        out[n - 1] = float(np.mean(layer_means[-n:]))
    return out


def pairwise_inner_products(x: np.ndarray) -> np.ndarray:
    return x @ x.T


def maximum_explainable_variance(x: np.ndarray, eps: float = 1e-12) -> float:
    """MEV = sigma_1^2 / sum_i sigma_i^2 = sigma_1^2 / ||X||_F^2."""
    x = np.asarray(x)
    frob_sq = float(np.sum(x * x))
    if frob_sq < eps:
        return float("nan")
    s1 = float(np.linalg.norm(x, ord=2))
    return (s1 * s1) / frob_sq


def singular_value_entropy_and_mev(
    x: np.ndarray,
    log: Callable[[np.ndarray], np.ndarray] = np.log2,
) -> Tuple[float, float]:
    """One SVD for Shannon entropy of singular-value proportions and MEV."""
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    total_sq = float(np.dot(s, s))
    if total_sq < np.finfo(float).tiny:
        return 0.0, float("nan")
    mev = float((s[0] * s[0]) / total_sq) if s.size else float("nan")
    s_pos = s[s > 0]
    if s_pos.size == 0:
        return 0.0, mev
    p = s_pos / s_pos.sum()
    entropy = float(-np.sum(p * log(p)))
    return entropy, mev


def singular_value_entropy(
    x: np.ndarray,
    log: Callable[[np.ndarray], np.ndarray] = np.log2,
) -> float:
    """p_i = sigma_i / sum_j sigma_j over positive singular values; -sum p_i log(p_i)."""
    ent, _ = singular_value_entropy_and_mev(x, log=log)
    return ent


def hfc_lfc(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LFC = (1/n) 11^T X; HFC = (I - (1/n) 11^T) X along tokens."""
    n = x.shape[0]
    lfc = (np.ones((n, n), dtype=x.dtype) / n) @ x
    hfc = x - lfc
    return hfc, lfc


def hfc_lfc_ratio(x: np.ndarray, eps: float = 1e-12) -> float:
    """||HFC||_2 / ||LFC||_2 with matrix 2-norm."""
    hfc, lfc = hfc_lfc(x)
    num = np.linalg.norm(hfc, ord=2)
    den = np.linalg.norm(lfc, ord=2)
    if den < eps:
        return float("nan") if num >= eps else 0.0
    return float(num / den)


def log_hfc_frobenius_relative(
    x: np.ndarray,
    x0: np.ndarray,
    log: Callable[[float], float] = np.log,
    eps: float = 1e-12,
) -> float:
    """log(||HFC(X)||_F / ||HFC(X_0)||_F)."""
    hfc, _ = hfc_lfc(x)
    hfc0, _ = hfc_lfc(x0)
    n0 = np.linalg.norm(hfc0, ord="fro")
    n = np.linalg.norm(hfc, ord="fro")
    if n0 < eps:
        return float("nan")
    return float(log(n / n0))


def per_layer_inner_products(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    return [pairwise_inner_products(z) for z in embeddings]


def per_layer_hfc_lfc_ratio(embeddings: List[np.ndarray]) -> np.ndarray:
    return np.array([hfc_lfc_ratio(z) for z in embeddings], dtype=np.float64)


def per_layer_singular_value_entropy_and_mev(
    embeddings: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if not embeddings:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    ent, mev = zip(*(singular_value_entropy_and_mev(z) for z in embeddings))
    return np.array(ent, dtype=np.float64), np.array(mev, dtype=np.float64)


def per_layer_singular_value_entropy(embeddings: List[np.ndarray]) -> np.ndarray:
    ent, _ = per_layer_singular_value_entropy_and_mev(embeddings)
    return ent


def per_layer_log_hfc_frobenius(embeddings: List[np.ndarray]) -> np.ndarray:
    """Uses embeddings[0] as X_0."""
    if not embeddings:
        return np.array([], dtype=np.float64)
    x0 = embeddings[0]
    return np.array([log_hfc_frobenius_relative(z, x0) for z in embeddings], dtype=np.float64)
