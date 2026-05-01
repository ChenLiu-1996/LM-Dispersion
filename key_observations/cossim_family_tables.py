#!/usr/bin/env python3
"""
Wide tables per model family: columns = models (small → large), rows = last-N layer means,
Spearman and Kendall vs layer index.

NPZ: results_cossim_{dataset}.npz with array ``cossim_matrix_by_layer`` of shape
(L, S, S) — L = number of hidden-state layers, S = sequence length (e.g. 512);
each [ℓ, :, :] is the token×token cosine-similarity matrix for layer ℓ, averaged
over forward-pass repetitions. Row metrics use mean over (1,2) → one scalar per layer.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr

import_dir = '/'.join(os.path.realpath(__file__).split("/")[:-2])
sys.path.insert(0, import_dir)
from utils.embedding_layer_metrics import mean_cossim_across_last_n_layers


FAMILIES: Dict[str, List[str]] = {
    "gpt2": [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
    ],
    "qwen25": [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
    ],
    "bloom": [
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
    ],
    "qwen3": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ],
}

# (row_id, concise label) — last-half uses n = max(1, L // 2) per model
ROW_SPECS: List[Tuple[str, str]] = [
    ("n1", "last 1 mean"),
    ("n2", "last 2 mean"),
    ("n3", "last 3 mean"),
    ("n4", "last 4 mean"),
    ("n5", "last 5 mean"),
    ("nhalf", "last half mean"),
    ("spearman", "spearman"),
    ("kendall", "kendall tau"),
]


def column_header(model_id: str) -> str:
    return model_id.split("/")[-1] if "/" in model_id else model_id


def load_metrics(
    model_id: str, dataset: str, tag_suffix: str
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Returns (mean_cossim_by_layer, spearman, kendall) or None if missing/corrupt."""
    cleaned = "-".join(model_id.split("/"))
    path = (
        f"../visualization/transformer/{cleaned}/"
        f"results_cossim_{dataset}{tag_suffix}.npz"
    )
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as data:
            cossim = data["cossim_matrix_by_layer"]
    except Exception as e:
        print(f"  warn: corrupt or unreadable npz for {model_id}: {e}")
        return None
    layer_mean = cossim.mean(axis=(1, 2))
    L = len(layer_mean)
    if L == 0:
        return None
    sp, _ = spearmanr(layer_mean, np.arange(L))
    kd, _ = kendalltau(layer_mean, np.arange(L))
    return layer_mean, float(sp), float(kd)


def cell_for_row(
    row_id: str,
    layer_mean: np.ndarray,
    mean_last_n: np.ndarray,
) -> float:
    L = len(layer_mean)
    if row_id == "nhalf":
        n = max(1, L // 2)
        return float(mean_last_n[n - 1])
    if row_id.startswith("n") and row_id[1:].isdigit():
        want = int(row_id[1:])
        n = min(want, L)
        return float(mean_last_n[n - 1])
    raise ValueError(row_id)


def fmt_num(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def format_family_markdown(family: str, cols: List[str], rows_out: List[List[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|")
    lines = [f"### {family}\n", ""]
    lines.append("| " + " | ".join(esc(x) for x in ["metric"] + cols) + " |\n")
    lines.append("| " + " | ".join(["---"] * (len(cols) + 1)) + " |\n")
    for row in rows_out:
        lines.append("| " + " | ".join(esc(x) for x in row) + " |\n")
    return "".join(lines)


def write_family_outputs(
    family: str,
    model_ids: Sequence[str],
    dataset: str,
    tag_suffix: str,
    out_dir: str,
    omit_model_ids: Set[str],
    decimals: int,
    print_markdown: bool,
) -> Optional[Tuple[str, str]]:
    cols = [column_header(m) for m in model_ids]
    col_loaded: Dict[str, Optional[Tuple[np.ndarray, float, float]]] = {}
    for mid in model_ids:
        h = column_header(mid)
        if mid in omit_model_ids:
            col_loaded[h] = None
        else:
            col_loaded[h] = load_metrics(mid, dataset, tag_suffix)

    missing = [mid for mid in model_ids if col_loaded[column_header(mid)] is None]
    if all(col_loaded[column_header(m)] is None for m in model_ids):
        print(f"[{family}] skip: no usable npz for any model")
        return None

    if missing:
        print(f"[{family}] '-' or missing for: {', '.join(missing)}")

    def cell_str(mid: str, row_id: str) -> str:
        h = column_header(mid)
        if mid in omit_model_ids or col_loaded[h] is None:
            return "-"
        layer_mean, sp, kd = col_loaded[h]
        if row_id == "spearman":
            return fmt_num(sp, decimals)
        if row_id == "kendall":
            return fmt_num(kd, decimals)
        mean_last_n = mean_cossim_across_last_n_layers(layer_mean)
        v = cell_for_row(row_id, layer_mean, mean_last_n)
        return fmt_num(v, decimals)

    rows_out: List[List[str]] = []
    for row_id, row_label in ROW_SPECS:
        rows_out.append([row_label] + [cell_str(mid, row_id) for mid in model_ids])

    os.makedirs(out_dir, exist_ok=True)
    base = f"table_{family}{tag_suffix}" if tag_suffix else f"table_{family}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    md_path = os.path.join(out_dir, f"{base}.md")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + cols)
        for row in rows_out:
            w.writerow(row)

    md_text = format_family_markdown(family, cols, rows_out)
    with open(md_path, "w") as f:
        f.write(md_text)
        if not md_text.endswith("\n"):
            f.write("\n")

    # print(f"Wrote {csv_path}")
    # print(f"Wrote {md_path}")
    if print_markdown:
        print()
        print(md_text.rstrip())
        print()
    return csv_path, md_path


def main() -> None:
    p = argparse.ArgumentParser(description="Per-family wide CSV + Markdown from results_cossim npz")
    p.add_argument(
        "--families",
        nargs="+",
        choices=list(FAMILIES.keys()) + ["all"],
        default=["all"],
    )
    p.add_argument("--dataset", type=str, default="wikipedia")
    p.add_argument("--output-tag", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="../visualization/transformer/_trend")
    p.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places in CSV and Markdown (default 4)",
    )
    p.add_argument(
        "--omit-model",
        action="append",
        default=[],
        metavar="MODEL_ID",
        help="HF model id to force as '-' in tables (repeatable).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print markdown tables to stdout (files are still written).",
    )
    args = p.parse_args()

    tag = (args.output_tag or "").strip().replace("/", "_").replace(" ", "_")
    tag_suffix = f"_{tag}" if tag else ""

    omit: Set[str] = set(args.omit_model)

    fams = list(FAMILIES.keys()) if "all" in args.families else args.families
    for fam in fams:
        write_family_outputs(
            fam,
            FAMILIES[fam],
            args.dataset,
            tag_suffix,
            args.out_dir,
            omit,
            args.decimals,
            print_markdown=not args.quiet,
        )


if __name__ == "__main__":
    main()
