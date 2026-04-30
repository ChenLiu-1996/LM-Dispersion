"""
Aggregate lm_eval JSONs from pretrain_toy_gpt2.py runs and write a LaTeX table.

Run folder names match pretrain_toy_gpt2.py ``args.output_dir``::

    ./results/pretrain_toy_ffn_{model}_nlayers-{L}_ninner-{F}_{dataset_dash}_lr-...
    disp-..._fewshot-..._maxsample-..._seed-{seed}

``dataset_dash`` is only ``"-".join(dataset_name.split("/"))``; ``--dataset_config`` is
not in the path.

One table row per run config (seeds merged): metrics at the best checkpoint step are
shown as mean \\pm std across seeds (stderr from lm_eval is not used in cells).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from glob import glob

import numpy as np

RUN_SEED_SUFFIX = re.compile(r"_seed-\d+$")
NLAYER_RE = re.compile(r"_nlayers-(\d+)_")
NINNER_RE = re.compile(r"_ninner-(\d+)_")


def glob_pretrain_ffn_run_dirs(results_dir: str, model_name: str, dataset_name: str | None) -> list[str]:
    """Glob aligned with pretrain_toy_gpt2.py; ``dataset_name=None`` skips dataset segment."""
    if dataset_name is not None:
        ds_dash = "-".join(dataset_name.split("/"))
        pattern = os.path.join(
            results_dir,
            f"pretrain_toy_ffn_{model_name}_nlayers-*_ninner-*_{ds_dash}_*",
        )
    else:
        pattern = os.path.join(results_dir, f"pretrain_toy_ffn_{model_name}_nlayers-*_ninner-*")
    return sorted(p for p in glob(pattern) if os.path.isdir(p))

empty_metrics_dict = {
    "step": [],
    "paloma_wikitext_103\nword_perplexity,none": {"mean": [], "std": []},
    "anli_r2\nacc,none": {"mean": [], "std": []},
    "lambada_openai\nacc,none": {"mean": [], "std": []},
    "openbookqa\nacc,none": {"mean": [], "std": []},
    "piqa\nacc,none": {"mean": [], "std": []},
    "truthfulqa_mc2\nacc,none": {"mean": [], "std": []},
    "winogrande\nacc,none": {"mean": [], "std": []},
    "arc_easy\nacc,none": {"mean": [], "std": []},
    "arc_challenge\nacc,none": {"mean": [], "std": []},
    "medmcqa\nacc,none": {"mean": [], "std": []},
    "mmlu\nacc,none": {"mean": [], "std": []},
}


def run_key_from_folder_basename(folder_basename):
    """Strip trailing _seed-<int> so all seeds of the same config group together."""
    return RUN_SEED_SUFFIX.sub("", folder_basename)


def parse_dispersion_from_pretrain_basename(basename):
    if "disp-" not in basename:
        raise ValueError(f"Missing disp- segment in results folder: {basename}")
    seg = basename.split("disp-", 1)[1]
    if "-tau_cos-" in seg:
        disp_part = seg.split("-tau_cos-", 1)[0]
    elif "_fewshot-" in seg:
        disp_part = seg.split("_fewshot-", 1)[0]
    else:
        disp_part = seg.split("_", 1)[0]
    parts = disp_part.split("-")
    if len(parts) < 3:
        raise ValueError(f"Expected disp-{{name}}-{{coeff}}-{{loc}} before tau_cos: {basename}")
    dispersion_name, dispersion_coefficient = parts[0], parts[1]
    dispersion_location = "-".join(parts[2:])
    return dispersion_name, dispersion_coefficient, dispersion_location


def parse_nlayers_from_basename(basename):
    m = NLAYER_RE.search(basename)
    if not m:
        raise ValueError(f"Missing nlayers- in folder name: {basename}")
    return int(m.group(1))


def parse_ninner_from_basename(basename):
    m = NINNER_RE.search(basename)
    if not m:
        raise ValueError(f"Missing ninner- in folder name: {basename}")
    return int(m.group(1))


def load_folder_metrics(run_folder, template_metrics_dict):
    metrics = deepcopy(template_metrics_dict)
    eval_json_list = glob(os.path.join(run_folder, "lm_eval_*.json"))
    for eval_json in sorted(eval_json_list):
        with open(eval_json, "r") as f:
            data_json = json.load(f)
        metrics["step"].append(int(eval_json.split("_")[-1].replace(".json", "")))
        for metric_key in template_metrics_dict.keys():
            if metric_key == "step":
                continue
            metric_dataset = metric_key.split("\n")[0]
            metric_measure = metric_key.split("\n")[1]
            metrics[metric_key]["mean"].append(
                float(data_json["results"][metric_dataset][metric_measure])
            )
            std_value = data_json["results"][metric_dataset][
                metric_measure.replace(",", "_stderr,")
            ]
            if std_value == "N/A":
                metrics[metric_key]["std"].append(np.nan)
            else:
                metrics[metric_key]["std"].append(float(std_value))
    return metrics


def aggregate_metrics_across_seeds(seed_metrics_list, template_metrics_dict):
    if not seed_metrics_list:
        return deepcopy(template_metrics_dict)
    all_steps = set()
    for sm in seed_metrics_list:
        all_steps.update(sm["step"])
    if not all_steps:
        return deepcopy(template_metrics_dict)
    all_steps_sorted = sorted(all_steps)
    out = deepcopy(template_metrics_dict)
    out["step"] = all_steps_sorted
    for metric_key in template_metrics_dict.keys():
        if metric_key == "step":
            continue
        means_out = []
        stds_out = []
        for step in all_steps_sorted:
            vals = []
            for sm in seed_metrics_list:
                if step not in sm["step"]:
                    continue
                idx = sm["step"].index(step)
                vals.append(sm[metric_key]["mean"][idx])
            if not vals:
                means_out.append(np.nan)
                stds_out.append(np.nan)
            else:
                means_out.append(float(np.mean(vals)))
                stds_out.append(0.0 if len(vals) < 2 else float(np.std(vals, ddof=1)))
        out[metric_key]["mean"] = means_out
        out[metric_key]["std"] = stds_out
    return out


def sort_series_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]


def compute_best_steps(metrics_one_run, selection_metric_names):
    steps_array = np.asarray(metrics_one_run["step"], dtype=int)
    if steps_array.size == 0:
        return None
    order = np.argsort(steps_array)
    steps_sorted = steps_array[order]
    per_metric_mean = {}
    for name in selection_metric_names:
        arr = np.asarray(metrics_one_run[name]["mean"], dtype=float)
        if arr.size == 0:
            continue
        per_metric_mean[name] = arr[order]
    if not per_metric_mean:
        return None
    n = steps_sorted.size
    scores = np.full(n, np.nan, dtype=float)
    for t in range(n):
        vals = []
        for name in per_metric_mean:
            v = float(per_metric_mean[name][t])
            if np.isfinite(v):
                vals.append(v)
        if vals:
            scores[t] = float(np.mean(vals))
    if np.all(~np.isfinite(scores)):
        return None
    return int(np.nanargmax(scores))


def sorted_series_at_steps(metrics_one_run, metric_key):
    steps = np.asarray(metrics_one_run["step"], dtype=int)
    means = np.asarray(metrics_one_run[metric_key]["mean"], dtype=float)
    stds = np.asarray(metrics_one_run[metric_key]["std"], dtype=float)
    if steps.size == 0:
        return steps, means, stds
    return sort_series_by_step(steps, means, stds)


def value_std_at_best_step(metrics_one_run, metric_key, best_idx):
    _, means, stds = sorted_series_at_steps(metrics_one_run, metric_key)
    if best_idx is None or means.size == 0 or best_idx >= means.size:
        return np.nan, np.nan
    m = float(means[best_idx])
    s = float(stds[best_idx]) if stds.size > best_idx else np.nan
    return m, s


def _average_scalar_at_step_single_seed(seed_metrics_dict, step, metric_name_list):
    vals = []
    if step not in seed_metrics_dict["step"]:
        return np.nan
    idx = seed_metrics_dict["step"].index(step)
    for metric_name in metric_name_list:
        if "perplexity" in metric_name:
            continue
        v = float(seed_metrics_dict[metric_name]["mean"][idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan
    return float(np.mean(vals))


def per_seed_avg_at_step_then_mean_std(seed_metrics_list, metric_names, step):
    seed_avgs = []
    for sd in seed_metrics_list:
        a = _average_scalar_at_step_single_seed(sd, step, metric_names)
        if np.isfinite(a):
            seed_avgs.append(a * 100.0)
    if not seed_avgs:
        return np.nan, np.nan
    mean_across = float(np.mean(seed_avgs))
    std_across = 0.0 if len(seed_avgs) < 2 else float(np.std(seed_avgs, ddof=1))
    return mean_across, std_across


def numeric_sort_key(coeff_str):
    try:
        return (0, float(coeff_str))
    except (TypeError, ValueError):
        return (1, str(coeff_str))


def render_mean_std_table(rows, metric_names_for_table, decimals=1, decimals_average=2, output_path=None):
    column_alignment = "r r l c c " + " ".join(["c"] * len(metric_names_for_table)) + " c"
    header_names = [name.replace("\n", " ").replace(",", " ") for name in metric_names_for_table]
    header_names.append("Average")

    lines = [
        r"\begin{tabular}{" + column_alignment + r"}",
        r"\toprule",
        "$L$ & $F$ & Disp. & Coeff & Loc & " + " & ".join(header_names) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        merged = row["merged_metrics"]
        best_idx = row["best_idx"]
        left = f"{row['nlayers']} & {row['n_inner']} & {row['disp_name']} & {row['coeff']} & {row['loc']}"
        cells = []
        steps_sorted, _, _ = sorted_series_at_steps(merged, metric_names_for_table[0])
        step_target = None
        if best_idx is not None and steps_sorted.size > best_idx:
            step_target = int(steps_sorted[best_idx])

        for metric_name in metric_names_for_table:
            m, s = value_std_at_best_step(merged, metric_name, best_idx)
            if not np.isfinite(m):
                cells.append("N/A")
            else:
                mp = m * 100.0
                sp = (s * 100.0) if np.isfinite(s) else 0.0
                cells.append(f"{mp:.{decimals}f} $\\pm$ {sp:.{decimals}f}")

        if step_target is not None:
            mean_a, std_a = per_seed_avg_at_step_then_mean_std(row["seed_metrics"], metric_names_for_table, step_target)
        else:
            mean_a, std_a = np.nan, np.nan

        if np.isfinite(mean_a) and np.isfinite(std_a):
            cells.append(f"{mean_a:.{decimals_average}f} $\\pm$ {std_a:.{decimals_average}f}")
        elif np.isfinite(mean_a):
            cells.append(f"{mean_a:.{decimals_average}f} $\\pm$ {0.0:.{decimals_average}f}")
        else:
            cells.append("N/A")

        lines.append(left + " & " + " & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_tex = "\n".join(lines)

    if output_path is None:
        os.makedirs("./tables", exist_ok=True)
        output_path = "./tables/pretrain_toy_ffn_results_mean_std.tex"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_tex)

    print("\n===== LaTeX table (mean $\\pm$ std across seeds) =====\n")
    print(table_tex)
    print(f"\n[Saved: {output_path}]\n")


def main(args):
    result_folder = os.path.abspath(args.results_dir)
    dataset_filter = None if args.any_dataset else args.dataset_name
    run_folder_list = glob_pretrain_ffn_run_dirs(result_folder, args.model_name, dataset_filter)
    run_folder_list = [rf for rf in run_folder_list if len(glob(os.path.join(rf, "lm_eval_*.json"))) > 0]
    pattern_note = (
        f"pretrain_toy_ffn_{args.model_name}_nlayers-*_ninner-* (any dataset)"
        if dataset_filter is None
        else os.path.join(
            result_folder,
            f"pretrain_toy_ffn_{args.model_name}_nlayers-*_ninner-*_{'-'.join(args.dataset_name.split('/'))}_*",
        )
    )

    grouped = defaultdict(list)
    for run_folder in run_folder_list:
        bn = os.path.basename(run_folder.rstrip(os.sep))
        grouped[run_key_from_folder_basename(bn)].append(run_folder)

    rows = []
    for run_key in sorted(grouped.keys()):
        folders = sorted(grouped[run_key])
        ref_basename = os.path.basename(folders[0].rstrip(os.sep))
        nlayers = parse_nlayers_from_basename(ref_basename)
        n_inner = parse_ninner_from_basename(ref_basename)
        disp_name, coeff, loc = parse_dispersion_from_pretrain_basename(ref_basename)

        seed_metrics = [load_folder_metrics(f, empty_metrics_dict) for f in folders]
        merged = aggregate_metrics_across_seeds(seed_metrics, empty_metrics_dict)

        all_metric_names = [k for k in merged.keys() if k != "step"]
        selection_metrics = [name for name in all_metric_names if "perplexity" not in name.lower()]
        best_idx = compute_best_steps(merged, selection_metrics)

        rows.append(
            {
                "nlayers": nlayers,
                "n_inner": n_inner,
                "disp_name": str(disp_name),
                "coeff": str(coeff),
                "loc": str(loc),
                "merged_metrics": merged,
                "seed_metrics": seed_metrics,
                "best_idx": best_idx,
            }
        )

    rows.sort(
        key=lambda r: (
            r["nlayers"],
            r["n_inner"],
            r["disp_name"].lower(),
            numeric_sort_key(r["coeff"]),
            r["loc"],
        )
    )

    if not rows:
        raise RuntimeError(f"No runs under {result_folder} matching {pattern_note} with lm_eval_*.json files.")

    metric_names_for_table = [
        k for k in empty_metrics_dict.keys() if k != "step" and "perplexity" not in k.lower()
    ]

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "tables", f"pretrain_toy_ffn_table_{args.model_name}.tex")

    render_mean_std_table(
        rows,
        metric_names_for_table,
        decimals=args.decimals,
        decimals_average=args.decimals_average,
        output_path=out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mean±std LaTeX table from pretrain_toy_gpt2 (FFN sweep) lm_eval outputs.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Salesforce/wikitext",
        help="Same as pretrain_toy_gpt2.py --dataset_name (slashes → hyphens only).",
    )
    parser.add_argument(
        "--any_dataset",
        action="store_true",
        help="Match pretrain_toy_ffn_{model}_nlayers-*_ninner-* regardless of dataset segment.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory with pretrain_toy_ffn_* run folders (default: ./results next to this script).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .tex path (default: ./tables/pretrain_toy_ffn_table_{model}.tex).",
    )
    parser.add_argument("--decimals", type=int, default=1, help="Decimals per task (percentage scale).")
    parser.add_argument("--decimals_average", type=int, default=2, help="Decimals for Average column.")
    args = parser.parse_args()
    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(__file__), "results")
    main(args)
