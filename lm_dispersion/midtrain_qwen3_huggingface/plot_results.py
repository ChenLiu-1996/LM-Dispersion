import argparse
import os
import re
import numpy as np
from glob import glob
import json
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy

RUN_SEED_SUFFIX = re.compile(r"_seed-\d+$")

results_dict = {
    'dispersion': [],
    'dispersion_coeff': [],
    'dispersion_loc': [],
    'metrics': [],
    'per_seed_metrics': [],
}

empty_metrics_dict = {
    'step': [],
    'paloma_wikitext_103\nword_perplexity,none': {'mean': [], 'std': []},
    # 'anli_r1\nacc,none': {'mean': [], 'std': []},
    'anli_r2\nacc,none': {'mean': [], 'std': []},
    # 'anli_r3\nacc,none': {'mean': [], 'std': []},
    # 'hellaswag\nacc_norm,none': {'mean': [], 'std': []},
    'lambada_openai\nacc,none': {'mean': [], 'std': []},
    # 'lambada_standard\nacc,none': {'mean': [], 'std': []},
    'openbookqa\nacc,none': {'mean': [], 'std': []},
    'piqa\nacc,none': {'mean': [], 'std': []},
    # 'squad_completion\ncontains,none': {'mean': [], 'std': []},
    'truthfulqa_mc2\nacc,none': {'mean': [], 'std': []},
    'winogrande\nacc,none': {'mean': [], 'std': []},
    'arc_easy\nacc,none': {'mean': [], 'std': []},
    'arc_challenge\nacc,none': {'mean': [], 'std': []},
    # 'drop\nf1,none': {'mean': [], 'std': []},
    # 'gsm8k\nexact_match,flexible-extract': {'mean': [], 'std': []},
    # 'mathqa\nacc,none': {'mean': [], 'std': []},
    'medmcqa\nacc,none': {'mean': [], 'std': []},
    'mmlu\nacc,none': {'mean': [], 'std': []},
    # 'mmlu_pro\nexact_match,custom-extract': {'mean': [], 'std': []},
}

def sort_series_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]

def run_key_from_folder_basename(folder_basename):
    """Strip trailing _seed-<int> so all seeds of the same config group together."""
    return RUN_SEED_SUFFIX.sub("", folder_basename)

def is_midtrain_results_folder_basename(basename):
    """True if folder matches midtrain_gpt2 (dispersion) or midtrain_gpt2_other_counter_condensation naming."""
    return (
        "disp-" in basename
        or "_ccnoise-" in basename
        or "_ccforget-" in basename
    )

def parse_run_folder_basename(basename):
    """
    Parse result folder basename into (method_name, coeff_str, loc_str) aligned with format_run_label.

    - midtrain_gpt2.py: ..._disp-{name}-{coeff}-{loc}-tau_cos-...
    - midtrain_gpt2_other_counter_condensation.py: ..._ccnoise-{neftune_alpha}_fewshot-... or ..._ccforget-{K}_fewshot-...
    """
    if "_ccnoise-" in basename:
        coeff = basename.split("_ccnoise-")[1].split("_")[0]
        return "ccnoise", coeff, "na"
    if "_ccforget-" in basename:
        coeff = basename.split("_ccforget-")[1].split("_")[0]
        return "ccforget", coeff, "na"
    if "disp-" not in basename:
        raise ValueError(f"Unrecognized midtrain results folder: {basename}")
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

def coeff_scaled_for_colormap(method_name, coeff_str):
    """Map a run's numeric hyperparameter to [0,1] for colormaps."""
    _ = method_name
    try:
        v = float(coeff_str)
    except (TypeError, ValueError):
        v = 1.0
    s = (np.log10(max(v, 1e-10)) + 4.0) / 7.0
    return float(np.clip(s, 0.0, 1.0))

def load_folder_metrics(run_folder, template_metrics_dict):
    """Load one run directory into the same nested dict structure as results_dict['metrics'][i]."""
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
            metrics[metric_key]["mean"].append(float(data_json["results"][metric_dataset][metric_measure]))
            std_value = data_json["results"][metric_dataset][metric_measure.replace(",", "_stderr,")]
            if std_value == "N/A":
                metrics[metric_key]["std"].append(np.nan)
            else:
                metrics[metric_key]["std"].append(float(std_value))
    return metrics

def aggregate_metrics_across_seeds(seed_metrics_list, template_metrics_dict):
    """Mean and across-seed std (0 if a single seed) on the union of checkpoint steps."""
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

def format_run_label(dispersion_name, coefficient_value, location_name):
    if str(dispersion_name) == 'None':
        return 'None'
    return f'{dispersion_name}-{coefficient_value}-{location_name}'

def extract_coefficient_from_label(label_text):
    if label_text == 'Default loss':
        return 0
    elif label_text == 'No mid-training':
        return 'N/A'
    else:
        return float(label_text.split('-')[1].split('-')[0])

def numeric_coefficient_value(value_text):
    try:
        return float(value_text)
    except (TypeError, ValueError):
        return np.inf

def build_sorted_cache(results_storage):
    """Aligned merged (across-seed mean/std) series per run, keyed like before."""
    sorted_cache = {}
    for run_index in range(len(results_storage["metrics"])):
        steps_array = np.asarray(results_storage["metrics"][run_index]["step"], dtype=int)
        if steps_array.size == 0:
            sorted_cache[(run_index, "__steps__")] = np.array([], dtype=int)
            for metric_key in results_storage["metrics"][run_index].keys():
                if metric_key == "step":
                    continue
                sorted_cache[(run_index, metric_key)] = (
                    np.array([], dtype=int),
                    np.array([], dtype=float),
                    np.array([], dtype=float),
                )
            continue

        order = np.argsort(steps_array)
        steps_sorted = steps_array[order]
        sorted_cache[(run_index, "__steps__")] = steps_sorted
        for metric_key in results_storage["metrics"][run_index].keys():
            if metric_key == "step":
                continue
            means_array = np.asarray(results_storage["metrics"][run_index][metric_key]["mean"], dtype=float)
            stds_array = np.asarray(results_storage["metrics"][run_index][metric_key]["std"], dtype=float)
            if means_array.size == 0:
                sorted_cache[(run_index, metric_key)] = (steps_sorted, means_array, stds_array)
            else:
                sorted_cache[(run_index, metric_key)] = sort_series_by_step(steps_array, means_array, stds_array)
    return sorted_cache

def _raw_mean_std_at_merged_index(sorted_cache, run_index, metric_name, step_index, use_initial):
    """Aligned mean/std (fraction scale, not %) at merged-series index for one metric."""
    _, means_sorted, stds_sorted = sorted_cache[(run_index, metric_name)]
    if means_sorted.size == 0:
        return np.nan, np.nan
    index_to_use = 0 if use_initial else step_index
    if index_to_use is None or index_to_use >= means_sorted.size:
        return np.nan, np.nan
    m = float(means_sorted[index_to_use])
    if not np.isfinite(m):
        return np.nan, np.nan
    if stds_sorted.size <= index_to_use:
        return m, np.nan
    s = float(stds_sorted[index_to_use])
    return m, s

def value_at_index_percentage(sorted_cache, run_index, metric_name, step_index, use_initial=False):
    m, _ = _raw_mean_std_at_merged_index(sorted_cache, run_index, metric_name, step_index, use_initial)
    return np.nan if not np.isfinite(m) else (m * 100.0)


def value_at_index_std_percentage(sorted_cache, run_index, metric_name, step_index, use_initial=False):
    """Across-seed std at the given merged-series index, in percentage points (x100)."""
    _, s = _raw_mean_std_at_merged_index(sorted_cache, run_index, metric_name, step_index, use_initial)
    if not np.isfinite(s):
        return 0.0
    return float(s * 100.0)


def _mean_std_sample(values):
    """Return (mean, sample std) in the same units as values; std is 0 if len < 2."""
    if not values:
        return np.nan, 0.0
    return float(np.mean(values)), (0.0 if len(values) < 2 else float(np.std(values, ddof=1)))


def _ylim_with_padding(ymin, ymax, default=(0.0, 1.0)):
    """Expand [ymin, ymax] with epsilon for degenerate range, then 5% padding."""
    if not np.isfinite(ymin):
        ymin = default[0]
    if not np.isfinite(ymax):
        ymax = default[1]
    if ymax == ymin:
        eps = 1e-6 if ymin == 0 else 0.01 * abs(ymin)
        ymin, ymax = ymin - eps, ymax + eps
    pad = 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad


def compute_metric_ylim_by_per_seed_best(
    results_storage,
    all_metric_names,
    baseline_run_index,
    best_steps_per_run,
    sorted_cache,
):
    """Y-limits from merged first checkpoint, then per-run mean and std across seeds at per-seed-best steps."""
    metric_ylim_ranges = {metric_name: [np.inf, -np.inf] for metric_name in all_metric_names}
    for metric_name in all_metric_names:
        candidate_values = []

        _, means_baseline, stds_baseline = sorted_cache[(baseline_run_index, metric_name)]
        if means_baseline.size > 0:
            candidate_values.append(float(means_baseline[0]))
            if stds_baseline.size > 0 and np.isfinite(stds_baseline[0]):
                candidate_values.append(float(means_baseline[0] + stds_baseline[0]))
                candidate_values.append(float(means_baseline[0] - stds_baseline[0]))

        for run_index in range(len(results_storage["metrics"])):
            m, s = mean_std_metric_at_per_seed_best(
                results_storage["per_seed_metrics"][run_index],
                metric_name,
                best_steps_per_run[run_index],
                as_percent=False,
            )
            if np.isfinite(m):
                candidate_values.append(m)
                if np.isfinite(s) and s > 0:
                    candidate_values.append(m + s)
                    candidate_values.append(m - s)

        if candidate_values:
            metric_ylim_ranges[metric_name][0] = float(np.nanmin(candidate_values))
            metric_ylim_ranges[metric_name][1] = float(np.nanmax(candidate_values))

    for metric_name, (ymin, ymax) in metric_ylim_ranges.items():
        metric_ylim_ranges[metric_name] = _ylim_with_padding(ymin, ymax)

    return metric_ylim_ranges


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


def best_training_step_per_seed(per_seed_metrics_for_run, selection_metric_names):
    """
    Per seed: average selection metrics at each checkpoint, then choose the training step
    that maximizes that scalar (first max on ties). Returns a list parallel to per_seed_metrics_for_run.
    """
    best_steps = []
    for seed_dict in per_seed_metrics_for_run:
        steps = seed_dict.get("step") or []
        if not steps:
            best_steps.append(None)
            continue
        best_step, best_score = None, -np.inf
        for step in steps:
            v = _average_scalar_at_step_single_seed(seed_dict, step, selection_metric_names)
            if np.isfinite(v) and v > best_score:
                best_score = v
                best_step = step
        best_steps.append(best_step)
    return best_steps


def mean_std_metric_at_per_seed_best(per_seed_list, metric_name, best_steps_per_seed, *, as_percent):
    """
    Per-seed best step: collect one value of metric_name per seed; return mean and sample std.
    as_percent True: [0,100] for LaTeX; False: raw [0,1] scores for plot bars.
    """
    scale = 100.0 if as_percent else 1.0
    vals = []
    for sd, bs in zip(per_seed_list, best_steps_per_seed):
        if bs is None or bs not in sd["step"] or metric_name not in sd:
            continue
        idx = sd["step"].index(bs)
        v = float(sd[metric_name]["mean"][idx])
        if np.isfinite(v):
            vals.append(v * scale)
    return _mean_std_sample(vals)


def composite_mean_std_at_per_seed_best(per_seed_list, selection_metric_names, best_steps_per_seed, *, as_percent):
    """Per-seed composite (mean over selection_metric_names) at that seed's best step; mean/std across seeds."""
    scale = 100.0 if as_percent else 1.0
    vals = []
    for sd, bs in zip(per_seed_list, best_steps_per_seed):
        if bs is None:
            continue
        a = _average_scalar_at_step_single_seed(sd, bs, selection_metric_names)
        if np.isfinite(a):
            vals.append(a * scale)
    return _mean_std_sample(vals)


def average_curve_with_seed_spread(per_seed_metrics, metric_name_list):
    """
    Per step: scalar = mean of task metrics (non-perplexity); then mean/std across seeds.
    Returns (steps, mean_curve, std_curve) with std 0 when only one seed has that step.
    """
    if not per_seed_metrics:
        return np.array([]), np.array([]), np.array([])
    all_steps = set()
    for sd in per_seed_metrics:
        all_steps.update(sd["step"])
    if not all_steps:
        return np.array([]), np.array([]), np.array([])
    all_steps_sorted = sorted(all_steps)
    mean_curve = []
    std_curve = []
    for step in all_steps_sorted:
        seed_avgs = []
        for sd in per_seed_metrics:
            v = _average_scalar_at_step_single_seed(sd, step, metric_name_list)
            if np.isfinite(v):
                seed_avgs.append(v)
        if not seed_avgs:
            mean_curve.append(np.nan)
            std_curve.append(np.nan)
        else:
            m, s = _mean_std_sample(seed_avgs)
            mean_curve.append(m)
            std_curve.append(s)
    return np.array(all_steps_sorted), np.array(mean_curve), np.array(std_curve)

def average_scalar_at_step_from_seed_curves(steps_arr, mean_arr, std_arr, step_target):
    """Look up mean and across-seed std on the 'average' curve at an exact training step."""
    steps_arr = np.asarray(steps_arr, dtype=int)
    if steps_arr.size == 0:
        return np.nan, 0.0
    match = np.where(steps_arr == int(step_target))[0]
    if match.size == 0:
        return np.nan, 0.0
    i = int(match[0])
    m = float(mean_arr[i])
    if std_arr is None or std_arr.size <= i:
        return m, 0.0
    s = float(std_arr[i])
    return m, (0.0 if not np.isfinite(s) else s)

def extend_ylim_candidates_from_band(candidate_list, mean_arr, std_arr):
    """Append mean, mean+std, mean-std (finite only) to a list used for axis y-limits."""
    m = np.asarray(mean_arr, dtype=float)
    s = np.asarray(std_arr, dtype=float)
    finite = np.isfinite(m)
    if not np.any(finite):
        return
    candidate_list.extend(m[finite].ravel().tolist())
    candidate_list.extend((m + s)[finite].ravel().tolist())
    candidate_list.extend((m - s)[finite].ravel().tolist())

def _latex_cell_mean_pm_std(value, std, decimals):
    """First table row: mean with LaTeX \\pm and std."""
    if not np.isfinite(value):
        return "N/A"
    s = float(std) if np.isfinite(std) else 0.0
    return f"{value:.{decimals}f} $\\pm$ {s:.{decimals}f}"


def _latex_cell_delta_colored(value, baseline_value, decimals):
    """Second table row: improvement vs baseline only (green/red), no subscript or parentheses."""
    if not np.isfinite(value) or not np.isfinite(baseline_value):
        return "---"
    difference = np.round(value, decimals) - np.round(baseline_value, decimals)
    sign = "+" if difference >= 0 else ""
    color_name = "forestgreen" if difference >= 0 else "crimson"
    return f"\\small \\textcolor{{{color_name}}}{{{sign}{difference:.{decimals}f}}}"

def per_seed_avg_metrics_then_mean_std_across_seeds(
    results_storage,
    sorted_cache,
    run_index,
    metric_names_for_table,
    step_index_merged,
    use_initial,
):
    """
    For each seed: average (in %, same scale as the table) over metric_names at the checkpoint
    step indexed by step_index_merged in the merged run, or the first checkpoint if use_initial.

    Returns (mean across seeds of those per-seed averages, std across seeds, ddof=1 if n>1 else 0).
    """
    per_seed = results_storage["per_seed_metrics"][run_index]
    steps_merged = sorted_cache[(run_index, '__steps__')]
    if steps_merged.size == 0 or not per_seed:
        return np.nan, 0.0
    idx = 0 if use_initial else step_index_merged
    if idx is None or idx >= steps_merged.size:
        return np.nan, 0.0
    step_target = int(steps_merged[idx])
    seed_avgs = []
    for seed_dict in per_seed:
        a = _average_scalar_at_step_single_seed(seed_dict, step_target, metric_names_for_table)
        if np.isfinite(a):
            seed_avgs.append(a * 100.0)
    return _mean_std_sample(seed_avgs)

def render_latex_table(
    results_storage,
    metric_names_for_table,
    baseline_run_index,
    rows_by_dispersion,
    dispersion_order,
    model_name,
    lora_suffix,
    decimals=1,
    decimals_average=2,
    output_path=None,
    best_steps_per_run=None,
    sorted_cache=None,
):
    rows = []
    rows.append({"method": "Pretrained (no mid-training)", "disp": "None", "coeff": "N/A", "loc": "-", "src": ("baseline","initial")})
    rows.append({"method": "Default loss", "disp": "None", "coeff": "0.0", "loc": "-", "src": ("baseline","best")})
    for dispersion_name in dispersion_order:
        indices = sorted(rows_by_dispersion[dispersion_name], key=lambda i: numeric_coefficient_value(results_storage['dispersion_coeff'][i]))
        for run_index in indices:
            rows.append({
                "method": results_storage['dispersion'][run_index],
                "disp":   results_storage['dispersion'][run_index],
                "coeff":  results_storage['dispersion_coeff'][run_index],
                "loc":    results_storage['dispersion_loc'][run_index],
                "idx":    run_index,
                "src":    ("run","best"),
            })

    baseline_reference_values = {
        name: value_at_index_percentage(sorted_cache, baseline_run_index, name, step_index=None, use_initial=True)
        for name in metric_names_for_table
    }

    baseline_mean_A_initial, baseline_std_A_initial = per_seed_avg_metrics_then_mean_std_across_seeds(
        results_storage,
        sorted_cache,
        baseline_run_index,
        metric_names_for_table,
        step_index_merged=0,
        use_initial=True,
    )

    column_alignment = "l c c " + " ".join(["c"]*len(metric_names_for_table)) + " c"
    header_names = [name.replace("\n"," ").replace(",", " ") for name in metric_names_for_table]
    header_names.append("Average")

    lines = []
    lines.append(r"\begin{tabular}{"+column_alignment+r"}")
    lines.append(r"\toprule")
    lines.append("Method & Coeff & Loc & " + " & ".join(header_names) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        left_cells = f"{row['method']} & {row['coeff']} & {row['loc']}"
        metric_cells_line1 = []
        metric_cells_line2 = []

        for metric_name in metric_names_for_table:
            if row["src"] == ("baseline", "initial"):
                value = baseline_reference_values[metric_name]
                std_pct = value_at_index_std_percentage(
                    sorted_cache, baseline_run_index, metric_name, step_index=None, use_initial=True
                )
                baseline_value = None
            else:
                run_i = baseline_run_index if row["src"] == ("baseline", "best") else row["idx"]
                value, std_pct = mean_std_metric_at_per_seed_best(
                    results_storage["per_seed_metrics"][run_i],
                    metric_name,
                    best_steps_per_run[run_i],
                    as_percent=True,
                )
                baseline_value = baseline_reference_values[metric_name]

            metric_cells_line1.append(_latex_cell_mean_pm_std(value, std_pct, decimals))
            if not np.isfinite(value):
                metric_cells_line2.append("N/A")
            elif row["src"] == ("baseline", "initial"):
                metric_cells_line2.append("---")
            else:
                metric_cells_line2.append(_latex_cell_delta_colored(value, baseline_value, decimals))

        if row["src"] == ("baseline", "initial"):
            mean_A, std_A = baseline_mean_A_initial, baseline_std_A_initial
        elif row["src"] == ("baseline", "best"):
            mean_A, std_A = composite_mean_std_at_per_seed_best(
                results_storage["per_seed_metrics"][baseline_run_index],
                metric_names_for_table,
                best_steps_per_run[baseline_run_index],
                as_percent=True,
            )
        else:
            run_index_avg = row["idx"]
            mean_A, std_A = composite_mean_std_at_per_seed_best(
                results_storage["per_seed_metrics"][run_index_avg],
                metric_names_for_table,
                best_steps_per_run[run_index_avg],
                as_percent=True,
            )

        if np.isfinite(mean_A) and np.isfinite(std_A):
            average_cell_line1 = f"{mean_A:.{decimals_average}f} $\\pm$ {std_A:.{decimals_average}f}"
        elif np.isfinite(mean_A):
            average_cell_line1 = f"{mean_A:.{decimals_average}f} $\\pm$ {0.0:.{decimals_average}f}"
        else:
            average_cell_line1 = "N/A"

        if not np.isfinite(mean_A):
            average_cell_line2 = "N/A"
        elif row["src"] == ("baseline", "initial"):
            average_cell_line2 = "---"
        else:
            average_cell_line2 = _latex_cell_delta_colored(mean_A, baseline_mean_A_initial, decimals_average)
        metric_cells_line1.append(average_cell_line1)
        metric_cells_line2.append(average_cell_line2)
        lines.append(left_cells + " & " + " & ".join(metric_cells_line1) + r" \\")
        lines.append(r" &  &  & " + " & ".join(metric_cells_line2) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_tex = "\n".join(lines)
    if output_path is None:
        os.makedirs("./tables", exist_ok=True)
        output_path = f"./tables/results_table_{model_name}{lora_suffix}.tex"
    with open(output_path, "w") as f:
        f.write(table_tex)

    print("\n===== LaTeX table (copy into Overleaf) =====\n")
    print(table_tex)
    print(f"\n[Saved LaTeX table to: {output_path}]\n")

def main(args):
    for k in results_dict:
        results_dict[k].clear()

    lora_suffix = "_lora" if args.lora else ""

    result_folder = './results/'
    figure_lines_save_path = f'./figures/results_lines_{args.model_name}{lora_suffix}.png'
    figure_bars_save_path = f'./figures/results_bars_{args.model_name}{lora_suffix}.png'

    os.makedirs(os.path.dirname(figure_lines_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_bars_save_path), exist_ok=True)
    pattern = os.path.join(
        result_folder,
        f'midtrain_{args.model_name}{lora_suffix}_{"-".join(args.dataset_name.split("/"))}_*',
    )
    run_folder_list = sorted(glob(pattern))
    run_folder_list = [
        run_folder
        for run_folder in run_folder_list
        if len(glob(os.path.join(run_folder, "lm_eval_*.json"))) > 0
        and is_midtrain_results_folder_basename(os.path.basename(run_folder.rstrip(os.sep)))
    ]

    grouped = defaultdict(list)
    for run_folder in run_folder_list:
        bn = os.path.basename(run_folder.rstrip(os.sep))
        grouped[run_key_from_folder_basename(bn)].append(run_folder)

    for run_key in sorted(grouped.keys()):
        folders = sorted(grouped[run_key])
        ref_folder = folders[0]
        dispersion_name, dispersion_coefficient, dispersion_location = parse_run_folder_basename(
            os.path.basename(ref_folder.rstrip(os.sep))
        )

        if dispersion_name not in ("ccnoise", "ccforget") and float(dispersion_coefficient) > 1:
            continue

        seed_metrics = [load_folder_metrics(f, empty_metrics_dict) for f in folders]
        merged = aggregate_metrics_across_seeds(seed_metrics, empty_metrics_dict)

        results_dict["dispersion"].append(dispersion_name)
        results_dict["dispersion_coeff"].append(dispersion_coefficient)
        results_dict["dispersion_loc"].append(dispersion_location)
        results_dict["metrics"].append(merged)
        results_dict["per_seed_metrics"].append(seed_metrics)

    if not results_dict["metrics"]:
        raise RuntimeError(
            "No runs found: check ./results/ for folders with lm_eval JSONs and "
            "names matching disp-*, _ccnoise-*, or _ccforget-* (same dataset/model as CLI)."
        )

    all_metric_names = [k for k in results_dict['metrics'][0].keys() if k != 'step']

    baseline_run_index = None
    for i, dispersion_name in enumerate(results_dict['dispersion']):
        if dispersion_name.lower() == "none":
            baseline_run_index = i
            break
    if baseline_run_index is None:
        raise RuntimeError("No baseline run found with dispersion == 'None'.")

    rows_by_dispersion = {}
    for i, dispersion_name in enumerate(results_dict['dispersion']):
        if i == baseline_run_index:
            continue
        rows_by_dispersion.setdefault(dispersion_name, []).append(i)

    method_order = [
        "decorrelation",
        "l2_repel",
        "angular_spread",
        "orthogonalization",
        "perplexity_entropy",
        "ccnoise",
        "ccforget",
    ]
    dispersion_order = [d for d in method_order if d in rows_by_dispersion]
    if not dispersion_order:
        dispersion_order = ["Baseline"]
        rows_by_dispersion["Baseline"] = []

    selection_metrics = [name for name in all_metric_names if "perplexity" not in name.lower()]
    sorted_cache = build_sorted_cache(results_dict)
    best_steps_per_run = {
        i: best_training_step_per_seed(results_dict["per_seed_metrics"][i], selection_metrics)
        for i in range(len(results_dict["metrics"]))
    }

    metric_ylim_ranges = compute_metric_ylim_by_per_seed_best(
        results_storage=results_dict,
        all_metric_names=all_metric_names,
        baseline_run_index=baseline_run_index,
        best_steps_per_run=best_steps_per_run,
        sorted_cache=sorted_cache,
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    figure_lines = plt.figure(figsize=(4*(len(all_metric_names)+1), 3*len(dispersion_order)))
    figure_bars = plt.figure(figsize=(4*(len(all_metric_names)+1), 3*len(dispersion_order)))

    for row_index, dispersion_name in enumerate(dispersion_order):
        run_indices_for_row = rows_by_dispersion[dispersion_name]
        color_map = cm.Reds

        for metric_index, metric_name in enumerate(all_metric_names):
            axis_lines = figure_lines.add_subplot(
                len(dispersion_order), len(all_metric_names)+1,
                row_index * (len(all_metric_names)+1) + metric_index + 1
            )
            axis_lines.spines["top"].set_visible(False)
            axis_lines.spines["right"].set_visible(False)

            axis_bars = figure_bars.add_subplot(
                len(dispersion_order), len(all_metric_names)+1,
                row_index * (len(all_metric_names)+1) + metric_index + 1
            )
            axis_bars.spines["top"].set_visible(False)
            axis_bars.spines["right"].set_visible(False)
            bar_labels = []
            bar_heights = []
            bar_errs = []
            bar_colors = []

            steps_baseline = sorted_cache[(baseline_run_index, '__steps__')]
            _, means_baseline, stds_baseline = sorted_cache[(baseline_run_index, metric_name)]
            stds_baseline = np.nan_to_num(np.asarray(stds_baseline, dtype=float), nan=0.0)
            if means_baseline.size > 0:
                bar_labels.append("No mid-training")
                bar_heights.append(float(means_baseline[0]))
                bar_errs.append(float(stds_baseline[0]) if stds_baseline.size > 0 else 0.0)
                bar_colors.append("lightgray")
                axis_lines.plot(
                    steps_baseline,
                    means_baseline,
                    linestyle="--",
                    linewidth=2,
                    label="Default loss",
                    color="black",
                    alpha=0.5,
                )
                axis_lines.fill_between(
                    steps_baseline,
                    np.asarray(means_baseline, dtype=float) - stds_baseline,
                    np.asarray(means_baseline, dtype=float) + stds_baseline,
                    color="black",
                    alpha=0.12,
                )

            m_bl, s_bl = mean_std_metric_at_per_seed_best(
                results_dict["per_seed_metrics"][baseline_run_index],
                metric_name,
                best_steps_per_run[baseline_run_index],
                as_percent=False,
            )
            if np.isfinite(m_bl):
                bar_labels.append("Default loss")
                bar_heights.append(m_bl)
                bar_errs.append(s_bl)
                bar_colors.append("gray")

            for run_index in run_indices_for_row:
                steps_run, means_run, stds_run = sorted_cache[(run_index, metric_name)]
                stds_run = np.nan_to_num(np.asarray(stds_run, dtype=float), nan=0.0)
                coeff_scaled = coeff_scaled_for_colormap(
                    results_dict["dispersion"][run_index],
                    results_dict["dispersion_coeff"][run_index],
                )
                c = color_map(coeff_scaled)
                axis_lines.plot(
                    steps_run,
                    means_run,
                    linewidth=2,
                    label=format_run_label(
                        results_dict['dispersion'][run_index],
                        results_dict['dispersion_coeff'][run_index],
                        results_dict['dispersion_loc'][run_index],
                    ),
                    color=c,
                )
                axis_lines.fill_between(
                    steps_run,
                    np.asarray(means_run, dtype=float) - stds_run,
                    np.asarray(means_run, dtype=float) + stds_run,
                    color=c,
                    alpha=0.2,
                )
                m_run, s_run = mean_std_metric_at_per_seed_best(
                    results_dict["per_seed_metrics"][run_index],
                    metric_name,
                    best_steps_per_run[run_index],
                    as_percent=False,
                )
                if np.isfinite(m_run):
                    bar_labels.append(
                        format_run_label(
                            results_dict['dispersion'][run_index],
                            results_dict['dispersion_coeff'][run_index],
                            results_dict['dispersion_loc'][run_index],
                        )
                    )
                    bar_heights.append(m_run)
                    bar_errs.append(s_run)
                    bar_colors.append(c)

            axis_lines.set_xlabel("Step", fontsize=12)
            axis_lines.set_ylabel(metric_name, fontsize=12)

            bars = axis_bars.bar(
                np.arange(len(bar_labels)),
                bar_heights,
                yerr=bar_errs,
                capsize=2,
                color=bar_colors,
                alpha=0.8,
                label=bar_labels,
                ecolor="black",
            )
            if len(bar_heights) >= 2:
                axis_bars.axhline(y=bar_heights[1], linestyle='--', linewidth=2, color=bar_colors[1], alpha=0.8)
            axis_bars.set_xticks(np.arange(len(bar_labels)))
            axis_bars.set_xticklabels([extract_coefficient_from_label(label) for label in bar_labels], rotation=0, ha='center', fontsize=9)
            axis_bars.set_ylabel(metric_name, fontsize=12)
            axis_bars.set_xlabel("Hyperparameter", fontsize=11)
            axis_bars.set_ylim(metric_ylim_ranges[metric_name])

            axis_bars.bar_label(
                bars,
                labels=[f"{v:.3f}" for v in bar_heights],
                rotation=90,
                padding=5,
                fontsize=9,
            )

            if metric_index == 0:
                axis_lines.legend(fontsize=10, frameon=False, loc='upper right')
                legend_bars = axis_bars.legend(fontsize=12, frameon=False, loc='center left')
                axis_bars.cla()
                axis_bars.axis("off")
                axis_bars.add_artist(legend_bars)

        axis_average = figure_lines.add_subplot(
            len(dispersion_order), len(all_metric_names)+1,
            row_index * (len(all_metric_names)+1) + (len(all_metric_names)) + 1
        )
        axis_average.spines["top"].set_visible(False)
        axis_average.spines["right"].set_visible(False)

        per_seed_bl = results_dict["per_seed_metrics"][baseline_run_index]
        st_bl, avg_bl_m, avg_bl_s = average_curve_with_seed_spread(per_seed_bl, all_metric_names)
        avg_bl_s = np.nan_to_num(np.asarray(avg_bl_s, dtype=float), nan=0.0)
        steps_merged_baseline = sorted_cache[(baseline_run_index, '__steps__')]

        candidate_average_values = []

        if np.asarray(avg_bl_m, dtype=float).size > 0:
            axis_average.plot(
                st_bl,
                avg_bl_m,
                linestyle="--",
                linewidth=2,
                label="Default loss",
                color="black",
                alpha=0.5,
            )
            axis_average.fill_between(
                st_bl,
                np.asarray(avg_bl_m, dtype=float) - avg_bl_s,
                np.asarray(avg_bl_m, dtype=float) + avg_bl_s,
                color="black",
                alpha=0.12,
            )
            extend_ylim_candidates_from_band(candidate_average_values, avg_bl_m, avg_bl_s)

        bar_labels_avg, bar_heights_avg, bar_errs_avg, bar_colors_avg = [], [], [], []

        if steps_merged_baseline.size > 0:
            s0 = int(steps_merged_baseline[0])
            m0, e0 = average_scalar_at_step_from_seed_curves(st_bl, avg_bl_m, avg_bl_s, s0)
            if np.isfinite(m0):
                candidate_average_values.extend([m0, m0 + e0, m0 - e0])
                bar_labels_avg.append("No mid-training")
                bar_heights_avg.append(m0)
                bar_errs_avg.append(e0)
                bar_colors_avg.append("lightgray")

        mb, eb = composite_mean_std_at_per_seed_best(
            per_seed_bl,
            selection_metrics,
            best_steps_per_run[baseline_run_index],
            as_percent=False,
        )
        if np.isfinite(mb):
            candidate_average_values.extend([mb, mb + eb, mb - eb])
            bar_labels_avg.append("Default loss")
            bar_heights_avg.append(mb)
            bar_errs_avg.append(eb)
            bar_colors_avg.append("gray")

        for run_index in run_indices_for_row:
            st_r, avg_r_m, avg_r_s = average_curve_with_seed_spread(
                results_dict["per_seed_metrics"][run_index], all_metric_names
            )
            avg_r_s = np.nan_to_num(np.asarray(avg_r_s, dtype=float), nan=0.0)
            coeff_scaled = coeff_scaled_for_colormap(
                results_dict["dispersion"][run_index],
                results_dict["dispersion_coeff"][run_index],
            )
            c = color_map(coeff_scaled)
            axis_average.plot(
                st_r,
                avg_r_m,
                linewidth=2,
                label=format_run_label(
                    results_dict["dispersion"][run_index],
                    results_dict["dispersion_coeff"][run_index],
                    results_dict["dispersion_loc"][run_index],
                ),
                color=c,
            )
            axis_average.fill_between(
                st_r,
                np.asarray(avg_r_m, dtype=float) - avg_r_s,
                np.asarray(avg_r_m, dtype=float) + avg_r_s,
                color=c,
                alpha=0.2,
            )
            extend_ylim_candidates_from_band(candidate_average_values, avg_r_m, avg_r_s)
            mr, er = composite_mean_std_at_per_seed_best(
                results_dict["per_seed_metrics"][run_index],
                selection_metrics,
                best_steps_per_run[run_index],
                as_percent=False,
            )
            if np.isfinite(mr):
                candidate_average_values.extend([mr, mr + er, mr - er])
                bar_labels_avg.append(
                    format_run_label(
                        results_dict["dispersion"][run_index],
                        results_dict["dispersion_coeff"][run_index],
                        results_dict["dispersion_loc"][run_index],
                    )
                )
                bar_heights_avg.append(mr)
                bar_errs_avg.append(er)
                bar_colors_avg.append(c)

        if candidate_average_values:
            y_lo = float(np.nanmin(candidate_average_values))
            y_hi = float(np.nanmax(candidate_average_values))
        else:
            y_lo, y_hi = np.nan, np.nan
        axis_average.set_ylim(*_ylim_with_padding(y_lo, y_hi))
        axis_average.set_xlabel("Step", fontsize=12)
        axis_average.set_ylabel("Average", fontsize=12)

        axis_bars_avg = figure_bars.add_subplot(
            len(dispersion_order), len(all_metric_names)+1,
            row_index * (len(all_metric_names)+1) + len(all_metric_names) + 1
        )
        axis_bars_avg.spines["top"].set_visible(False)
        axis_bars_avg.spines["right"].set_visible(False)

        bars_avg = axis_bars_avg.bar(
            np.arange(len(bar_labels_avg)),
            bar_heights_avg,
            yerr=bar_errs_avg,
            capsize=2,
            color=bar_colors_avg,
            alpha=0.8,
            label=bar_labels_avg,
            ecolor="black",
        )
        if len(bar_heights_avg) >= 2:
            axis_bars_avg.axhline(y=bar_heights_avg[1], linestyle="--", linewidth=2, color=bar_colors_avg[1], alpha=0.8)
        axis_bars_avg.set_xticks(np.arange(len(bar_labels_avg)))
        axis_bars_avg.set_xticklabels(
            [extract_coefficient_from_label(l) for l in bar_labels_avg],
            rotation=0,
            ha="center",
            fontsize=9,
        )
        axis_bars_avg.set_ylabel("Average", fontsize=12)
        axis_bars_avg.set_xlabel("Hyperparameter", fontsize=11)

        if bar_heights_avg:
            low = [h - e for h, e in zip(bar_heights_avg, bar_errs_avg)]
            high = [h + e for h, e in zip(bar_heights_avg, bar_errs_avg)]
            ymin_b, ymax_b = float(np.nanmin(low)), float(np.nanmax(high))
            if ymax_b == ymin_b:
                eps_b = 1e-6 if ymin_b == 0 else 0.01 * abs(ymin_b)
                ymin_b, ymax_b = ymin_b - eps_b, ymax_b + eps_b
            pad_b = 0.05 * (ymax_b - ymin_b)
            axis_bars_avg.set_ylim(ymin_b - pad_b, ymax_b + pad_b)

        axis_bars_avg.bar_label(
            bars_avg,
            labels=[f"{v:.3f}" for v in bar_heights_avg],
            rotation=90,
            padding=5,
            fontsize=9,
        )

    figure_lines.tight_layout(pad=2)
    figure_lines.savefig(figure_lines_save_path, dpi=300)

    figure_bars.tight_layout(pad=2)
    figure_bars.savefig(figure_bars_save_path, dpi=300)

    render_latex_table(
        results_storage=results_dict,
        metric_names_for_table=selection_metrics,
        baseline_run_index=baseline_run_index,
        rows_by_dispersion=rows_by_dispersion,
        dispersion_order=dispersion_order,
        model_name=args.model_name,
        lora_suffix=lora_suffix,
        decimals=1,
        decimals_average=2,
        best_steps_per_run=best_steps_per_run,
        sorted_cache=sorted_cache,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mid-train Qwen3 with a token budget.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    args = parser.parse_args()
    args.model_name = args.model_name.replace('/', '-')
    main(args)