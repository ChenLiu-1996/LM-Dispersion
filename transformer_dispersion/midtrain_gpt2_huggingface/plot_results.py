import argparse
import os
import numpy as np
from glob import glob
import json
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy

results_dict = {
    'dispersion': [],
    'dispersion_coeff': [],
    'dispersion_loc': [],
    'metrics': [],
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
    except:
        return np.inf

def compute_best_steps(results_storage, selection_metric_names):
    best_step_index_per_run = {}
    sorted_cache = {}

    for run_index in range(len(results_storage['metrics'])):
        steps_array = np.asarray(results_storage['metrics'][run_index]['step'], dtype=int)
        if steps_array.size == 0:
            best_step_index_per_run[run_index] = None
            continue

        order = np.argsort(steps_array)
        steps_sorted = steps_array[order]
        average_scores_over_metrics = np.full_like(steps_sorted, fill_value=np.nan, dtype=float)

        per_metric_mean_values = {}
        for metric_name in selection_metric_names:
            means_array = np.asarray(results_storage['metrics'][run_index][metric_name]['mean'], dtype=float)
            if means_array.size == 0:
                continue
            per_metric_mean_values[metric_name] = means_array[order]

        for time_index in range(len(steps_sorted)):
            values_at_step = []
            for metric_name in selection_metric_names:
                if metric_name in per_metric_mean_values and np.isfinite(per_metric_mean_values[metric_name][time_index]):
                    values_at_step.append(per_metric_mean_values[metric_name][time_index])
            if len(values_at_step) > 0:
                average_scores_over_metrics[time_index] = float(np.mean(values_at_step))

        if np.all(~np.isfinite(average_scores_over_metrics)):
            best_step_index_per_run[run_index] = None
        else:
            best_step_index_per_run[run_index] = int(np.nanargmax(average_scores_over_metrics))

        sorted_cache[(run_index, '__steps__')] = steps_sorted
        for metric_key in results_storage['metrics'][run_index].keys():
            if metric_key == 'step':
                continue
            means_array = np.asarray(results_storage['metrics'][run_index][metric_key]['mean'], dtype=float)
            stds_array = np.asarray(results_storage['metrics'][run_index][metric_key]['std'], dtype=float)
            if means_array.size == 0:
                sorted_cache[(run_index, metric_key)] = (steps_sorted, means_array, stds_array)
            else:
                sorted_cache[(run_index, metric_key)] = sort_series_by_step(steps_array, means_array, stds_array)

    return best_step_index_per_run, sorted_cache

def value_at_index_percentage(results_storage, sorted_cache, run_index, metric_name, step_index, use_initial=False):
    steps_sorted = sorted_cache[(run_index, '__steps__')]
    means_array = np.asarray(results_storage['metrics'][run_index][metric_name]['mean'], dtype=float)
    if means_array.size == 0 or steps_sorted.size == 0:
        return np.nan
    _, means_sorted, _ = sorted_cache[(run_index, metric_name)]
    if means_sorted.size == 0:
        return np.nan
    index_to_use = 0 if use_initial else step_index
    if index_to_use is None or index_to_use >= means_sorted.size:
        return np.nan
    return float(means_sorted[index_to_use]) * 100.0

def compute_metric_ylim_by_best_step(results_storage, all_metric_names, baseline_run_index, best_step_index_per_run, sorted_cache):
    metric_ylim_ranges = {metric_name: [np.inf, -np.inf] for metric_name in all_metric_names}
    for metric_name in all_metric_names:
        candidate_values = []

        _, means_baseline, _ = sorted_cache[(baseline_run_index, metric_name)]
        if means_baseline.size > 0:
            candidate_values.append(float(means_baseline[0]))

        baseline_best_index = best_step_index_per_run[baseline_run_index]
        if baseline_best_index is not None and means_baseline.size > baseline_best_index:
            candidate_values.append(float(means_baseline[baseline_best_index]))

        for run_index in range(len(results_storage['metrics'])):
            if run_index == baseline_run_index:
                continue
            _, means_run, _ = sorted_cache[(run_index, metric_name)]
            best_index = best_step_index_per_run[run_index]
            if means_run.size > 0 and best_index is not None and best_index < means_run.size and np.isfinite(means_run[best_index]):
                candidate_values.append(float(means_run[best_index]))

        if candidate_values:
            min_value = float(np.nanmin(candidate_values))
            max_value = float(np.nanmax(candidate_values))
            metric_ylim_ranges[metric_name][0] = min_value
            metric_ylim_ranges[metric_name][1] = max_value

    for metric_name, (ymin, ymax) in metric_ylim_ranges.items():
        if not np.isfinite(ymin):
            ymin = 0.0
        if not np.isfinite(ymax):
            ymax = 1.0
        if ymax == ymin:
            epsilon = 1e-6 if ymin == 0 else 0.01 * abs(ymin)
            ymin, ymax = ymin - epsilon, ymax + epsilon
        padding = 0.05 * (ymax - ymin)
        metric_ylim_ranges[metric_name] = (ymin - padding, ymax + padding)

    return metric_ylim_ranges

def average_metric_of_run(sorted_cache, run_index, metric_name_list):
    steps_sorted = sorted_cache[(run_index, '__steps__')]
    series_list = []
    for metric_name in metric_name_list:
        if 'perplexity' in metric_name:
            continue
        _, means_sorted, _ = sorted_cache[(run_index, metric_name)]
        if means_sorted.size > 0:
            series_list.append(np.asarray(means_sorted, dtype=float))
    if not series_list:
        return steps_sorted, np.array([])
    stacked = np.vstack(series_list)
    average_series = np.nanmean(stacked, axis=0)
    return steps_sorted, average_series

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
    best_step_index_per_run=None,
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

    baseline_reference_values = {}
    for metric_name in metric_names_for_table:
        value = value_at_index_percentage(results_storage, sorted_cache, baseline_run_index, metric_name, step_index=None, use_initial=True)
        baseline_reference_values[metric_name] = value

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
        metric_cells = []
        values_for_average = []
        baseline_values_for_average = []

        for metric_name in metric_names_for_table:
            if row['src'] == ("baseline","initial"):
                value = baseline_reference_values[metric_name]
                baseline_value = baseline_reference_values[metric_name]
                cell_text = f"{value:.{decimals}f}" if np.isfinite(value) else "N/A"
            elif row['src'] == ("baseline","best"):
                baseline_best_index = best_step_index_per_run[baseline_run_index]
                value = value_at_index_percentage(results_storage, sorted_cache, baseline_run_index, metric_name, step_index=baseline_best_index, use_initial=False)
                baseline_value = baseline_reference_values[metric_name]
                cell_text = f"{value:.{decimals}f}"
                if np.isfinite(baseline_value):
                    difference = np.round(value, decimals) - np.round(baseline_value, decimals)
                    sign = "+" if difference >= 0 else ""
                    color_name = "forestgreen" if difference >= 0 else "crimson"
                    cell_text += f"$_{{\\textcolor{{{color_name}}}{{({sign}{difference:.{decimals}f})}}}}$"
            else:
                run_index = row['idx']
                best_index = best_step_index_per_run[run_index]
                value = value_at_index_percentage(results_storage, sorted_cache, run_index, metric_name, step_index=best_index, use_initial=False)
                baseline_value = baseline_reference_values[metric_name]
                cell_text = f"{value:.{decimals}f}"
                if np.isfinite(baseline_value):
                    difference = np.round(value, decimals) - np.round(baseline_value, decimals)
                    sign = "+" if difference >= 0 else ""
                    color_name = "forestgreen" if difference >= 0 else "crimson"
                    cell_text += f"$_{{\\textcolor{{{color_name}}}{{({sign}{difference:.{decimals}f})}}}}$"

            metric_cells.append(cell_text)
            if np.isfinite(value):
                values_for_average.append(value)
            if np.isfinite(baseline_value):
                baseline_values_for_average.append(baseline_value)

        if values_for_average:
            average_value = float(np.mean(values_for_average))
            average_baseline = float(np.mean(baseline_values_for_average)) if baseline_values_for_average else np.nan
            if row['src'] == ("baseline","initial") or not np.isfinite(average_baseline):
                average_cell_text = f"{average_value:.{decimals_average}f}"
            else:
                difference = np.round(average_value, decimals_average) - np.round(average_baseline, decimals_average)
                sign = "+" if difference >= 0 else ""
                color_name = "forestgreen" if difference >= 0 else "crimson"
                average_cell_text = f"{average_value:.{decimals_average}f}$_{{\\textcolor{{{color_name}}}{{({sign}{difference:.{decimals_average}f})}}}}$"
        else:
            average_cell_text = "N/A"

        metric_cells.append(average_cell_text)
        lines.append(left_cells + " & " + " & ".join(metric_cells) + r" \\")
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
    lora_suffix = "_lora" if args.lora else ""

    result_folder = './results/'
    figure_lines_save_path = f'./figures/results_lines_{args.model_name}{lora_suffix}.png'
    figure_bars_save_path = f'./figures/results_bars_{args.model_name}{lora_suffix}.png'

    os.makedirs(os.path.dirname(figure_lines_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_bars_save_path), exist_ok=True)
    run_folder_list = sorted(glob(os.path.join(result_folder, f'midtrain_{args.model_name}{lora_suffix}_{"-".join(args.dataset_name.split("/"))}_*seed-1*')))

    # Ignore the folder if the folder does not contain any `lm_eval_*.json` files.
    run_folder_list = [run_folder for run_folder in run_folder_list if len(glob(os.path.join(run_folder, 'lm_eval_*.json'))) > 0]

    for run_folder in run_folder_list:
        dispersion_name = run_folder.split('disp-')[1].split('-')[0]
        dispersion_coefficient = run_folder.split(f'{dispersion_name}-')[1].split('-')[0]
        dispersion_location = run_folder.split(f'{dispersion_coefficient}-')[1].split('-')[0]

        if float(dispersion_coefficient) > 1:
            continue

        results_dict['dispersion'].append(dispersion_name)
        results_dict['dispersion_coeff'].append(dispersion_coefficient)
        results_dict['dispersion_loc'].append(dispersion_location)
        results_dict['metrics'].append(deepcopy(empty_metrics_dict))

        eval_json_list = glob(os.path.join(run_folder, 'lm_eval_*.json'))

        for eval_json in eval_json_list:
            with open(eval_json, "r") as f:
                data_json = json.load(f)

            results_dict['metrics'][-1]['step'].append(int(eval_json.split('_')[-1].replace('.json', '')))
            for metric_key in empty_metrics_dict.keys():
                if metric_key != 'step':
                    metric_dataset = metric_key.split('\n')[0]
                    metric_measure = metric_key.split('\n')[1]
                    results_dict['metrics'][-1][metric_key]['mean'].append(
                        float(data_json['results'][metric_dataset][metric_measure]))
                    std_value = data_json['results'][metric_dataset][metric_measure.replace(',', '_stderr,')]
                    if std_value == 'N/A':
                        results_dict['metrics'][-1][metric_key]['std'].append(np.nan)
                    else:
                        results_dict['metrics'][-1][metric_key]['std'].append(float(std_value))

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

    dispersion_order = [d for d in ["decorrelation", "l2_repel", "angular_spread", "orthogonalization", "perplexity_entropy"] if d in rows_by_dispersion]
    if not dispersion_order:
        dispersion_order = ["Baseline"]
        rows_by_dispersion["Baseline"] = []

    selection_metrics = [name for name in all_metric_names if 'perplexity' not in name.lower()]
    best_step_index_per_run, sorted_cache = compute_best_steps(results_dict, selection_metrics)

    metric_ylim_ranges = compute_metric_ylim_by_best_step(
        results_storage=results_dict,
        all_metric_names=all_metric_names,
        baseline_run_index=baseline_run_index,
        best_step_index_per_run=best_step_index_per_run,
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
            bar_colors = []

            steps_baseline = sorted_cache[(baseline_run_index, '__steps__')]
            _, means_baseline, _ = sorted_cache[(baseline_run_index, metric_name)]
            if means_baseline.size > 0:
                bar_labels.append('No mid-training')
                bar_heights.append(float(means_baseline[0]))
                bar_colors.append('lightgray')

            baseline_best_index = best_step_index_per_run[baseline_run_index]
            if baseline_best_index is not None and means_baseline.size > baseline_best_index:
                bar_labels.append('Default loss')
                bar_heights.append(float(means_baseline[baseline_best_index]))
                bar_colors.append('gray')
                axis_lines.plot(steps_baseline, means_baseline, linestyle='--', linewidth=2, label='Default loss', color='black', alpha=0.5)

            for run_index in run_indices_for_row:
                steps_run = sorted_cache[(run_index, '__steps__')]
                _, means_run, _ = sorted_cache[(run_index, metric_name)]
                best_index = best_step_index_per_run[run_index]
                coeff_value = float(results_dict['dispersion_coeff'][run_index])
                coeff_scaled = (np.log10(coeff_value) + 4) / 7
                axis_lines.plot(
                    steps_run, means_run, linewidth=2,
                    label=format_run_label(results_dict['dispersion'][run_index],
                                          results_dict['dispersion_coeff'][run_index],
                                          results_dict['dispersion_loc'][run_index]),
                    color=color_map(coeff_scaled)
                )
                if means_run.size > 0 and best_index is not None and best_index < means_run.size:
                    bar_labels.append(format_run_label(results_dict['dispersion'][run_index],
                                                       results_dict['dispersion_coeff'][run_index],
                                                       results_dict['dispersion_loc'][run_index]))
                    bar_heights.append(float(means_run[best_index]))
                    bar_colors.append(color_map(coeff_scaled))

            axis_lines.set_xlabel("Step", fontsize=12)
            axis_lines.set_ylabel(metric_name, fontsize=12)

            bars = axis_bars.bar(np.arange(len(bar_labels)), bar_heights, color=bar_colors, alpha=0.8, label=bar_labels)
            if len(bar_heights) >= 2:
                axis_bars.axhline(y=bar_heights[1], linestyle='--', linewidth=2, color=bar_colors[1], alpha=0.8)
            axis_bars.set_xticks(np.arange(len(bar_labels)))
            axis_bars.set_xticklabels([extract_coefficient_from_label(label) for label in bar_labels], rotation=0, ha='center', fontsize=9)
            axis_bars.set_ylabel(metric_name, fontsize=12)
            axis_bars.set_xlabel('Dispersion Coefficient', fontsize=12)
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

        steps_baseline_avg, average_baseline_series = average_metric_of_run(sorted_cache, baseline_run_index, all_metric_names)
        if average_baseline_series.size > 0:
            axis_average.plot(steps_baseline_avg, average_baseline_series, linestyle='--', linewidth=2, label='Default loss', color='black', alpha=0.5)

        candidate_average_values = []
        baseline_best_index = best_step_index_per_run[baseline_run_index]
        if average_baseline_series.size > 0:
            candidate_average_values.append(float(average_baseline_series[0]))
            if baseline_best_index is not None and baseline_best_index < average_baseline_series.size and np.isfinite(average_baseline_series[baseline_best_index]):
                candidate_average_values.append(float(average_baseline_series[baseline_best_index]))

        for run_index in run_indices_for_row:
            steps_run_avg, average_run_series = average_metric_of_run(sorted_cache, run_index, all_metric_names)
            coeff_value = float(results_dict['dispersion_coeff'][run_index])
            coeff_scaled = (np.log10(coeff_value) + 4) / 7
            axis_average.plot(
                steps_run_avg, average_run_series, linewidth=2,
                label=format_run_label(results_dict['dispersion'][run_index],
                                       results_dict['dispersion_coeff'][run_index],
                                       results_dict['dispersion_loc'][run_index]),
                color=color_map(coeff_scaled)
            )
            best_index = best_step_index_per_run[run_index]
            if average_run_series.size > 0 and best_index is not None and best_index < average_run_series.size and np.isfinite(average_run_series[best_index]):
                candidate_average_values.append(float(average_run_series[best_index]))

        if candidate_average_values:
            ymin = float(np.nanmin(candidate_average_values))
            ymax = float(np.nanmax(candidate_average_values))
        else:
            ymin, ymax = 0.0, 1.0
        if ymax == ymin:
            epsilon = 1e-6 if ymin == 0 else 0.01 * abs(ymin)
            ymin, ymax = ymin - epsilon, ymax + epsilon
        padding = 0.05 * (ymax - ymin)
        axis_average.set_ylim(ymin - padding, ymax + padding)

        axis_average.set_xlabel("Step", fontsize=12)
        axis_average.set_ylabel("Average", fontsize=12)

        axis_bars_avg = figure_bars.add_subplot(
            len(dispersion_order), len(all_metric_names)+1,
            row_index * (len(all_metric_names)+1) + len(all_metric_names) + 1
        )
        axis_bars_avg.spines["top"].set_visible(False)
        axis_bars_avg.spines["right"].set_visible(False)

        bar_labels_avg, bar_heights_avg, bar_colors_avg = [], [], []

        if average_baseline_series.size > 0:
            bar_labels_avg.append('No mid-training')
            bar_heights_avg.append(float(average_baseline_series[0]))
            bar_colors_avg.append('lightgray')

        if average_baseline_series.size > 0 and baseline_best_index is not None and baseline_best_index < average_baseline_series.size:
            bar_labels_avg.append('Default loss')
            bar_heights_avg.append(float(average_baseline_series[baseline_best_index]))
            bar_colors_avg.append('gray')

        for run_index in run_indices_for_row:
            _, average_run_series = average_metric_of_run(sorted_cache, run_index, all_metric_names)
            best_index = best_step_index_per_run[run_index]
            if average_run_series.size > 0 and best_index is not None and best_index < average_run_series.size and np.isfinite(average_run_series[best_index]):
                bar_labels_avg.append(format_run_label(results_dict['dispersion'][run_index],
                                                       results_dict['dispersion_coeff'][run_index],
                                                       results_dict['dispersion_loc'][run_index]))
                bar_heights_avg.append(float(average_run_series[best_index]))
                coeff_value = float(results_dict['dispersion_coeff'][run_index])
                coeff_scaled = (np.log10(coeff_value) + 4) / 7
                bar_colors_avg.append(color_map(coeff_scaled))

        bars_avg = axis_bars_avg.bar(np.arange(len(bar_labels_avg)), bar_heights_avg, color=bar_colors_avg, alpha=0.8, label=bar_labels_avg)
        if len(bar_heights_avg) >= 2:
            axis_bars_avg.axhline(y=bar_heights_avg[1], linestyle='--', linewidth=2, color=bar_colors_avg[1], alpha=0.8)
        axis_bars_avg.set_xticks(np.arange(len(bar_labels_avg)))
        axis_bars_avg.set_xticklabels([extract_coefficient_from_label(l) for l in bar_labels_avg], rotation=0, ha='center', fontsize=9)
        axis_bars_avg.set_ylabel("Average", fontsize=12)
        axis_bars_avg.set_xlabel('Dispersion Coefficient', fontsize=12)

        if bar_heights_avg:
            ymin_b, ymax_b = float(np.nanmin(bar_heights_avg)), float(np.nanmax(bar_heights_avg))
            if ymax_b == ymin_b:
                eps_b = 1e-6 if ymin_b == 0 else 0.01 * abs(ymin_b)
                ymin_b, ymax_b = ymin_b - eps_b, ymax_b + eps_b
            pad_b = 0.05 * (ymax_b - ymin_b)
            axis_bars_avg.set_ylim(ymin_b - pad_b, ymax_b + pad_b)

        axis_bars_avg.bar_label(bars_avg, labels=[f"{v:.3f}" for v in bar_heights_avg], rotation=90, padding=5, fontsize=9)

    figure_lines.tight_layout(pad=2)
    figure_lines.savefig(figure_lines_save_path, dpi=300)

    figure_bars.tight_layout(pad=2)
    figure_bars.savefig(figure_bars_save_path, dpi=300)

    metrics_for_table = [name for name in all_metric_names if 'perplexity' not in name.lower()]
    render_latex_table(
        results_storage=results_dict,
        metric_names_for_table=metrics_for_table,
        baseline_run_index=baseline_run_index,
        rows_by_dispersion=rows_by_dispersion,
        dispersion_order=dispersion_order,
        model_name=args.model_name,
        lora_suffix=lora_suffix,
        decimals=1,
        decimals_average=2,
        best_step_index_per_run=best_step_index_per_run,
        sorted_cache=sorted_cache,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mid-train GPT-2 with a token budget.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext")
    args = parser.parse_args()
    main(args)