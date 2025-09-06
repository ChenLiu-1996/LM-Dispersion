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
    'paloma_wikitext_103\nword_perplexity': {'mean': [], 'std': []},
    'lambada_openai\nacc': {'mean': [], 'std': []},
    'lambada_standard\nacc': {'mean': [], 'std': []},
    'medmcqa\nacc': {'mean': [], 'std': []},
    'mmlu\nacc': {'mean': [], 'std': []},
}

metric_range_dict = {
    'paloma_wikitext_103\nword_perplexity': [20, 60],
    'lambada_openai\nacc': [0.29, 0.34],
    'lambada_standard\nacc': [0.23, 0.28],
    'medmcqa\nacc': [0.28, 0.34],
    'mmlu\nacc': [0.25, 0.265],
}

def sort_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]

def run_label(dispersion, coeff, loc):
    if str(dispersion) == 'None':
        return 'None'
    return f'{dispersion}-{coeff}-{loc}'

def extract_coeff_from_label(str):
    if str == 'None':
        return 0
    else:
        return float(str.split('-')[1].split('-')[0])

def best_over_history(means, stds, metric_name):
    if 'perplexity' in metric_name.lower():
        i = int(np.nanargmin(means))
    else:
        i = int(np.nanargmax(means))
    return float(means[i]), (np.nan if np.isnan(stds[i]) else float(stds[i]))


if __name__ == '__main__':
    result_folder = './results/midtrain_meta-llama/'
    figure_lines_save_path = './figures/results_lines.png'
    figure_bars_save_path = './figures/results_bars.png'

    os.makedirs(os.path.dirname(figure_lines_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_bars_save_path), exist_ok=True)
    run_folder_list = sorted(glob(os.path.join(result_folder, 'Llama-3.2-1B_*')))

    for run_folder in run_folder_list:
        dispersion = run_folder.split('disp-')[1].split('-')[0]
        dispersion_coeff = run_folder.split(f'{dispersion}-')[1].split('-')[0]
        dispersion_loc = run_folder.split(f'{dispersion_coeff}-')[1].split('_')[0]
        
        if float(dispersion_coeff) > 1:
            continue

        results_dict['dispersion'].append(dispersion)
        results_dict['dispersion_coeff'].append(dispersion_coeff)
        results_dict['dispersion_loc'].append(dispersion_loc)
        results_dict['metrics'].append(deepcopy(empty_metrics_dict))

        eval_json_list = glob(os.path.join(run_folder, 'lm_eval_*.json'))

        for eval_json in eval_json_list:
            with open(eval_json, "r") as f:
                data_json = json.load(f)

            results_dict['metrics'][-1]['step'].append(int(eval_json.split('_')[-1].replace('.json', '')))
            for metric in empty_metrics_dict.keys():
                if metric != 'step':
                    metric_dataset = metric.split('\n')[0]
                    metric_measure = metric.split('\n')[1]
                    results_dict['metrics'][-1][metric]['mean'].append(
                        float(data_json['results'][metric_dataset][f'{metric_measure},none']))
                    std = data_json['results'][metric_dataset][f'{metric_measure}_stderr,none']
                    if std == 'N/A':
                        results_dict['metrics'][-1][metric]['std'].append(np.nan)
                    else:
                        results_dict['metrics'][-1][metric]['std'].append(float(std))

    all_metric_names = [k for k in results_dict['metrics'][0].keys() if k != 'step']

    baseline_idx = None
    for i, d in enumerate(results_dict['dispersion']):
        if d.lower() == "none":
            baseline_idx = i
            break
    if baseline_idx is None:
        raise RuntimeError("No baseline run found with dispersion == 'None'.")

    # group indices by dispersion (excluding baseline)
    rows_by_dispersion = {}
    for i, d in enumerate(results_dict['dispersion']):
        if i == baseline_idx:
            continue
        rows_by_dispersion.setdefault(d, []).append(i)

    # enforce the requested row order
    row_order = [d for d in ["covariance", "hinge", "infonce_l2", "infonce_cosine"] if d in rows_by_dispersion]
    
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    fig_lines = plt.figure(figsize=(20, 12))
    fig_bars = plt.figure(figsize=(20, 12))

    for row_idx, row_disp in enumerate(row_order):
        run_indices = rows_by_dispersion[row_disp]
        # colors = ["#00C100", "#CE0E00", "#C8A300", "#005BC3"]
        cmap = cm.Reds
        
        for metric_idx, metric_name in enumerate(all_metric_names):
            ax_lines = fig_lines.add_subplot(len(row_order), len(all_metric_names), row_idx * len(all_metric_names) + metric_idx + 1)
            ax_lines.spines["top"].set_visible(False)
            ax_lines.spines["right"].set_visible(False)

            ax_bars = fig_bars.add_subplot(len(row_order), len(all_metric_names), row_idx * len(all_metric_names) + metric_idx + 1)
            ax_bars.spines["top"].set_visible(False)
            ax_bars.spines["right"].set_visible(False)
            bars_labels, bars_heights, bars_colors = [], [], []

            # Overlay baseline on every plot.
            baseline_steps = results_dict['metrics'][baseline_idx]['step']
            baseline_means = results_dict['metrics'][baseline_idx][metric_name]['mean']
            baseline_stds  = results_dict['metrics'][baseline_idx][metric_name]['std']
            baseline_steps, baseline_means, baseline_stds = sort_by_step(baseline_steps, baseline_means, baseline_stds)
            baseline_best, _ = best_over_history(baseline_means, baseline_stds, metric_name)
            baseline_label = run_label(results_dict['dispersion'][baseline_idx],
                                       results_dict['dispersion_coeff'][baseline_idx],
                                       results_dict['dispersion_loc'][baseline_idx])
            bars_labels.append(baseline_label)
            bars_heights.append(baseline_best)
            bars_colors.append('gray')
            ax_lines.plot(baseline_steps, baseline_means, linestyle='--', linewidth=2, label=baseline_label, color='black', alpha=0.5)

            # Plot every run of this dispersion.
            for idx in run_indices:
                curr_steps = results_dict['metrics'][idx]['step']
                curr_means = results_dict['metrics'][idx][metric_name]['mean']
                curr_stds = results_dict['metrics'][idx][metric_name]['std']
                curr_steps, curr_means, curr_stds = sort_by_step(curr_steps, curr_means, curr_stds)
                curr_best, _ = best_over_history(curr_means, curr_stds, metric_name)
                curr_label = run_label(results_dict['dispersion'][idx],
                                       results_dict['dispersion_coeff'][idx],
                                       results_dict['dispersion_loc'][idx])
                coeff_logscaled = (np.log10(float(results_dict['dispersion_coeff'][idx])) + 4) / 7
                ax_lines.plot(curr_steps, curr_means, linewidth=2, label=curr_label, color=cmap(coeff_logscaled))
                # ax.fill_between(xs, means - stds, means + stds, linewidth=2, color=cmap(coeff_logscaled), alpha=0.2)

                bars_labels.append(curr_label)
                bars_heights.append(curr_best)
                bars_colors.append(cmap(coeff_logscaled))

            ax_lines.set_xlabel("Step", fontsize=15)
            ax_lines.set_ylabel(metric_name, fontsize=15)

            ax_bars.bar(np.arange(len(bars_labels)), bars_heights, color=bars_colors, alpha=0.8, label=bars_labels)
            ax_bars.axhline(y=bars_heights[0], linestyle='--', linewidth=2, color=bars_colors[0], alpha=0.8)
            ax_bars.set_xticks(np.arange(len(bars_labels)))
            ax_bars.set_xticklabels([extract_coeff_from_label(label) for label in bars_labels], rotation=0, ha='center', fontsize=9)
            ax_bars.set_ylabel(metric_name, fontsize=15)
            ax_bars.set_xlabel('Dispersion Coefficient', fontsize=15)
            ax_bars.set_ylim(metric_range_dict[metric_name])

            if metric_idx == 0:
                ax_lines.legend(fontsize=10, frameon=False, loc='upper right')
                # For the barplots, remove the perplexity plot, but put the legend back.
                legend_bars = ax_bars.legend(fontsize=12, frameon=False, loc='center left')
                ax_bars.cla()
                ax_bars.axis("off")
                ax_bars.add_artist(legend_bars)

    fig_lines.tight_layout(pad=2)
    fig_lines.savefig(figure_lines_save_path, dpi=300)

    fig_bars.tight_layout(pad=2)
    fig_bars.savefig(figure_bars_save_path, dpi=300)
