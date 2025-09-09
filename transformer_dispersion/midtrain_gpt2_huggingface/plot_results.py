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
    # 'hellaswag\nacc_norm,none': {'mean': [], 'std': []},
    'lambada_openai\nacc,none': {'mean': [], 'std': []},
    'lambada_standard\nacc,none': {'mean': [], 'std': []},
    # 'piqa\nacc,none': {'mean': [], 'std': []},
    'truthfulqa_mc2\nacc,none': {'mean': [], 'std': []},
    'winogrande\nacc,none': {'mean': [], 'std': []},
    'arc_challenge\nacc,none': {'mean': [], 'std': []},
    'gsm8k\nexact_match,flexible-extract': {'mean': [], 'std': []},
    'medmcqa\nacc,none': {'mean': [], 'std': []},
    'mmlu\nacc,none': {'mean': [], 'std': []},
}


def sort_by_step(steps, means, stds):
    order = np.argsort(np.array(steps))
    return np.array(steps)[order], np.array(means)[order], np.array(stds)[order]

def run_label(dispersion, coeff, loc):
    if str(dispersion) == 'None':
        return 'None'
    return f'{dispersion}-{coeff}-{loc}'

def extract_coeff_from_label(str):
    if str == 'Default loss':
        return 0
    elif str == 'No mid-training':
        return 'N/A'
    else:
        return float(str.split('-')[1].split('-')[0])

def best_over_history(means, stds, metric_name):
    if 'perplexity' in metric_name.lower():
        i = int(np.nanargmin(means))
    else:
        i = int(np.nanargmax(means))
    return float(means[i]), (np.nan if np.isnan(stds[i]) else float(stds[i]))

def find_metric_ylims(results_dict, all_metric_names, baseline_idx):
    metric_ylim_bars = {m: [np.inf, -np.inf] for m in all_metric_names}

    for m in all_metric_names:
        cand = []
        for idx in range(len(results_dict['metrics'])):
            if idx == baseline_idx:
                # Take the initial result too (before mid-traininig).
                baseline_steps = results_dict['metrics'][idx]['step']
                baseline_means = results_dict['metrics'][idx][m]['mean']
                baseline_stds  = results_dict['metrics'][idx][m]['std']
                baseline_steps, baseline_means, baseline_stds = sort_by_step(baseline_steps, baseline_means, baseline_stds)
                cand.append(baseline_means[0])
            curr_means = np.asarray(results_dict['metrics'][idx][m]['mean'], dtype=float)
            if curr_means.size:
                curr_best, _ = best_over_history(curr_means, np.asarray(results_dict['metrics'][idx][m]['std']), m)
                cand.append(curr_best)

        if cand:
            vmin = np.nanmin(cand); vmax = np.nanmax(cand)
            metric_ylim_bars[m][0] = vmin; metric_ylim_bars[m][1] = vmax

    for m, (lo, hi) in metric_ylim_bars.items():
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi): hi = 1.0
        if hi == lo:
            eps = 1e-6 if lo == 0 else 0.01 * abs(lo)
            lo, hi = lo - eps, hi + eps
        pad = 0.05 * (hi - lo)
        metric_ylim_bars[m] = (lo - pad, hi + pad)

    return metric_ylim_bars

def _num_coeff(x):
    try: return float(x)
    except: return np.inf

def _best_or_initial_in_pct(results_dict, idx, metric_name, initial=False):
    means = np.asarray(results_dict['metrics'][idx][metric_name]['mean'], dtype=float)
    stds  = np.asarray(results_dict['metrics'][idx][metric_name]['std'], dtype=float)
    steps = np.asarray(results_dict['metrics'][idx]['step'], dtype=int)
    if steps.size == 0: return np.nan
    order = np.argsort(steps)
    means = means[order]
    stds  = stds[order]
    if initial:
        return float(means[0]) * 100 if means.size else np.nan
    if means.size:
        m, _ = best_over_history(means, stds, metric_name)
        return float(m) * 100
    return np.nan

def render_latex_table_simple(
    results_dict,
    metric_names,
    baseline_idx,
    rows_by_dispersion,
    row_order,
    model_name,
    lora_suffix,
    decimals=1,
    out_path=None
):
    rows = []
    rows.append({"method": "Pretrained (no mid-training)", "disp": "None", "coeff": "N/A", "loc": "-", "src": ("baseline","initial")})
    rows.append({"method": "Default loss", "disp": "None", "coeff": "0.0", "loc": "-", "src": ("baseline","best")})
    for disp in row_order:
        idxs = sorted(rows_by_dispersion[disp], key=lambda i: _num_coeff(results_dict['dispersion_coeff'][i]))
        for i in idxs:
            rows.append({
                "method": results_dict['dispersion'][i],
                "disp":   results_dict['dispersion'][i],
                "coeff":  results_dict['dispersion_coeff'][i],
                "loc":    results_dict['dispersion_loc'][i],
                "idx":    i,
                "src":    ("run","best"),
            })

    ref = {m: _best_or_initial_in_pct(results_dict, baseline_idx, m, initial=True) for m in metric_names}

    # Add an Average column
    align = "l c c " + " ".join(["c"]*len(metric_names)) + " c"
    headers = [m.replace("\n"," ").replace(",", " ") for m in metric_names]
    headers.append("Average")

    lines = []
    lines.append(r"\begin{tabular}{"+align+r"}")
    lines.append(r"\toprule")
    lines.append("Method & Coeff & Loc & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        left = f"{row['method']} & {row['coeff']} & {row['loc']}"
        cells = []
        vals_for_avg = []
        bases_for_avg = []

        for m in metric_names:
            if row['src'] == ("baseline","initial"):
                val = ref[m]
                cell = f"{val:.{decimals}f}" if np.isfinite(val) else "N/A"
                base = ref[m]  # for averaging baseline vs itself; we won't show diff on this row
            elif row['src'] == ("baseline","best"):
                val = _best_or_initial_in_pct(results_dict, baseline_idx, m, initial=False)
                base = ref[m]
                cell = f"{val:.{decimals}f}"
                if np.isfinite(base):
                    diff = val - base
                    sgn  = "+" if diff >= 0 else ""
                    color = "forestgreen" if diff >= 0 else "crimson"
                    cell += f" \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"
            else:
                val = _best_or_initial_in_pct(results_dict, row['idx'], m, initial=False)
                base = ref[m]
                cell = f"{val:.{decimals}f}"
                if np.isfinite(base):
                    diff = val - base
                    sgn  = "+" if diff >= 0 else ""
                    color = "forestgreen" if diff >= 0 else "crimson"
                    cell += f" \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"

            cells.append(cell)
            if np.isfinite(val):
                vals_for_avg.append(val)
            if np.isfinite(base):
                bases_for_avg.append(base)

        # Average column with diff
        if vals_for_avg:
            avg_val = float(np.mean(vals_for_avg))
            if bases_for_avg:
                avg_base = float(np.mean(bases_for_avg))
            else:
                avg_base = np.nan

            if row['src'] == ("baseline","initial") or not np.isfinite(avg_base):
                # Baseline initial row: show value only (no diff), or if base is NaN
                avg_cell = f"{avg_val:.{decimals}f}"
            else:
                avg_cell = f"{avg_val:.{decimals}f}"
                diff = avg_val - avg_base
                sgn  = "+" if diff >= 0 else ""
                color = "forestgreen" if diff >= 0 else "crimson"
                avg_cell += f" \\textcolor{{{color}}}{{({sgn}{diff:.{decimals}f})}}"
        else:
            avg_cell = "N/A"

        cells.append(avg_cell)

        lines.append(left + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_tex = "\n".join(lines)
    if out_path is None:
        os.makedirs("./tables", exist_ok=True)
        out_path = f"./tables/results_table_{model_name}{lora_suffix}.tex"
    with open(out_path, "w") as f:
        f.write(table_tex)

    print("\n===== LaTeX table (copy into Overleaf) =====\n")
    print(table_tex)
    print(f"\n[Saved LaTeX table to: {out_path}]\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Mid-train GPT-2 with a token budget.")
    ap.add_argument("--model_name", type=str, default="gpt2")
    ap.add_argument("--lora", action="store_true", help="Use LoRA (Low-Rank Adaptation) instead of full fine-tuning")
    ap.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                    help="Hugging Face dataset id.")
    args = ap.parse_args()
    lora_suffix = "_lora" if args.lora else ""

    result_folder = './results/'
    figure_lines_save_path = f'./figures/results_lines_{args.model_name}{lora_suffix}.png'
    figure_bars_save_path = f'./figures/results_bars_{args.model_name}{lora_suffix}.png'

    os.makedirs(os.path.dirname(figure_lines_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_bars_save_path), exist_ok=True)
    run_folder_list = sorted(glob(os.path.join(result_folder, f'midtrain_{args.model_name}{lora_suffix}_{"-".join(args.dataset_name.split("/"))}_*')))

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
                        float(data_json['results'][metric_dataset][metric_measure]))
                    std = data_json['results'][metric_dataset][metric_measure.replace(',', '_stderr,')]
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

    metric_ylim_bars = find_metric_ylims(results_dict=results_dict, all_metric_names=all_metric_names, baseline_idx=baseline_idx)

    # enforce the requested row order
    row_order = [d for d in ["Covariance", "Hinge", "InfoNCE_l2", "InfoNCE_cosine"] if d in rows_by_dispersion]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    fig_lines = plt.figure(figsize=(4*len(all_metric_names), 3*len(row_order)))
    fig_bars = plt.figure(figsize=(4*len(all_metric_names), 3*len(row_order)))

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
            baseline_label = 'Default loss'
            # Initial performance before mid-training.
            bars_labels.append('No mid-training')
            bars_heights.append(baseline_means[0])
            bars_colors.append('lightgray')

            # Mid-training with default loss.
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

            ax_lines.set_xlabel("Step", fontsize=12)
            ax_lines.set_ylabel(metric_name, fontsize=12)

            bars = ax_bars.bar(np.arange(len(bars_labels)), bars_heights, color=bars_colors, alpha=0.8, label=bars_labels)
            ax_bars.axhline(y=bars_heights[1], linestyle='--', linewidth=2, color=bars_colors[1], alpha=0.8)
            ax_bars.set_xticks(np.arange(len(bars_labels)))
            ax_bars.set_xticklabels([extract_coeff_from_label(label) for label in bars_labels], rotation=0, ha='center', fontsize=9)
            ax_bars.set_ylabel(metric_name, fontsize=12)
            ax_bars.set_xlabel('Dispersion Coefficient', fontsize=12)
            ax_bars.set_ylim(metric_ylim_bars[metric_name])

            # Annotate the values.
            ax_bars.bar_label(
                bars,
                labels=[f"{v:.3f}" for v in bars_heights],
                rotation=90,
                padding=5,
                fontsize=9,
            )

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

    metrics_for_table = [m for m in all_metric_names if 'perplexity' not in m.lower()]
    render_latex_table_simple(
        results_dict=results_dict,
        metric_names=metrics_for_table,
        baseline_idx=baseline_idx,
        rows_by_dispersion=rows_by_dispersion,
        row_order=row_order,
        model_name=args.model_name,
        lora_suffix=lora_suffix,
        decimals=1,
    )
