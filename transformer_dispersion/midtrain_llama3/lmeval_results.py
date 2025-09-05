import json
import pandas as pd
import numpy as np
import os

def parse_lm_eval_json_to_dataframe(json_file_path):
    """
    Convert LM evaluation JSON results to a pandas DataFrame for easier analysis.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        tuple: (main_df, detailed_df)
            - main_df: DataFrame with main metrics (aggregated scores)
            - detailed_df: DataFrame with all sub-metrics
    """
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Lists to store parsed data
    main_metrics = []
    detailed_metrics = []
    
    for task_name, task_data in results.items():
        # Extract alias for cleaner names
        alias = task_data.get('alias', task_name)
        
        # Parse all metrics for this task
        for key, value in task_data.items():
            if key == 'alias':
                continue
                
            # Split metric name and aggregation type
            if ',none' in key:
                metric_name = key.replace(',none', '')
                is_stderr = '_stderr' in metric_name
                base_metric = metric_name.replace('_stderr', '') if is_stderr else metric_name
                
                # Determine if this is a main task or sub-task
                is_main_task = not task_name.startswith(('mmlu_', 'paloma_'))
                
                # Create record
                record = {
                    'task': task_name,
                    'alias': alias,
                    'metric': base_metric,
                    'value': value if value != 'N/A' else np.nan,
                    'is_stderr': is_stderr,
                    'is_main_task': is_main_task
                }
                
                if is_main_task:
                    main_metrics.append(record)
                
                detailed_metrics.append(record)
    
    # Create DataFrames
    main_df = pd.DataFrame(main_metrics)
    detailed_df = pd.DataFrame(detailed_metrics)
    
    # Pivot to get a cleaner structure for main metrics
    if not main_df.empty:
        main_pivot = main_df.pivot_table(
            index=['task', 'alias'], 
            columns=['metric', 'is_stderr'], 
            values='value', 
            aggfunc='first'
        )
        main_pivot = main_pivot.fillna(np.nan)
    else:
        main_pivot = pd.DataFrame()
    
    return main_df, detailed_df, main_pivot

def create_simple_summary(main_df):
    """Create a simple summary with just task and main metric value."""
    
    if main_df.empty:
        return pd.DataFrame()
    
    # Get only values (not stderr)
    values_df = main_df[~main_df['is_stderr']].copy()
    
    summary_data = []
    
    for _, row in values_df.iterrows():
        task = row['task']
        metric = row['metric']
        value = row['value']
        
        # Use acc if available, otherwise use word_perplexity
        if metric == 'acc' or (metric == 'word_perplexity' and task not in [r['task'] for r in summary_data if r.get('metric') == 'acc']):
            summary_data.append({
                'task': row['alias'],
                'metric': metric,
                'value': value
            })
    
    return pd.DataFrame(summary_data)

# Load and convert the JSON file
json_root_path = '../../external_src/LLaMA-Factory/saves/llama/full/midtrain_wikitext_lr1e-4/'
json_files = ['lm_eval_begin_0.json', 'lm_eval_interval_100.json', 'lm_eval_end_165.json']

# Simple combined results
all_results = []

for json_file in json_files:
    json_file_path = os.path.join(json_root_path, json_file)
    
    # Extract step from filename
    if 'begin' in json_file:
        step = 0
    elif 'interval' in json_file:
        step = int(json_file.split('_')[3].split('.')[0])  # lm_eval_interval_100.json
    elif 'end' in json_file:
        step = int(json_file.split('_')[3].split('.')[0])  # lm_eval_end_165.json
    else:
        step = -1
    
    main_df, _, _ = parse_lm_eval_json_to_dataframe(json_file_path)
    summary_table = create_simple_summary(main_df)
    summary_table['step'] = step
    
    all_results.append(summary_table)

# Combine all results
combined_df = pd.concat(all_results, ignore_index=True)

# Create wide table: tasks as rows, steps as columns
wide_df = combined_df.pivot_table(
    index='task', 
    columns='step', 
    values='value', 
    aggfunc='first'
)

# Clean up column names
wide_df.columns = [f'step_{int(col)}' for col in wide_df.columns]
wide_df = wide_df.reset_index()

# Save the wide format
wide_df.to_csv('combined_results_wide.csv', index=False)

print(f"Created wide table with {len(wide_df)} tasks across {len(wide_df.columns)-1} steps")
print(f"Saved to: combined_results_wide.csv")

# Also create detailed wide table with all sub-tasks
detailed_results = []

for json_file in json_files:
    json_file_path = os.path.join(json_root_path, json_file)
    
    # Extract step from filename
    if 'begin' in json_file:
        step = 0
    elif 'interval' in json_file:
        step = int(json_file.split('_')[3].split('.')[0])
    elif 'end' in json_file:
        step = int(json_file.split('_')[3].split('.')[0])
    else:
        step = -1
    
    _, detailed_df, _ = parse_lm_eval_json_to_dataframe(json_file_path)
    
    # Filter for main metrics (acc or word_perplexity) and not stderr
    detailed_filtered = detailed_df[
        (~detailed_df['is_stderr']) & 
        ((detailed_df['metric'] == 'acc') | (detailed_df['metric'] == 'word_perplexity'))
    ].copy()
    
    detailed_filtered['step'] = step
    detailed_results.append(detailed_filtered[['alias', 'metric', 'value', 'step']])

# Combine detailed results
detailed_combined = pd.concat(detailed_results, ignore_index=True)

# Clean up alias names - remove "- " prefix
detailed_combined['alias_clean'] = detailed_combined['alias'].str.replace(r'^[\s\-]+', '', regex=True)

# Create detailed wide table
detailed_wide = detailed_combined.pivot_table(
    index='alias_clean',
    columns='step', 
    values='value',
    aggfunc='first'
)

# Clean up column names
detailed_wide.columns = [f'step_{int(col)}' for col in detailed_wide.columns]
detailed_wide = detailed_wide.reset_index()

# Add comparison column (step_165 - step_0)
if 'step_0' in detailed_wide.columns and 'step_165' in detailed_wide.columns:
    detailed_wide['change_0_to_165'] = detailed_wide['step_165'] - detailed_wide['step_0']

# Save detailed wide format
detailed_wide.to_csv('combined_results_detailed_wide.csv', index=False)

print(f"Created detailed wide table with {len(detailed_wide)} tasks")
print(f"Saved to: combined_results_detailed_wide.csv")
print(f"Added change column showing step_165 - step_0")
