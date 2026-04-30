from typing import List
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import re


def string_to_arr(string_list: List[str]) -> np.ndarray:
    array_list = []
    for string in string_list:
        string = string.strip("'[]\n")
        array_list.append(np.fromstring(string, sep=' '))
    return np.array(array_list)

def extract_series(model_name):
    parts = model_name.split('/')
    if len(parts) == 1:
        return parts[0]
    prefix = parts[0]
    # Match the first part of the model name after the slash that is not purely numeric or hyphenated
    match = re.search(r'([a-zA-Z]+)', parts[1])
    suffix = match.group(1).lower() if match else ''
    return f'{prefix}/{suffix}' if suffix else prefix


if __name__ == '__main__':

    df = pd.read_csv('condensation_vs_performance.csv')
    df = df[df['Type'] == 'pretrained']

    performance_measure = 'open_llm_average'

    fig = plt.figure(figsize=(16, 8))
    for metric_idx, metric in enumerate(['DSE', 'VNE']):
        for dataset_idx, dataset in enumerate(df.dataset.unique()):
            df_subset = df[df['dataset'] == dataset]
            ax = fig.add_subplot(2, 4, metric_idx * 4 + dataset_idx + 1)
            if metric == 'DSE':
                for model_name in df_subset['huggingface_ID'].unique():
                    df_curr_model = df_subset[df_subset['huggingface_ID'] == model_name]
                    DSE_arr = string_to_arr(df_curr_model['DSE_arr'])
                    DSE_arr_mean = DSE_arr.mean(axis=0)
                    DSE_arr_std = DSE_arr.mean(axis=0)
                    ax.scatter(np.repeat(df_curr_model[performance_measure].astype(float).mean(), len(DSE_arr_mean)),
                               DSE_arr_mean,
                               s=df_curr_model['parameters (B)'].mean() * 5,
                               c=np.linspace(0.2, 1, len(DSE_arr_mean)),
                               cmap='Reds')
                    ax.plot(np.repeat(df_curr_model[performance_measure].astype(float).mean(), len(DSE_arr_mean)),
                            DSE_arr_mean,
                            color='black', alpha=0.2)
            else:
                for model_name in df_subset['huggingface_ID'].unique():
                    df_curr_model = df_subset[df_subset['huggingface_ID'] == model_name]
                    VNE_arr = string_to_arr(df_curr_model['VNE_arr'])
                    VNE_arr_mean = VNE_arr.mean(axis=0)
                    VNE_arr_std = VNE_arr.mean(axis=0)
                    ax.scatter(np.repeat(df_curr_model[performance_measure].astype(float).mean(), len(VNE_arr_mean)),
                               VNE_arr_mean,
                               s=df_curr_model['parameters (B)'].mean() * 5,
                               c=np.linspace(0.2, 1, len(VNE_arr_mean)),
                               cmap='Reds')
                    ax.plot(np.repeat(df_curr_model[performance_measure].astype(float).mean(), len(VNE_arr_mean)),
                            VNE_arr_mean,
                            color='black', alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel(performance_measure)
            ax.set_ylabel(metric)
            ax.set_title(dataset)

    fig.tight_layout(pad=2)
    fig.savefig('motivation1.png')

    fig = plt.figure(figsize=(16, 8))
    for metric_idx, metric in enumerate(['DSE', 'VNE']):
        for dataset_idx, dataset in enumerate(df.dataset.unique()):
            df_subset = df[df['dataset'] == dataset]
            ax = fig.add_subplot(2, 4, metric_idx * 4 + dataset_idx + 1)
            if metric == 'DSE':
                df_subset_averaged_over_seed = df_subset[['huggingface_ID', 'parameters (B)', performance_measure, 'final_layer_DSE']].groupby(['huggingface_ID']).mean().reset_index()
                df_subset_averaged_over_seed['series'] = df_subset_averaged_over_seed['huggingface_ID'].apply(extract_series)

                ax.scatter(df_subset_averaged_over_seed[performance_measure], df_subset_averaged_over_seed['final_layer_DSE'],
                           s=df_subset_averaged_over_seed['parameters (B)'] * 10,
                           color='firebrick', alpha=0.5)

                for name, group in df_subset_averaged_over_seed.groupby('series'):
                    if len(group) > 1:
                        sorted_group = group.sort_values(performance_measure)
                        ax.plot(
                            sorted_group[performance_measure],
                            sorted_group['final_layer_DSE'],
                            linestyle='-', marker='',
                            color='black', alpha=0.5,
                        )
            else:
                df_subset_averaged_over_seed = df_subset[['huggingface_ID', 'parameters (B)', performance_measure, 'final_layer_VNE']].groupby(['huggingface_ID']).mean().reset_index()
                df_subset_averaged_over_seed['series'] = df_subset_averaged_over_seed['huggingface_ID'].apply(extract_series)
                ax.scatter(df_subset_averaged_over_seed[performance_measure], df_subset_averaged_over_seed['final_layer_VNE'],
                           s=df_subset_averaged_over_seed['parameters (B)'] * 10,
                           color='firebrick', alpha=0.5)

                for name, group in df_subset_averaged_over_seed.groupby('series'):
                    if len(group) > 1:
                        sorted_group = group.sort_values(performance_measure)
                        ax.plot(
                            sorted_group[performance_measure],
                            sorted_group['final_layer_VNE'],
                            linestyle='-', marker='',
                            color='black', alpha=0.5,
                        )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel(performance_measure)
            ax.set_ylabel(metric)
            ax.set_title(dataset)

    fig.tight_layout(pad=2)
    fig.savefig('motivation2.png')
