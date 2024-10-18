"""
Author: Elham Bagheri
Date: October 2024

This script analyzes the correlation between memorability scores and reconstruction loss using
various models trained on the MemCat dataset. The script includes helper functions for retrieving
model filenames, loading dataframes, calculating Spearman correlations with confidence intervals,
and processing multiple files to extract relevant details.

Usage:
- To run the analysis for all models and save the results:
  python autoencoder_evaluation.py --output-path ./output
- To run the analysis for specific models and save the results:
  python autoencoder_evaluation.py --output-path ./output --model-names model1.pth model2.pth
  example: autoencoder_evaluation.py --output-path ./output --model-names mem_vgg_autoencoder.pth
"""

import os
import re
import numpy as np
import pandas as pd
import argparse
from scipy.stats import spearmanr
from sklearn.utils import resample


# Helper Functions
def get_model_filenames(directory, pattern):
    """
    Retrieve model filenames matching the given regex pattern.
    
    :param directory: Path to the directory containing model files
    :param pattern: Regex pattern to match model filenames
    :return: List of filenames that match the pattern
    """
    return [f for f in os.listdir(directory) if re.search(pattern, f)]

def load_dataframe(filepath):
    """
    Load a CSV file into a DataFrame.
    
    :param filepath: Path to the CSV file
    :return: Loaded DataFrame
    """
    return pd.read_csv(filepath)

def calculate_spearmanr_with_ci(data, n_bootstraps=1000):
    """
    Calculate Spearman correlation with confidence intervals using bootstrapping.
    
    :param data: DataFrame containing 'Loss' and 'Score' columns
    :param n_bootstraps: Number of bootstrap samples
    :return: Tuple containing mean correlation, lower confidence interval, and upper confidence interval
    """
    correlations = []
    for _ in range(n_bootstraps):
        sample = resample(data)
        corr, _ = spearmanr(sample['Loss'], sample['Score'])
        correlations.append(corr)
    correlations = np.array(correlations)
    correlation_mean = np.mean(correlations)
    confidence_lower = np.percentile(correlations, 2.5)
    confidence_upper = np.percentile(correlations, 97.5)
    return correlation_mean, confidence_lower, confidence_upper

def extract_details(match):
    """
    Extract details from the regex match.
    
    :param match: Regex match object
    :return: Dictionary with extracted details
    """
    return {
        "lr": match.group("lr"),
        "batch_size": match.group("batch_size"),
        "epoch": int(match.group("epoch")) + 1,
        "loss_type": match.group("loss_type"),
    }

def process_files(directory_path, pattern, model_names=None):
    """
    Process multiple files to extract relevant details and calculate correlations.
    
    :param directory_path: Path to the directory containing model and CSV files
    :param pattern: Regex pattern to match model filenames
    :param model_names: List of specific model filenames to process (optional)
    :return: DataFrame containing the extracted details and correlation results
    """
    details_list = []
    if model_names:
        model_filenames = [f for f in model_names if re.search(pattern, f)]
    else:
        model_filenames = get_model_filenames(directory_path, pattern)

    for model_filename in model_filenames:
        match = re.search(pattern, model_filename)
        if match:
            print(model_filename)
            details = extract_details(match)
            details['model'] = model_filename
            
            test_filename = f'MemCat_{model_filename.replace("_statedict.pth", "")}_testLoss.csv'
            df = load_dataframe(os.path.join(directory_path, test_filename))
            
            corr, p_value = spearmanr(df['Loss'], df['Score'])
            correlation_mean, confidence_lower, confidence_upper = calculate_spearmanr_with_ci(df[['Loss', 'Score']])
            
            details.update({
                'correlation': corr,
                'p_value': p_value,
                'correlation_mean_boot': correlation_mean,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper,
                'loss_value_mean': df['Loss'].mean(),
                'loss_value_std': df['Loss'].std(),
                'loss_value_median': df['Loss'].median()
            })
            
            details_list.append(details)

    return pd.DataFrame(details_list)

def main():
    PATTERN = r'MemCat_Autoencoder_vgg16_lr(?P<lr>[\d\w.-]+)_bs(?P<batch_size>\d+)_LossFunc(?P<loss_type>[\w-]+)_epoch(?P<epoch>\d+)_statedict\.pth'

    parser = argparse.ArgumentParser(description='Correlation Analysis between Memorability Scores and Reconstruction Loss')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output files')
    parser.add_argument('--model-names', type=str, nargs='*', help='Specific model filenames to process (optional)')
    args = parser.parse_args()

    df_sorted = process_files(args.output_path, PATTERN, args.model_names)
    df_sorted = df_sorted.sort_values(by='correlation', ascending=False)
    df_sorted.to_csv(os.path.join(args.output_path, 'loss_correlations_memcat_test.csv'), index=False)
    print(df_sorted)

if __name__ == "__main__":
    main()
