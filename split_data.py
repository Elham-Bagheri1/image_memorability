"""
This script splits a dataset into training, validation, and test sets based on the memorability scores.
It creates a structured directory for the splits and saves the information about the splits in a CSV file.
The split ratio is configurable, and the paths are set to be relative for easy integration into any project.

Directory structure:
MemCat_splits/
    ├── train/
    ├── val/
    └── test/

The CSV file will contain the old and new paths, category, subcategory, type of split, memorability category, and the score.
"""

import os
import shutil
import numpy as np
import pandas as pd

# Set your paths here
path = os.getcwd()
dataset_path = os.path.join(path, 'MemCat')  # Replace with your dataset path
output_path = os.path.join(path, "MemCat_splits") #modify if saving different data splits with different seeds
scores_df = pd.read_csv(os.path.join(path, 'memcat_image_data.csv'))

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)

def split_dataset(scores_df, dataset_path, output_path, train_ratio=0.6, val_ratio=0.1, seed=42): # you can choose different seeds to save to save different data splits
    """
    Split the dataset into train, validation, and test sets and save the split information.

    :param scores_df: DataFrame containing scores and image file paths
    :param dataset_path: Path to the original dataset
    :param output_path: Path to save the split dataset
    :param train_ratio: Ratio of training data
    :param val_ratio: Ratio of validation data
    :param seed: Random seed for reproducibility
    :return: DataFrame containing split information
    """
    np.random.seed(seed)
    median_score = np.median(scores_df['memorability_w_fa_correction'].values)

    # Assign high/low memorability based on the median score
    scores_df['memorability_category'] = scores_df['memorability_w_fa_correction'].apply(
        lambda x: 'high' if x >= median_score else 'low'
    )

    # Prepare the path information dictionary
    path_info = {
        "old_path": [],
        "new_path": [],
        "category": [],
        "subcategory": [],
        "type": [],
        "memorability_category": [],
        "score": []
    }

    # Split the data
    for category in scores_df['category'].unique():
        for subcategory in scores_df[scores_df['category'] == category]['subcategory'].unique():
            for memorability_category in ['high', 'low']:
                relevant_files = scores_df[
                    (scores_df['category'] == category) &
                    (scores_df['subcategory'] == subcategory) &
                    (scores_df['memorability_category'] == memorability_category)
                ]['image_file'].tolist()

                np.random.shuffle(relevant_files)
                train_files, val_test_files = np.split(relevant_files, [int(len(relevant_files) * train_ratio)])
                val_files, test_files = np.split(val_test_files, [int(len(val_test_files) * val_ratio / (1 - train_ratio))])

                # Copy files and populate path information
                for file_name, files, dtype in zip(
                        ['train', 'val', 'test'],
                        [train_files, val_files, test_files],
                        ['train', 'val', 'test']):
                    output_folder = os.path.join(output_path, dtype, category, subcategory)
                    os.makedirs(output_folder, exist_ok=True)

                    for f in files:
                        old_path = os.path.join(dataset_path, category, subcategory, f)
                        new_path = os.path.join(output_folder, f)
                        shutil.copyfile(old_path, new_path)

                        path_info["old_path"].append(old_path)
                        path_info["new_path"].append(new_path)
                        path_info["category"].append(category)
                        path_info["subcategory"].append(subcategory)
                        path_info["type"].append(dtype)
                        path_info["memorability_category"].append(memorability_category)
                        path_info["score"].append(
                            scores_df.loc[scores_df['image_file'] == f, 'memorability_w_fa_correction'].values[0]
                        )

    # Create the dataframe
    path_df = pd.DataFrame(path_info)
    return path_df

if __name__ == "__main__":
    # Call the function and save the result to CSV
    path_df = split_dataset(scores_df, dataset_path, output_path)
    path_df.to_csv(os.path.join(output_path, "MemCat_split_info.csv"), index=False)

    # Calculating the mean scores for train, val, and test sets
    mean_scores = path_df.groupby('type')['score'].mean()
    print("Mean Memorability Score for Training Set:", mean_scores.get('train', np.nan))
    print("Mean Memorability Score for Validation Set:", mean_scores.get('val', np.nan))
    print("Mean Memorability Score for Test Set:", mean_scores.get('test', np.nan))
