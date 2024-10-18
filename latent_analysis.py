"""
This script evaluates the distinctiveness of images using the autoencoder model. 
It extracts latent features from images, calculates the normalized distinctiveness, 
and computes the correlation between distinctiveness and memorability scores. 
The results are saved to CSV files.

Usage:
- To calculate latent representations: 
  python latent_analysis.py --calculate-latents --output-path ./output
- To perform distinctiveness analysis:
  python latent_analysis.py --distinctiveness-analysis --output-path ./output
- To perform both calculations and evaluations:
  python latent_analysis.py --calculate-latents --distinctiveness-analysis --output-path ./output

The following directory structure is assumed:
MemCat/
    ├── animal/
    ├── vehicle/
    ├── landscape/
    ├── sports
    └── food/
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from imagenet_autoencoder_main.models import vgg


def prepare_data(path):
    """Prepares the data by merging memorability scores with image paths."""
    memcat_info = pd.read_csv(os.path.join(path, 'memcat_image_data.csv'))
    split_info = pd.read_csv(os.path.join(path, 'MemCat_split_info.csv'))
    train_images = split_info.copy()
    train_images.reset_index(inplace=True, drop=True)
    train_images['image'] = train_images['new_path'].str.split('/').str[-1]
    train_images = pd.merge(train_images, memcat_info[['image_file', 'memorability_w_fa_correction', 'memorability_wo_fa_correction']],
                            left_on='image', right_on='image_file', how='left')
    train_images.rename(columns={'memorability_w_fa_correction': 'mem score',
                                 'memorability_wo_fa_correction': 'mem score wo'}, inplace=True)
    return train_images

def extract_latent_features(model, data_loader, latent_dir, model_name):
    """Extracts latent features from the images using the model and saves them as .npy files."""
    model.eval()
    os.makedirs(latent_dir, exist_ok=True)
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.view(outputs.size(0), -1).cpu().detach().numpy()
        image_name = os.path.basename(data_loader.dataset.imgs[i][0]).replace('.', '_')
        output_filename = f'latent_{image_name}_{model_name}.npy'
        np.save(os.path.join(latent_dir, output_filename), outputs)

def load_latent_features(latent_dir, data_loader, model_name):
    """Loads the latent features from the saved .npy files."""
    all_outputs = []
    all_filenames = []
    for i in range(len(data_loader.dataset)):
        filename = data_loader.dataset.imgs[i][0]
        image_name = os.path.basename(filename).replace(".", "_")
        output_filename = f'latent_{image_name}_{model_name}.npy'
        output = np.load(os.path.join(latent_dir, output_filename)).reshape(1, -1)
        all_outputs.append(output)
        all_filenames.append(filename)
    return np.vstack(all_outputs), all_filenames

def calculate_distinctiveness(latent_features):
    """Calculates the normalized distinctiveness of the images."""
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(latent_features)
    distances, indices = nbrs.kneighbors(latent_features)
    nearest_neighbors_distances = distances[:, 1]
    normalized_distances = zscore(nearest_neighbors_distances)
    return normalized_distances

def save_distinctiveness_metrics(output_path, model_name, filenames, normalized_distances):
    """Saves the distinctiveness metrics to a CSV file."""
    df = pd.DataFrame({
        'ImageName': filenames,
        'Normalized Distinctiveness': normalized_distances
    })
    df.to_csv(os.path.join(output_path, f'distinctiveness_metrics_{model_name}.csv'), index=False)

def calculate_spearmanr_with_ci(df1, df2, n_bootstraps=1000):
    """Calculates the Spearman correlation with confidence intervals using bootstrapping."""
    df1['image'] = df1['ImageName'].apply(lambda x: x.split('/')[-1])
    merged_df = pd.merge(df1, df2, on='image')
    
    # Calculate Spearman correlation and p-value
    correlation, p_value = spearmanr(merged_df['mem score'], merged_df['Normalized Distinctiveness'])
    
    # Bootstrap for confidence intervals
    correlations = [spearmanr(resample(merged_df)['mem score'], resample(merged_df)['Normalized Distinctiveness'])[0] for _ in range(n_bootstraps)]
    correlation_mean = np.mean(correlations)
    confidence_lower = np.percentile(correlations, 2.5)
    confidence_upper = np.percentile(correlations, 97.5)
    
    return correlation, p_value, correlation_mean, confidence_lower, confidence_upper

# Main function
def main():
    parser = argparse.ArgumentParser(description='Latent Feature Extraction and Distinctiveness Analysis')
    parser.add_argument('--calculate-latents', action='store_true', help='Calculate and save latent representations')
    parser.add_argument('--distinctiveness-analysis', action='store_true', help='Perform distinctiveness analysis')
    parser.add_argument('--output-path', type=str, default='./output', help='Path to save outputs')
    args = parser.parse_args()

    path = os.getcwd()
    # model_name = 'mem_vgg_autoencoder_statedict.pth' # MemCat_Autoencoder_vgg16_lr1e-05_bs1_LossFunclpips-sq_epoch1_statedict.pth'
    model_name ='MemCat_Autoencoder_vgg16_lr1e-05_bs1_LossFunclpips-sq_epoch1_statedict.pth'
    model_path = os.path.join(path, model_name)
    latent_dir = os.path.join(path, 'latent')
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    # Load data
    train_images = prepare_data(path)
    test_dir = os.path.join(path, 'MemCat')
    preprocess = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    test_data = ImageFolder(test_dir, transform=preprocess)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Load model
    # Load model
    state_dict = torch.load(model_path)
    configs = vgg.get_configs()
    model = vgg.VGGEncoder(configs, enable_bn=True)
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    model.load_state_dict(encoder_state_dict)
    model = model.to(device)

    if args.calculate_latents:
        # Extract latent features
        extract_latent_features(model, test_loader, latent_dir, model_name)
        print(f"Latent features extracted and saved to {latent_dir}")

    if args.distinctiveness_analysis:
        # Load latent features and calculate distinctiveness
        latent_features, filenames = load_latent_features(latent_dir, test_loader, model_name)
        normalized_distances = calculate_distinctiveness(latent_features)
        save_distinctiveness_metrics(output_path, model_name, filenames, normalized_distances)
        print(f"Distinctiveness metrics calculated and saved to {output_path}")

        # Calculate correlation and confidence intervals
        df1 = pd.read_csv(os.path.join(output_path, f'distinctiveness_metrics_{model_name}.csv'))
        correlation, p_value, correlation_mean, confidence_lower, confidence_upper = calculate_spearmanr_with_ci(df1, train_images)
        print(f"Spearman correlation: {correlation}")
        print(f"p-value: {p_value}")
        print(f"Mean correlation (bootstrapped): {correlation_mean}")
        print(f"95% confidence interval: ({confidence_lower}, {confidence_upper})")

        # Save correlation results
        results = {
            'Spearman Correlation': [correlation],
            'p-value': [p_value],
            'Mean Correlation (bootstrapped)': [correlation_mean],
            '95% CI Lower': [confidence_lower],
            '95% CI Upper': [confidence_upper]
        }
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_path, f'distinctiveness_results_{model_name}.csv'), index=False)
        print(f"Correlation results saved to {output_path}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
