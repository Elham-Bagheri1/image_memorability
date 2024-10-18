"""
Feature Analysis Script

This script calculates and evaluates features for images in a dataset. It supports saving the calculated features and evaluating their correlation with memorability scores and IG attributes.

Usage:
- To calculate and save the features: `python feature_analysis.py --calculate-features --output-path ./output`
- To evaluate feature correlations with memorability scores: `python feature_analysis.py --evaluate-features --output-path ./output`
- To perform both calculations and evaluations: `python feature_analysis.py --calculate-features --evaluate-features --output-path ./output`

Arguments:
- --calculate-features: Calculate and save the features.
- --evaluate-features: Evaluate feature correlations with memorability scores.
- --output-path: Specify the path where the output files will be saved.

Example Commands:
- Calculate and save features:
  `python feature_analysis.py --calculate-features --output-path ./output`
- Evaluate feature correlations with memorability scores:
  `python feature_analysis.py --evaluate-features --output-path ./output`
- Perform both calculations and evaluations:
  `python feature_analysis.py --calculate-features --evaluate-features --output-path ./output`
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy.stats import spearmanr
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
from skimage import filters
import torchvision.models as models

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_resnet = models.resnet50(pretrained=True).to(device).eval()
segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()
object_detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()

# Image Transformations
train_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

test_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Prepare data
def prepare_data(path):
    memcat_info = pd.read_csv(os.path.join(path, 'memcat_image_data.csv'))
    split_info = pd.read_csv(os.path.join(path, 'MemCat_split_info.csv'))
    split_info['image'] = split_info['new_path'].str.split('/').str[-1]
    split_info['old_path'] = split_info['old_path'].str.replace('MemCat', 'MemCat_new/MemCat', regex=False)
    split_info = pd.merge(split_info, memcat_info[['image_file', 'memorability_w_fa_correction']], 
                          left_on='image', right_on='image_file', how='left')
    split_info.rename(columns={'memorability_w_fa_correction': 'mem_score'}, inplace=True)
    return split_info

# Create sequence mappings
def create_sequence_mappings(dataframe, sequence_length=2, seed=42):
    np.random.seed(seed)
    all_indices = dataframe.index.tolist()
    np.random.shuffle(all_indices)
    sequences = [all_indices[x:x+sequence_length] for x in range(0, len(all_indices), sequence_length)]
    return sequences

# Dataset class
class MemCatSequenceDataset(Dataset):
    def __init__(self, dataframe, sequences, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_indices = self.sequences[idx]
        images = []
        mem_scores = []

        for seq_idx in sequence_indices:
            if seq_idx < len(self.dataframe):
                img_path = self.dataframe.iloc[seq_idx]['old_path']
                image = Image.open(img_path).convert('RGB')
                mem_score = torch.tensor(self.dataframe.iloc[seq_idx]['mem_score'], dtype=torch.float32)
            else:
                continue

            if self.transform:
                image = self.transform(image)

            images.append(image)
            mem_scores.append(mem_score)

        images = torch.stack(images)
        mem_scores = torch.tensor(mem_scores).float().mean()

        return images, mem_scores

# Feature extraction functions
def extract_color_features(image_cv):
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2Lab)
    color_diversity = shannon_entropy(lab)
    mean_color = np.mean(lab, axis=(0, 1))
    return color_diversity, mean_color

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, energy

def calculate_clutter_and_complexity(image):
    gray_image = rgb2gray(image)
    edges = filters.sobel(gray_image)
    complexity_score = shannon_entropy(edges)
    return complexity_score

def extract_comprehensive_features_with_class_distribution(image_tensor, detection_model):
    detection_model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        predictions = detection_model(image_tensor)[0]

    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()

    num_objects = len(boxes)
    object_sizes = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]

    average_object_size = np.mean(object_sizes) if num_objects > 0 else 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique_labels, counts))

    return {
        'num_objects': num_objects,
        'avg_object_size': average_object_size,
        'num_distinct_classes': len(class_distribution)
    }

# Bootstrapping function for confidence intervals
def bootstrap_spearman_ci(X, y, n_bootstrap=1000):
    correlations = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = np.random.choice(len(y), len(y), replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        correlations[i] = spearmanr(X_sample, y_sample)[0]
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)
    return ci_lower, ci_upper

def calculate_feature_correlations(df_features, target_col, output_path, file_name):
    features = df_features.columns.drop(['Batch_Index', 'Image_Index', target_col])
    results = {'Feature': list(features)}

    spearman_corr = []
    spearman_pvalues = []
    spearman_ci_lower = []
    spearman_ci_upper = []

    for feature in features:
        corr, pvalue = spearmanr(df_features[feature], df_features[target_col])
        ci_lower, ci_upper = bootstrap_spearman_ci(df_features[feature].values, df_features[target_col].values)
        
        spearman_corr.append(corr)
        spearman_pvalues.append(pvalue)
        spearman_ci_lower.append(ci_lower)
        spearman_ci_upper.append(ci_upper)

    results['Spearman Correlation'] = spearman_corr
    results['Spearman p-value'] = spearman_pvalues
    results['Spearman CI Lower'] = spearman_ci_lower
    results['Spearman CI Upper'] = spearman_ci_upper

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_path, file_name), index=False)
    print(f"Feature correlation analysis saved to '{file_name}'")

# Main function
def main():
    parser = argparse.ArgumentParser(description='Feature Analysis on MemCat Dataset')
    parser.add_argument('--calculate-features', action='store_true', help="Calculate and save features")
    parser.add_argument('--evaluate-features', action='store_true', help="Evaluate feature correlations")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save outputs")
    args = parser.parse_args()

    path = os.getcwd()
    output_path = args.output_path

    if args.calculate_features:
        # Prepare data
        split_info = prepare_data(path)

        train_df = split_info[split_info['type'] == 'train']
        val_df = split_info[split_info['type'] == 'val']
        test_df = split_info[split_info['type'] == 'test']

        train_sequences = create_sequence_mappings(train_df)
        val_sequences = create_sequence_mappings(val_df)
        test_sequences = create_sequence_mappings(test_df)

        with open('train_sequences.pkl', 'wb') as f:
            pickle.dump(train_sequences, f)
        with open('val_sequences.pkl', 'wb') as f:
            pickle.dump(val_sequences, f)
        with open('test_sequences.pkl', 'wb') as f:
            pickle.dump(test_sequences, f)

        # train_dataset = MemCatSequenceDataset(train_df, sequences=train_sequences, transform=train_preprocess)
        # val_dataset = MemCatSequenceDataset(val_df, sequences=val_sequences, transform=test_preprocess)
        test_dataset = MemCatSequenceDataset(test_df, sequences=test_sequences, transform=test_preprocess)

        batch_size = 1
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Feature extraction
        df_features = pd.DataFrame()

        for batch_index, (images, _) in enumerate(test_loader):
            images = images.to(device)
            for img_index in range(images.size(0)):
                image_tensor = images[img_index]
                image_pil = transforms.ToPILImage()(image_tensor)
                image_np = np.array(image_pil)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Extract features
                color_features = extract_color_features(image_cv)
                texture_features = extract_texture_features(image_cv)
                clutter_complexity_score = calculate_clutter_and_complexity(image_np)
                comprehensive_features = extract_comprehensive_features_with_class_distribution(image_tensor, object_detection_model)

                features = {
                    'Batch_Index': batch_index,
                    'Image_Index': img_index,
                    'Color_Diversity': color_features[0],
                    'Mean_Color_LAB_L': color_features[1][0],
                    'Mean_Color_LAB_A': color_features[1][1],
                    'Mean_Color_LAB_B': color_features[1][2],
                    'Texture_Contrast': texture_features[0],
                    'Texture_Energy': texture_features[1],
                    'Clutter_Score': clutter_complexity_score,
                    'Num_Objects': comprehensive_features['num_objects'],
                    'Average_Object_Size': comprehensive_features['avg_object_size'],
                    'Num_Distinct_Classes': comprehensive_features['num_distinct_classes']
                }

                df_features = pd.concat([df_features, pd.DataFrame([features])], ignore_index=True)

        df_features.to_csv(os.path.join(output_path, 'extracted_features.csv'), index=False)
        print("Features extracted and saved to extracted_features.csv")

    if args.evaluate_features:
        df_features = pd.read_csv(os.path.join(output_path, 'extracted_features.csv'))
        mem_scores = []

        # Load sequences
        with open('test_sequences.pkl', 'rb') as f:
            test_sequences = pickle.load(f)

        test_dataset = MemCatSequenceDataset(test_df, sequences=test_sequences, transform=test_preprocess)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for _, (_, mem_score) in enumerate(test_loader):
            mem_scores.extend(mem_score.numpy())

        assert len(df_features) == len(mem_scores), "Mismatch between features and memorability scores length."
        df_features['mem_score'] = mem_scores

        # Calculate correlations with memorability scores
        calculate_feature_correlations(df_features, 'mem_score', output_path, 'feature_correlation_with_mem_score.csv')

        # If IG attributes are available, calculate correlations with IG attributes
        if 'Attribution_Value' in df_features.columns:
            calculate_feature_correlations(df_features, 'Attribution_Value', output_path, 'feature_correlation_with_ig_attributes.csv')

if __name__ == "__main__":
    main()
