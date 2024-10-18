"""
This script performs Integrated Gradients analysis on a sequence model and visualizes the attributions.
It allows saving and/or displaying the attributions for each image in the test dataset.

Usage:
- To save the attributions: python model_interpretation_IG.py --save
- To display the attributions: model_interpretation_IG.py --display
- To do both: python model_interpretation_IG.py --save --display

Prerequisites:
- The script expects the 'MemCat' dataset and its split information in the current working directory.
- The pretrained VGG autoencoder model should be available in the specified path.
- The GRU model should be trained and saved as 'best_gru_model.pth' or another specified model.

Modules:
1. Image Transformations
2. Data Preparation
3. Dataset Definition
4. GRU Regressor Definition
5. Saving and Displaying Attributions
6. Main Execution Function
"""

import os
import argparse
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from captum.attr import IntegratedGradients
from imagenet_autoencoder_main.models import vgg
from mem_gru import MemGRU

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
    """
    Prepares the dataset by merging memorability scores with split information.

    Args:
    - path (str): Path to the directory containing dataset information.

    Returns:
    - pd.DataFrame: DataFrame containing the merged dataset.
    """
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
    """
    Creates sequence mappings for the dataset.

    Args:
    - dataframe (pd.DataFrame): Input DataFrame.
    - sequence_length (int): Length of the sequences.
    - seed (int): Random seed for shuffling.

    Returns:
    - list: List of sequences.
    """
    np.random.seed(seed)
    all_indices = dataframe.index.tolist()
    np.random.shuffle(all_indices)
    sequences = [all_indices[x:x+sequence_length] for x in range(0, len(all_indices), sequence_length)]
    return sequences

# Dataset class
class MemCatSequenceDataset(Dataset):
    def __init__(self, dataframe, sequences, transform=None):
        """
        Initializes the dataset.

        Args:
        - dataframe (pd.DataFrame): DataFrame containing dataset information.
        - sequences (list): List of sequences.
        - transform (callable, optional): Transform to be applied on an image.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
        - idx (int): Index of the item to retrieve.

        Returns:
        - tuple: (images, mem_scores)
        """
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

# Function to save and display image with attributions
def save_and_display_image(tensor, attribution, mem_score, filename, display=False):
    """
    Saves and optionally displays an image with its attributions.

    Args:
    - tensor (torch.Tensor): Image tensor.
    - attribution (torch.Tensor): Attribution tensor.
    - mem_score (float): Memorability score of the image.
    - filename (str): Path to save the image.
    - display (bool): Whether to display the image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    image = ToPILImage()(tensor.cpu()).convert("RGB")
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\nMemorability Score: {mem_score:.2f}')
    axes[0].axis('off')
    
    # Attribution map
    attribution_map = attribution.abs().sum(dim=0).squeeze().cpu().numpy()
    if attribution_map.ndim == 3:
        attribution_map = np.mean(attribution_map, axis=0)  # Convert to 2D
    im = axes[1].imshow(attribution_map, cmap='Reds')
    axes[1].set_title('Attribution Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])

    plt.savefig(filename)
    if display:
        plt.show()
    plt.close()

# Function to process and save attributions
def process_and_save_attributions(images, attributions, mem_scores, batch_index, output_path, display=False):
    """
    Processes and saves attributions for a batch of images.

    Args:
    - images (torch.Tensor): Batch of images.
    - attributions (torch.Tensor): Batch of attributions.
    - mem_scores (torch.Tensor): Memorability scores for the batch.
    - batch_index (int): Index of the batch.
    - output_path (str): Path to save the images.
    - display (bool): Whether to display the images.
    """
    batch_size, seq_len, C, H, W = images.size()
    for i in range(batch_size):
        for j in range(seq_len):
            img_tensor = images[i, j]
            attribution = attributions[i, j]
            mem_score = mem_scores.item()
            combined_image_path = os.path.join(output_path, f'combined_image_attributions_batch_{batch_index}_seq_{j}.png')
            save_and_display_image(img_tensor, attribution, mem_score, combined_image_path, display)
        
    def forward(self, x):
        """
        Forward pass for the GRU regressor.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        if len(x.size()) == 5:
            batch_size, seq_len, C, H, W = x.size()
            x = x.view(batch_size * seq_len, C, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {x.size()}")

        encoded_images = self.encoder(x)
        encoded_images = torch.flatten(encoded_images, start_dim=1)
        encoded_images = encoded_images.view(batch_size, seq_len, -1)
        gru_out, _ = self.gru(encoded_images)
        gru_out = gru_out[:, -1, :]
        output = self.fc(gru_out)
        return output

# Main function
def main():
    parser = argparse.ArgumentParser(description='Run Integrated Gradients Analysis')
    parser.add_argument('--save', action='store_true', help='Save the attributions')
    parser.add_argument('--display', action='store_true', help='Display the attributions')
    parser.add_argument('--model_path', type=str, default="mem_gru_statedict.pth", help='Path to the GRU model')
    args =parser.parse_args()

    path = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    # model_name = 'mem_vgg_autoencoder_statedict.pth'
    model_name ='MemCat_Autoencoder_vgg16_lr1e-05_bs1_LossFunclpips-sq_epoch1_statedict.pth'
    model_path = os.path.join(path, model_name)
    

    state_dict = torch.load(model_path)
    configs = vgg.get_configs()
    vgg_encoder = vgg.VGGEncoder(configs, enable_bn=True)
    encoder_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if 'encoder.' in k}
    vgg_encoder.load_state_dict(encoder_state_dict)
        
    vgg_encoder = vgg_encoder.to(device)
    
    # Prepare the data
    split_info = prepare_data(path)

    # train_df = split_info[split_info['type'] == 'train']
    # val_df = split_info[split_info['type'] == 'val']
    test_df = split_info[split_info['type'] == 'test']

    # train_sequences = create_sequence_mappings(train_df)
    # val_sequences = create_sequence_mappings(val_df)
    test_sequences = create_sequence_mappings(test_df)

    test_dataset = MemCatSequenceDataset(test_df, sequences=test_sequences, transform=test_preprocess)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the GRU model
    best_hyperparams = {
        'hidden_layer_size': 256,  # Use the best hyperparameters from previous experiments, or another set of hyperparameters 
        'num_layers': 2,
        'bidirectional': True,
        'dropout': 0.5
    }
    model = MemGRU(encoder=vgg_encoder, input_size=512*7*7, hidden_dim=best_hyperparams['hidden_layer_size'], 
                           num_layers=best_hyperparams['num_layers'], bidirectional=best_hyperparams['bidirectional'], 
                           dropout=best_hyperparams['dropout']).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Initialize the IntegratedGradients object
    integrated_gradients = IntegratedGradients(model)

    output_path = './integrated_gradients_output/'
    os.makedirs(output_path, exist_ok=True)

    # Process all test images
    for batch_index, (images, mem_scores) in enumerate(test_loader):
        images = images.to(device)
        mem_scores = mem_scores.to(device)
        baseline = torch.zeros_like(images)

        # Compute attributions
        attributions = integrated_gradients.attribute(images, baselines=baseline, target=0, n_steps=50)

        # Process and save attributions and original images
        process_and_save_attributions(images, attributions, mem_scores, batch_index, output_path, display=args.display)

if __name__ == "__main__":
    main()
