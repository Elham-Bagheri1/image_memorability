# train_evaluate_gru.py

"""
This script trains and evaluates a GRU model for image memorability prediction using the VGG autoencoder as an encoder.
The script supports training and evaluation modes with options to save and load models.

Usage:
To train the model:
    python gru_training_evaluation.py --train
To evaluate the model:
    python gru_training_evaluation.py --evaluate
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imagenet_autoencoder_main.models import vgg
from mem_gru import MemGRU
import pickle
import copy

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

# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, early_stopping_patience, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, mem_scores in train_loader:
            inputs, mem_scores = inputs.to(device), mem_scores.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), mem_scores.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, mem_scores in val_loader:
                inputs, mem_scores = inputs.to(device), mem_scores.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1), mem_scores.view(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print('Early stopping triggered')
                break

    model.load_state_dict(best_model_wts)
    return model, best_loss

# Bootstrapping function for confidence intervals
def bootstrap_metric(metric_func, y_true, y_scores, n_bootstrap=1000):
    bootstrapped_scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_sample = np.array(y_true)[indices]
        y_scores_sample = np.array(y_scores)[indices]
        bootstrapped_scores.append(metric_func(y_true_sample, y_scores_sample))
    return np.percentile(bootstrapped_scores, [2.5, 97.5])

# Function to evaluate model performance and calculate metrics
def evaluate_model_performance(model, data_loader, device, median_mem_score):
    model.eval()
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().numpy().ravel()
            y_scores.extend(probabilities)
            actual_scores = labels.cpu().numpy().ravel()
            y_true.extend(actual_scores)

    # Calculate Spearman's rank correlation
    spearman_corr, p_val = spearmanr(y_true, y_scores)
    spearman_ci_lower, spearman_ci_upper = bootstrap_metric(lambda y_true, y_scores: spearmanr(y_true, y_scores)[0], y_true, y_scores)
    print(f"Spearman's rank correlation: {spearman_corr} with CI: [{spearman_ci_lower}, {spearman_ci_upper}] p_value: {p_val}")

    # Convert actual scores to binary labels based on median_mem_score for further metrics
    binary_labels = (np.array(y_true) > median_mem_score).astype(int)
    
    # AUC and other metrics calculations using binary_labels
    fpr, tpr, thresholds_roc = roc_curve(binary_labels, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_auc_ci_lower, roc_auc_ci_upper = bootstrap_metric(lambda y_true, y_scores: auc(roc_curve(y_true, y_scores)[0], roc_curve(y_true, y_scores)[1]), binary_labels, y_scores)
    
    pr_auc = average_precision_score(binary_labels, y_scores)
    pr_auc_ci_lower, pr_auc_ci_upper = bootstrap_metric(average_precision_score, binary_labels, y_scores)
    print(f"ROC AUC: {roc_auc} with CI: [{roc_auc_ci_lower}, {roc_auc_ci_upper}], PR AUC: {pr_auc} with CI: [{pr_auc_ci_lower}, {pr_auc_ci_upper}]")

    # Dynamic threshold metrics
    thresholds = sorted(np.unique(y_scores), reverse=True)  # Dynamic thresholds from model output
    results = []
    for threshold in thresholds:
        y_pred_binary = (np.array(y_scores) >= threshold).astype(int)
        accuracy = accuracy_score(binary_labels, y_pred_binary)
        precision = precision_score(binary_labels, y_pred_binary, zero_division=0)
        sensitivity = recall_score(binary_labels, y_pred_binary)  # Recall
        specificity = calculate_specificity(binary_labels, y_pred_binary)
        f1 = f1_score(binary_labels, y_pred_binary, zero_division=0)
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'F1 Score': f1
        })

    # Calculate performance metrics at median threshold
    median_threshold = np.median(y_scores)
    y_pred_binary = (np.array(y_scores) >= median_threshold).astype(int)
    accuracy = accuracy_score(binary_labels, y_pred_binary)
    precision = precision_score(binary_labels, y_pred_binary, zero_division=0)
    sensitivity = recall_score(binary_labels, y_pred_binary)  # Recall
    specificity = calculate_specificity(binary_labels, y_pred_binary)
    f1 = f1_score(binary_labels, y_pred_binary, zero_division=0)

    # Bootstrapped confidence intervals for performance metrics
    accuracy_ci_lower, accuracy_ci_upper = bootstrap_metric(accuracy_score, binary_labels, y_pred_binary)
    precision_ci_lower, precision_ci_upper = bootstrap_metric(lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0), binary_labels, y_pred_binary)
    sensitivity_ci_lower, sensitivity_ci_upper = bootstrap_metric(recall_score, binary_labels, y_pred_binary)
    specificity_ci_lower, specificity_ci_upper = bootstrap_metric(calculate_specificity, binary_labels, y_pred_binary)
    f1_ci_lower, f1_ci_upper = bootstrap_metric(lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0), binary_labels, y_pred_binary)

    print(f"Metrics at median threshold: Accuracy: {accuracy} with CI: [{accuracy_ci_lower}, {accuracy_ci_upper}], "
          f"Precision: {precision} with CI: [{precision_ci_lower}, {precision_ci_upper}], Sensitivity: {sensitivity} with CI: [{sensitivity_ci_lower}, {sensitivity_ci_upper}], "
          f"Specificity: {specificity} with CI: [{specificity_ci_lower}, {specificity_ci_upper}], F1 Score: {f1} with CI: [{f1_ci_lower}, {f1_ci_upper}]")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_performance_metrics.csv', index=False)
    print("Saved model performance metrics to model_performance_metrics.csv")

    return spearman_corr, roc_auc, pr_auc

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GRU Model Development and Evaluation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    args = parser.parse_args()

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

    split_info = prepare_data(path)
    median_mem_score = split_info['mem_score'].median()

    train_df = split_info[split_info['type'] == 'train']
    val_df = split_info[split_info['type'] == 'val']
    test_df = split_info[split_info['type'] == 'test']

    train_sequences = create_sequence_mappings(train_df)
    val_sequences = create_sequence_mappings(val_df)
    test_sequences = create_sequence_mappings(test_df)

    if args.train:
        with open('train_sequences.pkl', 'wb') as f:
            pickle.dump(train_sequences, f)
        with open('val_sequences.pkl', 'wb') as f:
            pickle.dump(val_sequences, f)
        with open('test_sequences.pkl', 'wb') as f:
            pickle.dump(test_sequences, f)

        train_dataset = MemCatSequenceDataset(train_df, sequences=train_sequences, transform=train_preprocess)
        val_dataset = MemCatSequenceDataset(val_df, sequences=val_sequences, transform=test_preprocess)

        hidden_layer_sizes = [512, 256, 1024]  
        learning_rates = [0.0001, 0.01, 0.001] 
        batch_sizes = [1, 16, 8, 32, 64]  # Adjust based on memory constraints
        num_layers_options = [1, 2, 3]
        bidirectional_options = [False, True]
        dropout_options = [0.0, 0.2, 0.5]

        best_auc_val = float('-inf')
        best_model = None
        best_hyperparams = None
        performance_records = []

        for hidden_layer_size in hidden_layer_sizes:
            for lr in learning_rates:
                for batch_size in batch_sizes:
                    for num_layers in num_layers_options:
                        for bidirectional in bidirectional_options:
                            for dropout in dropout_options:
                                print(f"Training: hidden_size={hidden_layer_size}, lr={lr}, batch_size={batch_size}, dropout={dropout}, bidirectional={bidirectional}, num_layers={num_layers}")

                                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                                model = MemGRU(encoder=vgg_encoder, input_size=512*7*7, hidden_dim=hidden_layer_size, 
                                                       num_layers=num_layers, bidirectional=bidirectional, dropout=dropout).to(device)
                                optimizer = optim.Adam(model.parameters(), lr=lr)
                                criterion = nn.MSELoss()

                                trained_model, val_loss = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=50, early_stopping_patience=5, device=device)

                                # Save the model
                                model_name = f"gru_hidden{hidden_layer_size}_lr{lr}_batch{batch_size}_layers{num_layers}_bidirectional{bidirectional}_dropout{dropout}.pth"
                                torch.save(trained_model.state_dict(), model_name)

                                # Evaluate performance
                                y_true = []
                                y_scores = []
                                trained_model.eval()
                                with torch.no_grad():
                                    for inputs, labels in val_loader:
                                        inputs, labels = inputs.to(device), labels.to(device)
                                        outputs = trained_model(inputs)
                                        probabilities = torch.sigmoid(outputs).cpu().numpy().ravel()
                                        y_scores.extend(probabilities)
                                        actual_scores = labels.cpu().numpy().ravel()
                                        y_true.extend(actual_scores)

                                # Calculate metrics
                                spearman_corr, _ = spearmanr(y_true, y_scores)
                                binary_labels = (np.array(y_true) > median_mem_score).astype(int)
                                fpr, tpr, thresholds_roc = roc_curve(binary_labels, y_scores)
                                roc_auc = auc(fpr, tpr)
                                pr_auc = average_precision_score(binary_labels, y_scores)

                                # Record performance
                                performance_records.append({
                                    'hidden_layer_size': hidden_layer_size,
                                    'lr': lr,
                                    'batch_size': batch_size,
                                    'num_layers': num_layers,
                                    'bidirectional': bidirectional,
                                    'dropout': dropout,
                                    'spearman_corr': spearman_corr,
                                    'roc_auc': roc_auc,
                                    'pr_auc': pr_auc
                                })

                                # Update best model based on validation performance
                                if roc_auc > best_auc_val:
                                    best_auc_val = roc_auc
                                    best_model = copy.deepcopy(trained_model)
                                    best_hyperparams = {
                                        'hidden_layer_size': hidden_layer_size,
                                        'lr': lr,
                                        'batch_size': batch_size,
                                        'num_layers': num_layers,
                                        'bidirectional': bidirectional,
                                        'dropout': dropout
                                    }

        # Save performance records to a CSV file for analysis
        performance_df = pd.DataFrame(performance_records)
        performance_df.to_csv('gru_performance_records.csv', index=False)

        # Save the best model
        torch.save(best_model.state_dict(), "mem_gru_statedict.pth") #best performing GRU model
        torch.save(best_model, "mem_gru.pth") #best performing GRU model

        print("Best Model Hyperparameters:")
        print(best_hyperparams)

    if args.evaluate:
        # Load the best model and evaluate
        model = MemGRU(encoder=vgg_encoder, inputsize=512*7*7, hidden_dim=best_hyperparams['hidden_layer_size'], 
                               num_layers=best_hyperparams['num_layers'], bidirectional=best_hyperparams['bidirectional'], 
                               dropout=best_hyperparams['dropout']).to(device)
        model.load_state_dict(torch.load("gmem_gru_statedict.pth", map_location=device))

        test_dataset = MemCatSequenceDataset(test_df, sequences=test_sequences, transform=test_preprocess)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Adjust batch size if needed

        # Evaluate model performance
        spearman_corr, roc_auc, pr_auc = evaluate_model_performance(model, test_loader, device, median_mem_score)

