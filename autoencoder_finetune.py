"""
This script fine-tunes the VGG autoencoder model on a dataset with various loss functions.
It supports different loss functions, batch sizes, and learning rates.
The script saves the trained model state after each epoch.

The following directory structure is assumed:
MemCat/
    ├── animal/
    ├── vehicle/
    ├── landscape/
    ├── sports
    └── food/
    

"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import lpips
from torchvision.models import vgg19
from imagenet_autoencoder_main.models import vgg
from pytorch_msssim import SSIM, ms_ssim

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Define paths for the dataset
path = os.getcwd()
data_dir = os.path.join(path, 'MemCat')


# Image transformations
train_preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# Load the data
train_data = ImageFolder(data_dir, transform=train_preprocess)

# Load pretrained model
state_dict_path = os.path.join(path, 'imagenet_vgg16_autoencoder.pth')
state_dict = torch.load(state_dict_path, map_location='cpu')['state_dict']
configs = vgg.get_configs()
model = vgg.VGGAutoEncoder(configs)
model.load_state_dict(state_dict, strict=False)
model.to(device)

class StyleLoss(nn.Module):
    """Defines a style loss using Gram matrix for style comparison."""
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.feature_extractor = vgg19(pretrained=True).features.eval()
        self.gram = lambda x: torch.mm(x, x.t())

    def forward(self, input, target):
        features_input = self.feature_extractor(input)
        features_target = self.feature_extractor(target)
        G_input = self.gram(features_input.view(features_input.size(1), -1))
        G_target = self.gram(features_target.view(features_target.size(1), -1))
        loss = nn.MSELoss()(G_input, G_target)
        return loss

def get_criterion(loss_fcn):
    """Returns the appropriate loss function."""
    if loss_fcn == 'ssim':
        return SSIM(data_range=1.0, channel=3, size_average=True).to(device)
    elif loss_fcn == 'lpips-vgg':
        return lpips.LPIPS(net='vgg').to(device)
    elif loss_fcn == 'lpips-sq':
        return lpips.LPIPS(net='squeeze').to(device)
    elif loss_fcn == 'lpips-alex':
        return lpips.LPIPS(net='alex').to(device)
    elif loss_fcn == 'styleLoss':
        return StyleLoss().to(device)
    elif loss_fcn == 'mae':
        return nn.L1Loss().to(device)
    elif loss_fcn == 'mse':
        return nn.MSELoss().to(device)
    elif loss_fcn == 'ms-ssim':
        return ms_ssim

def train_model(model, train_loader, criterion, optimizer, loss_fcn):
    """Trains the model for one epoch."""
    model.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        reconstructed = model(images)
        if loss_fcn == 'ms-ssim':
            loss = 1 - criterion(images, reconstructed, data_range=1.0, size_average=True)
        elif isinstance(criterion, (SSIM, lpips.LPIPS)):
            loss = 1 - criterion(images, reconstructed)
        else:
            loss = criterion(reconstructed, images)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)

def validate_model(model, val_loader, criterion, loss_fcn):
    """Validates the model."""
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            if loss_fcn == 'ms-ssim':
                loss = 1 - criterion(images, outputs, data_range=1.0, size_average=True)
            elif isinstance(criterion, (SSIM, lpips.LPIPS)):
                loss = 1 - criterion(images, outputs)
            else:
                loss = criterion(outputs, images)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def save_model_state(model, path, model_name):
    """Saves the model state."""
    model_state_path = os.path.join(path, f'{model_name}_statedict.pth')
    torch.save(model.state_dict(), model_state_path)

def fine_tune_model(model, train_data, val_data, learning_rates, batch_sizes, loss_funcs, num_epochs):
    """Fine-tunes the model using different loss functions, batch sizes, and learning rates."""
    for loss_fcn in loss_funcs:
        criterion = get_criterion(loss_fcn)
        for lr in learning_rates:
            for bs in batch_sizes:
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
                for epoch in range(num_epochs):
                    model_name = f'MemCat_Autoencoder_vgg16_lr{lr}_bs{bs}_LossFunc{loss_fcn}_epoch{epoch+1}'
                    train_loss = train_model(model, train_loader, criterion, optimizer, loss_fcn)
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
                    save_model_state(model, path, model_name)



if __name__ == "__main__":
    # Define learning rates, batch sizes, and loss functions
    learning_rates = [5e-6]#[1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05, 0.1]
    batch_sizes = [1]
    loss_funcs = ['lpips-sq']#['lpips-sq', 'lpips-vgg', 'lpips-alex', 'mse', 'ssim', 'ms-ssim', 'mae', 'styleLoss']
    num_epochs = 1
    # num_epochs = 20 # uncomment to explore the effect of contunuing fine-tuning for more epochs

    # Fine-tune the model
    fine_tune_model(model, train_data, learning_rates, batch_sizes, loss_funcs, num_epochs)
