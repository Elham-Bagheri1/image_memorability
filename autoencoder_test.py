"""
This script evaluates a VGG autoencoder model using various loss functions on a test dataset.
The script loads the test dataset, applies the model, computes the loss, and saves the results to CSV files.

Ensure the necessary dependencies are installed before running this script.

The following directory structure is assumed:
MemCat/
    ├── animal/
    ├── vehicle/
    ├── landscape/
    ├── sports
    └── food/
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from imagenet_autoencoder_main.models import vgg
from pytorch_msssim import SSIM, ms_ssim
import lpips
from torchvision.models import vgg19

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

def get_files_with_extension(directory, extension):
    """Returns a list of files in the directory with the specified extension."""
    if not extension.startswith("."):
        extension = "." + extension
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)]

def load_data(data_dir):
    """Loads the dataset with preprocessing transformations."""
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    return ImageFolder(data_dir, transform=preprocess)

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

def run_model(model_name, test_data, loss_fcn, trainfiles, memscore):
    """Runs the model on the test data and computes the loss."""
    criterion = get_criterion(loss_fcn)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    state_dict = torch.load(model_name, map_location=torch.device(device))
    configs = vgg.get_configs()
    model = vgg.VGGAutoEncoder(configs)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    image_filenames = []
    test_loss = []
    
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        if loss_fcn == 'ms-ssim':
            loss = 1 - ms_ssim(images, outputs, data_range=1.0, size_average=True)
        elif isinstance(criterion, (SSIM, lpips.LPIPS)):
            loss = 1 - criterion(outputs, images)
        else:
            loss = criterion(outputs, images)
        if len(loss.size()) > 0:
            loss = loss.mean()
        test_loss.append(loss.item())
        
        batch_indices = test_loader.batch_sampler.sampler.indices[i * test_loader.batch_size: (i + 1) * test_loader.batch_size] if hasattr(test_loader.batch_sampler.sampler, 'indices') else range(i * test_loader.batch_size, (i + 1) * test_loader.batch_size)
        image_filenames.append([test_data.imgs[idx][0] for idx in batch_indices])

    images_names = [x[0].split('/')[-1] for x in image_filenames]
    
    mem_orig = []
    ids = []
    names = []
    for i, namen in enumerate(images_names):
        for j, name in enumerate(trainfiles):
            if namen == name:
                mem_orig.append(memscore[j])
                ids.append(i)
                names.append(name)

    test_loss_new = [test_loss[i] for i in ids]

    scores_df = pd.DataFrame({'image': names, 'Score': mem_orig, 'Loss': test_loss_new})
    return scores_df

def main():
    """Main function to run the evaluation on multiple models and loss functions."""
    path = os.getcwd()
    data_dir = os.path.join(path, 'MemCat')
    test_data = load_data(os.path.join(data_dir))

    memcat_info = pd.read_csv(os.path.join(path, 'memcat_image_data.csv'))
    split_info = pd.read_csv(os.path.join(path, 'MemCat_split_info.csv'))

    train_images = split_info.copy()
    train_images.reset_index(inplace=True, drop=True)
    train_images['image'] = train_images['new_path'].str.split('/').str[-1]
    train_images = pd.merge(train_images, memcat_info[['image_file', 'memorability_w_fa_correction', 'memorability_wo_fa_correction']],
                            left_on='image', right_on='image_file', how='left')
    train_images.rename(columns={'memorability_w_fa_correction': 'mem score',
                                 'memorability_wo_fa_correction': 'mem score wo'}, inplace=True)

    memscore = train_images['mem score'].values
    trainfiles = train_images['image'].values

    model_folder = path
    all_files = os.listdir(model_folder)
    model_names = [file for file in all_files if (
        'MemCat_Autoencoder_vgg16' in file and
        'statedict.pth' in file
    )]

    loss_funcs = ['mse', 'lpips-vgg', 'ssim', 'ms-ssim', 'mae', 'lpips-sq', 'lpips-alex', 'styleLoss']
    output_path = os.path.join(path, 'output/')
    os.makedirs(output_path, exist_ok=True)

    for model_name in model_names:
        for loss_fcn in loss_funcs:
            scores_df = run_model(os.path.join(model_folder, model_name), test_data, loss_fcn, trainfiles, memscore)
            scores_df.to_csv(os.path.join(output_path, f'MemCat_{model_name}_{loss_fcn}_testLoss.csv'), index=False)

if __name__ == "__main__":
    main()
