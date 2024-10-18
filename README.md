# Modeling Image Memorability 

This repository contains scripts for modeling image memorability using a vgg16-based autoencoder.

## Prerequisites

- Python 3.11.9
- Required Python packages can be installed using the provided `requirements.txt` file.

## Installation

1. Clone the repository:
    git clone https://github.com/yourusername/memcat-analysis.git
    cd memcat-analysis
2. Install the required packages:
    pip install -r requirements.txt

## Dataset

Download the MemCat data from https://gestaltrevision.be/projects/memcat/ and place it in the ./MemCat/ directory.

## Analysis

   1. Download the pre-trained model:
      Download the vgg16 autoencoder pre-trained on ImageNet from https://github.com/Horizon2333/imagenet-autoencoder and place it in the current directory.
   2. Fine-tune autoencoder:
      to fine-tune the autoencoder on the MemCat dataset:  python autoencoder_finetune.py
   3. Test the fine-tuned models: python autoencoder_test.py
   4. Autoencoder evaluation:
      - To run the analysis for all models and save the results:
        python autoencoder_evaluation.py --output-path ./output
      - To run the analysis for specific models and save the results:
        python autoencoder_evaluation.py --output-path ./output --model-names model1.pth model2.pth
        example: autoencoder_evaluation.py --output-path ./output --model-names mem_vgg_autoencoder.pth
   5. Calculate latent space representations and latent code distinctiveness:
     - To calculate latent representations: 
       python latent_analysis.py --calculate-latents --output-path ./output
     - To perform distinctiveness analysis:
       python latent_analysis.py --distinctiveness-analysis --output-path ./output
    - To perform both calculations and evaluations:
       python latent_analysis.py --calculate-latents --distinctiveness-analysis --output-path ./output
   6. Category plot:
      to create binned plots for each category:  python category_plot.py  
      
   For memorability prediction using latent code:
   
   7. Split data into train, validation, and test Sets:
      to split the data:  python split_data.py
   8. Train and evaluate the GRU model:
      - to train the GRU model: python gru_training_evaluation.py --train
      - to evaluate the GRU model: python gru_training_evaluation.py --evaluate
   9. Interpret the model using Integrated Gradients analysis:
      - to save the IG attributions: python model_interpretation_IG.py --save
      - to display the attributions: python model_interpretation_IG.py --display
      - to do both: python model_interpretation_IG.py --save --display
   10. Feature analysis: feature extraction and evaluation
      - to calculate and save the features: python feature_analysis.py --calculate-features --output-path ./output
      - to evaluate feature correlations with memorability scores: python feature_analysis.py --evaluate-features --output-path ./output
      - to perform both calculations and evaluations: python feature_analysis.py --calculate-features --evaluate-features --output-path ./output

**Notes**

- The scripts assume a specific directory structure for datasets and models.
- Modify the paths in the scripts if your directory structure is different.
- Ensure all necessary dependencies are installed before running the scripts.

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license. You are free to share, copy, and redistribute the material in any medium or format for non-commercial purposes, provided appropriate credit is given, and no modifications or derivative works are made. 

## Citation

If you use this repository in your work, please cite our paper.
