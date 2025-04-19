# Unsupervised Representation Learning for Understanding and Predicting Image Memorability through Autoencoders and Explainable AI

This repository contains scripts for the official implementation of the paper:

**Elham Bagheri and Yalda Mohsenzadeh**  
*Modeling Visual Memorability Assessment with Autoencoders Reveals Characteristics of Memorable Images*  
📄 [arXiv:2410.15235](https://arxiv.org/abs/2410.15235)

We introduce an unsupervised deep learning framework based on convolutional autoencoders to investigate the visual and structural features that make images memorable. Trained under a single-exposure setting to simulate human memory experiments, the model captures image representations that reveal how distinctiveness and reconstruction difficulty relate to memorability. We further leverage explainable AI to interpret these findings and explore the cognitive underpinnings of memory encoding.

Our approach departs from traditional supervised memorability prediction by modeling the phenomenon through **reconstruction dynamics**, **latent space distinctiveness**, and **feature-level analysis**—without requiring labeled training data.

### Key components of this work include:

- **Fine-tuning a convolutional autoencoder** (based on VGG16), pretrained on ImageNet, using the MemCat dataset to mimic single-exposure human memory conditions  
- **Analyzing reconstruction error and latent space distinctiveness**, hypothesizing that harder-to-reconstruct and more distinctive images are more memorable  
- **Predicting memorability using deep neural networks (MLPs)** trained on high-level latent representations extracted from the autoencoder  
- **Interpreting memorability predictions** using Integrated Gradients to identify spatially salient regions responsible for memorability outcomes  
- **Feature analysis using a rich set of visual and semantic descriptors** to quantify what makes images memorable:
  - **Scene composition and object richness**, extracted via pre-trained object detection (Faster R-CNN) and semantic segmentation (DeepLabV3) models  
  - **Color statistics** from LAB and HSV color spaces, including saturation, chromatic contrast (A/B channels), and **color diversity** via Shannon entropy  
  - **Texture and structural complexity**, measured through image entropy, GLCM-based **texture energy**, and **clutter scores** based on Canny edge detection  
  - Correlating these features with both **memorability scores** and **Integrated Gradient attributions** reveals how **scene complexity**, **visual saliency**, and **foreground-background separation** drive memory encoding  

This work provides computational insights into **visual memorability mechanisms** and offers practical tools for identifying or generating high-memorability images in real-world settings. The proposed **unsupervised framework** supports applications in **content creation, education, advertising, media, and visual communication**, enabling the generation of memorable visual content *without relying on large-scale labeled datasets*.

---

## Prerequisites

- Python 3.11.9
- Required Python packages can be installed using the provided `requirements.txt` file.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/memcat-analysis.git
    cd memcat-analysis
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Dataset

Download the MemCat data from [https://gestaltrevision.be/projects/memcat/](https://gestaltrevision.be/projects/memcat/) and place it in the `./MemCat/` directory.

---

## Analysis Pipeline

### 🔧 Autoencoder Training & Evaluation

1. **Download the pre-trained base model** (VGG16 autoencoder trained on ImageNet):  
   [https://github.com/Horizon2333/imagenet-autoencoder](https://github.com/Horizon2333/imagenet-autoencoder)  
   Place the downloaded model in the current directory.

2. **Fine-tune the autoencoder on the MemCat dataset**  
    ```bash
    python autoencoder_finetune.py
    ```

3. **Test the fine-tuned model**  
    ```bash
    python autoencoder_test.py
    ```

4. **Evaluate autoencoder performance**  
   - All models:
     ```bash
     python autoencoder_evaluation.py --output-path ./output
     ```
   - Specific models:
     ```bash
     python autoencoder_evaluation.py --output-path ./output --model-names mem_vgg_autoencoder.pth
     ```

5. **Calculate latent space representations and distinctiveness**  
   - To calculate latent representations: 
     ```bash
     python latent_analysis.py --calculate-latents --output-path ./output
     ```
   - To perform distinctiveness analysis:
     ```bash
     python latent_analysis.py --distinctiveness-analysis --output-path ./output
     ```
   - To perform both:
     ```bash
     python latent_analysis.py --calculate-latents --distinctiveness-analysis --output-path ./output
     ```

6. **Generate category-level visualizations**  
    ```bash
    python category_plot.py
    ```

---

### 🧠 Memorability Prediction Using Latent Codes

*Note: The MLP training script for memorability prediction will be released soon.*  
We train a Multi-Layer Perceptron (MLP) on latent representations extracted from the autoencoder to predict image memorability.

---

### 🔍 Interpretability and Feature Analysis

7. **Integrated Gradients (IG) Attribution Analysis**  
   - To save the IG attributions:
     ```bash
     python model_interpretation_IG.py --save
     ```
   - To display the attributions:
     ```bash
     python model_interpretation_IG.py --display
     ```
   - To run both:
     ```bash
     python model_interpretation_IG.py --save --display
     ```

8. **Feature extraction and correlation evaluation**  
   - To calculate and save high-level image features:
     ```bash
     python feature_analysis.py --calculate-features --output-path ./output
     ```
   - To evaluate feature correlations with memorability and IG attributions:
     ```bash
     python feature_analysis.py --evaluate-features --output-path ./output
     ```
   - To perform both:
     ```bash
     python feature_analysis.py --calculate-features --evaluate-features --output-path ./output
     ```

---

## Notes

- The scripts assume a specific directory structure for datasets and models.
- Modify the paths in the scripts if your directory structure is different.
- Ensure all necessary dependencies are installed before running the scripts.

---

## License

This repository is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license. You are free to share, copy, and redistribute the material in any medium or format for non-commercial purposes, provided appropriate credit is given and no modifications or derivative works are made.

---

## Citation

If you use this repository or findings from the paper in your work, please cite:

```bibtex
@article{bagheri2025memorability,
  title={Modeling Visual Memorability Assessment with Autoencoders Reveals Characteristics of Memorable Images}, 
  author={Bagheri, Elham and Mohsenzadeh, Yalda},
  journal={arXiv preprint arXiv:2410.15235},
  year={2025},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.15235}
}
