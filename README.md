# CVTC: Cross Vision Transformer with Coordinate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0-orange)](https://pytorch.org/)

## Overview

CVTC is a lightweight deep learning model designed for accurate analysis of Alzheimer's Disease (AD) MRI images and lesion annotation. It integrates advanced techniques such as scale-adaptive embedding, dynamic position bias, long-short attention mechanisms, and the Coordinate and Feature Map Guided Mechanism (CAGM) to achieve high diagnostic accuracy, generalization, and interpretability. The model is particularly suited for clinical applications, reducing diagnostic burden on physicians while providing reliable lesion annotations.

Key features:
- **High Accuracy**: Achieves up to 98.80% on ADNI dataset for AD/MCI/CN classification and 98.51% for AD subtypes.
- **Lightweight**: Model size of only 21.850 MB, making it efficient for deployment.
- **Interpretability**: CAGM generates intuitive lesion annotation maps.
- **Generalization**: Validated on multiple datasets including ADNI, OASIS-1, Kaggle, and cross-dataset scenarios like tumors and multiple sclerosis.

This repository contains the implementation of CVTC, including code for training, inference, and the user-friendly UI for lesion exploration.

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.5.0
- Other dependencies: `opencv-python==4.10.0.84`, `nibabel==5.3.2`, `scikit-learn==1.5.0`, `scipy==1.14.0`

Install dependencies via pip:
```
pip install -r requirements.txt
```

### Setup
Clone the repository:
```
git clone https://github.com/52hearts3/CVTC.git
cd CVTC
```

## Usage

### Training
To train the model on your dataset:
```
python train.py --dataset_path /path/to/dataset --epochs 50 --batch_size 32
```

### Inference
Run inference on MRI images:
```
python infer.py --model_path checkpoints/cvtc.pth --image_path /path/to/mri_image.nii --output_dir results
```

This will output diagnostic results and lesion annotation maps.

### UI Interface
Launch the user-friendly UI for interactive lesion annotation:
```
python app.py
```
The UI allows physicians to upload MRI images, view annotations, and validate results intuitively.

## Methods

### Workflow Overview
The CVTC framework processes MRI images through several stages: skull stripping, data augmentation, feature extraction, diagnosis, and lesion annotation. The overall workflow is illustrated below:

![Workflow Diagram](CVTC/images/image1.png)

1. **Skull Stripping**: Use LinkNet3D to separate brain regions from the skull.
2. **Data Augmentation**: Apply MBIE (Multi-dimensional Brain Image Enhancement) for multi-channel enhancements.
3. **Feature Extraction**: Employ scale-adaptive embedding, LS-Transformer with dynamic position bias and long-short attention.
4. **Diagnosis**: Classify AD severity.
5. **Lesion Annotation**: Use CAGM to track coordinates and feature maps for annotation.

### Skull and Brain Separation with LinkNet3D
LinkNet3D is a 3D segmentation network that efficiently extracts brain regions by combining downsampling, upsampling, and skip connections. It processes MRI volumes to generate a brain mask, binarizes it, and applies it to isolate the brain. This step reduces noise and focuses on relevant areas.

Example results of unprocessed vs. processed MRI slices after skull stripping:

![Skull Stripping Results](CVTC/images/image2.png)

### Data Augmentation with MBIE
MBIE enhances MRI images by processing RGB channels separately (e.g., CLAHE on green, unsharp masking on blue, sharpening on red) to improve contrast and detail while preserving 95.33% of original features. This augments the dataset and boosts model robustness.

### Scale-Adaptive Embedding (SAE)
SAE dynamically adjusts embedding based on image scales, capturing both fine-grained and global features.

### LS-Transformer with DPB
The Long-Short Transformer (LS-Transformer) integrates long attention for global context and short attention for local details, enhanced by Dynamic Position Bias (DPB) for positional awareness.

### Diagnostic Module
Outputs AD classification probabilities.

### Coordinate and Feature Map Guided Mechanism (CAGM)
CAGM computes importance thresholds from pixel coordinates and multi-scale feature maps to generate lesion annotations, providing spatial-semantic insights.

## Results

### Performance Metrics
CVTC outperforms several baselines across datasets:

| Model                | Kaggle | OASIS-1 | ADNI   | Memory Size |
|----------------------|--------|---------|--------|-------------|
| Inception-ResNet [29]| 91.43% | 92%    | N/A    | 170 MB     |
| CNNs [30]            | N/A    | N/A    | 99%    | N/A        |
| CNN and SVM [31]     | N/A    | 88.84% | N/A    | 20-50 MB   |
| LSTM [32]            | N/A    | 91.8%  | 89.8%  | N/A        |
| CNN [32]             | N/A    | 92.5%  | 90.5%  | N/A        |
| LGBM [33]            | N/A    | 91.95% | 99.63% | N/A        |
| **CVTC**             | **99.61%** | **98.16%** | **98.50%** | **21.850 MB** |

Additional results:
- Pseudo-RGB synthetic datasets: 92.96%
- Cross-dataset (tumors/MS): High annotation accuracy

### Lesion Annotation Showcase
CAGM provides precise lesion maps, highlighting suspicious areas in AD MRI images. Below is an example of annotated MRI slices:

![Lesion Annotation Results](CVTC/images/image3.png)

Ablation studies confirm the contributions of each component (e.g., SAE improves accuracy by 2-3%).

## Datasets
All datasets used are publicly available:
- ADNI: https://adni.loni.usc.edu/
- OASIS: https://sites.wustl.edu/oasisbrains/
- Kaggle (Augmented AD MRI): https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset
- Skull-stripped MRI (our release): https://www.kaggle.com/datasets/yiweilu2033/well-documented-alzheimers-dataset
- Other: BraTS2020, NFBS, etc. (see paper for full links)

## Acknowledgements
Thanks to the dataset providers and reviewers. Special thanks to Yao Miaoran for support.

## Citation
If you use CVTC in your research, please cite:
```
@article{CVTC2025,
  title={CVTC: A Lightweight Model for Accurate Alzheimer's MRI Analysis and Lesion Annotation},
  author={Lu, Yiwei and Yu, Hongcheng and Li, Tianbao and Meng, Yuting and Lu, Jianbo and Li, Peiluan},
  journal={TBD},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions, contact the corresponding author: Peiluan Li (email not provided in doc).
