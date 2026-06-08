# 🔬 Skin Cancer Classification — CNN Deep Learning Approach

### **Multi-Class Dermoscopic Image Classification Pipeline utilizing TensorFlow/Keras & the HAM10000 Dataset**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-2.12%2B-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-success.svg?style=for-the-badge)](LICENSE)

**An automated clinical diagnostic support tool classifying 7 distinct types of skin cancer lesions from dermoscopic images using custom convolutional layers, class imbalance augmentation, and adaptive learning rate scheduling.**

[Overview](#-about-the-project) • [Pipeline Architecture](#-data-processing-pipeline) • [CNN Model Design](#-cnn-model-architecture) • [Classes Description](#-skin-lesion-classes) • [Saved Artifacts](#-saved-artifacts) • [Quick Start](#-installation)

---

## 📖 About the Project

Melanoma and other skin carcinomas present severe public health concerns. Early, objective detection drastically improves prognosis. This project outlines the implementation of an end-to-end deep learning diagnostic pipeline built from scratch to categorize skin lesions into 7 classes using the benchmark **HAM10000** dataset.

The system handles severe class imbalances using real-time spatial transformations (augmentations) and maps final results onto a detailed confusion matrix.

---

## 🏗️ Data Processing Pipeline

The following flowchart maps the dataset's path from raw ingestion to the trained model:

```mermaid
graph TD
    %% Dataset
    Raw[HAM10000 Dataset: 10,000 Images] -->|Download via Kagglehub| Local[Local archive/ Folder]
    Local -->|Extract Metadata| Meta[HAM10000_metadata.csv]
    
    %% Preprocessing
    Meta --> Clean[Parse Age, Gender & Localization]
    Clean --> Resize[Resize Images to 100x75 Pixels]
    Resize --> Scale[Rescale Intensities to 0-1]
    
    %% Augmentation
    Scale --> Aug{Imbalance Augmentation}
    Aug -->|Rotation, Zoom, Flips & Shifts| Split[90/10 Train-Val Split]
    
    %% Training
    Split --> Train[Model Training: 50 Epochs]
    Train -->|Adam Optimizer| Eval[Model Evaluation]
    Train -->|ReduceLROnPlateau Scheduler| Eval
    
    %% Output
    Eval --> Dump[Save: model.h5]
    Eval --> Matrix[Compute Confusion Matrix]

    style Train fill:#111827,stroke:#ff7c00,stroke-width:2px,color:#fff
    style Eval fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff
```

---

## 🧠 CNN Model Architecture

The neural network is built with stacked convolutional blocks, pooling nodes, dropout regularization, and a dense multi-class classifier:

```mermaid
graph TB
    %% Inputs
    Input[Input Image: 75 x 100 x 3] --> ConvBlock1[Convolutional Block 1]
    
    %% Conv Block 1
    subgraph ConvBlock1 [Conv Block 1: Feature Extraction]
        C1[Conv2D: 32 Filters, 3x3] --> R1[ReLU Activation]
        R1 --> C2[Conv2D: 32 Filters, 3x3]
        C2 --> R2[ReLU Activation]
        R2 --> P1[MaxPooling2D: 2x2]
        P1 --> D1[Dropout: 25%]
    end
    
    %% Conv Block 2
    D1 --> ConvBlock2[Convolutional Block 2]
    subgraph ConvBlock2 [Conv Block 2: Deep Features]
        C3[Conv2D: 64 Filters, 3x3] --> R3[ReLU Activation]
        R3 --> C4[Conv2D: 64 Filters, 3x3]
        C4 --> R4[ReLU Activation]
        R4 --> P2[MaxPooling2D: 2x2]
        P2 --> D2[Dropout: 40%]
    end
    
    %% Dense Block
    D2 --> DenseBlock[Fully Connected Dense Block]
    subgraph DenseBlock [Dense Block: Classification Mapping]
        F1[Flatten Features] --> Dn1[Dense: 128 Neurons]
        Dn1 --> Rd1[ReLU Activation]
        Rd1 --> D3[Dropout: 50%]
    end
    
    %% Softmax Out
    D3 --> Out[Dense Output: 7 Neurons]
    Out --> Softmax[Softmax Layer]
    Softmax --> Predict[7 Class Probability Vector]

    style ConvBlock1 fill:#111827,stroke:#3b82f6,stroke-width:2px,color:#fff
    style ConvBlock2 fill:#111827,stroke:#22d3ee,stroke-width:2px,color:#fff
    style DenseBlock fill:#111827,stroke:#a78bfa,stroke-width:2px,color:#fff
```

---

## 🩺 Skin Lesion Classes

The pipeline maps predictions to these seven diagnostic classes defined by the International Skin Imaging Collaboration (ISIC):

| Label Code | Disease / Lesion Class | Clinical Classification |
|------------|------------------------|-------------------------|
| **`akiec`** | Actinic keratoses & Intraepithelial Carcinoma | Pre-cancerous |
| **`bcc`** | Basal cell carcinoma | Cancerous (Malignant) |
| **`bkl`** | Benign keratosis-like lesions | Benign |
| **`df`** | Dermatofibroma | Benign |
| **`mel`** | Melanoma | Cancerous (Highly Malignant) |
| **`nv`** | Melanocytic nevi (Common Moles) | Benign |
| **`vasc`** | Vascular lesions | Benign |

---

## 🛠️ Hyperparameter Specifications

| Parameter | Value | Details |
|-----------|-------|---------|
| **Optimizer** | Adam | Initial Learning Rate = $0.001$ |
| **Loss Function** | Categorical Crossentropy | Categorical target arrays |
| **Epochs** | 50 | Early stopping patience = 5 |
| **Batch Size** | 10 | Mini-batch gradient descent |
| **Learning Rate Scheduler** | ReduceLROnPlateau | Reduces LR by $0.3\times$ if validation loss stalls for 2 epochs |

---

## 📦 Saved Artifacts

```
skin-cancer-classification/
├── skin-cancer-classification-cnn-approach.ipynb   # Diagnostic Pipeline Notebook
├── requirements.txt                                 # Core Dependencies
├── model.h5                                         # Trained weights (HDF5 format)
├── model_plot.png                                   # Plot of Keras Model Architecture
└── README.md
```

---

## ⚙️ Installation & Usage

### 1. Clone & Environment Set Up
```bash
git clone https://github.com/rajmodi262/SKIN_DISEASE_CLASSIFICATION.git
cd SKIN_DISEASE_CLASSIFICATION
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, run: `pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pillow jupyter`)*

### 2. Download HAM10000 Dataset
Download from [Kaggle HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place the folders inside an `archive/` directory in the project root:
```
SKIN_DISEASE_CLASSIFICATION/
├── archive/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
```

### 3. Run the Pipeline
```bash
jupyter notebook skin-cancer-classification-cnn-approach.ipynb
```
Select "Run All Cells" to trigger preprocessing, augmentation, network training, and model evaluation.
