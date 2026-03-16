# 🔬 Skin Cancer Classification — CNN Approach

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.12%2B-red?logo=keras)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning project that classifies **7 types of skin cancer lesions** from dermoscopic images using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras. The model is trained on the well-known **HAM10000 dataset** from Kaggle.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Skin Lesion Classes](#skin-lesion-classes)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)

---

## 📖 About the Project

Skin cancer is the most common form of cancer. Early and accurate diagnosis significantly improves survival rates. This project builds an automated classification system using CNN to detect and classify skin lesions from dermoscopic images — helping support clinical decision-making.

The notebook covers the full ML pipeline:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Augmentation
- CNN Model Building & Training
- Evaluation & Visualization

---

## 📦 Dataset

**HAM10000** (*Human Against Machine with 10000 training images*)

- Source: [Kaggle — Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- ~10,000 dermoscopic images across 7 categories
- Metadata includes: patient age, gender, localization area, and diagnosis type

To download the dataset, run:
```bash
# Using kagglehub
import kagglehub
path = kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000')
```

Place the downloaded files inside an `archive/` folder in the project root.

---

## 🩺 Skin Lesion Classes

| Code | Full Name |
|------|-----------|
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `bkl` | Benign keratosis-like lesions |
| `bcc` | Basal cell carcinoma |
| `akiec` | Actinic keratoses |
| `vasc` | Vascular lesions |
| `df` | Dermatofibroma |

---

## 📁 Project Structure

```
skin-cancer-classification/
│
├── archive/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
├── skin-cancer-classification-cnn-approach.ipynb   # Main notebook
├── requirements.txt                                 # Dependencies
├── model.h5                                         # Saved model (after training)
├── model_plot.png                                   # CNN architecture diagram
├── category_samples.png                             # Sample images per class
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/skin-cancer-classification.git
cd skin-cancer-classification
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it in the `archive/` folder.

---

## 🚀 Usage

Launch the Jupyter Notebook:
```bash
jupyter notebook skin-cancer-classification-cnn-approach.ipynb
```

Run all cells sequentially. The notebook will:
1. Load and explore the HAM10000 dataset
2. Perform EDA (gender, age, localization, cell type distributions)
3. Preprocess and normalize images (resized to `100×75`)
4. Apply data augmentation to handle class imbalance
5. Train the CNN model for 50 epochs
6. Evaluate and save the model as `model.h5`

---

## 🧠 Model Architecture

```
Input (75 × 100 × 3)
    ↓
[Conv2D(32) → ReLU → Conv2D(32) → ReLU] → MaxPool2D → Dropout(0.25)
    ↓
[Conv2D(64) → ReLU → Conv2D(64) → ReLU] → MaxPool2D → Dropout(0.40)
    ↓
Flatten → Dense(128, ReLU) → Dropout(0.50)
    ↓
Dense(7, Softmax)  →  Output (7 classes)
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 10 |
| Train/Val Split | 90% / 10% |
| Train/Test Split | 80% / 20% |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.3) |

**Data Augmentation applied:**
- Random rotation (±10°)
- Random zoom (10%)
- Horizontal & vertical flips
- Width & height shifts (10%)

---

## 📊 Results

After training, the model is evaluated on the test set. Key metrics are printed at the end of the notebook:

```
Validation: accuracy = X.XX  ;  loss = X.XX
Test:        accuracy = X.XX  ;  loss = X.XX
```

A **confusion matrix** is also generated to visualize per-class performance.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | CNN model building & training |
| `NumPy / Pandas` | Data manipulation |
| `Matplotlib / Seaborn` | Static visualizations |
| `Plotly` | Interactive visualizations |
| `Pillow (PIL)` | Image loading & resizing |
| `scikit-learn` | Train/test split, confusion matrix |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset: [HAM10000 by kmader on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Inspired by research on AI-assisted dermatology diagnosis
