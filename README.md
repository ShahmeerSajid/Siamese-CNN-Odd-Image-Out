# Odd-One-Out Image Classification using a Siamese Convolutional Neural Network

## Project Overview

This project presents a deep learning solution for the **Odd-One-Out Image Classification** task, where each sample consists of five grayscale images and the objective is to identify the image that does not belong to the same visual category as the remaining four.

Unlike conventional image classification, this problem requires **relational reasoning** rather than independent image recognition. To address this, a **Siamese Convolutional Neural Network (CNN)** was designed to learn a shared embedding space in which similar images are mapped close together while the outlier naturally exhibits the largest average embedding distance.

The project was completed as part of **COMP 551 – Applied Machine Learning** at **McGill University**.

---

# Problem Statement

Given five grayscale images

```
Image 1
Image 2
Image 3
Image 4
Image 5
```

predict the index of the image that does **not** share the common visual property of the remaining four images.

The challenge is inherently **relational**, meaning the prediction cannot be made from a single image in isolation.

---

# Dataset

- 3,000 training groups
- 2,000 testing groups
- Five grayscale images per sample
- Resolution: **32 × 32**
- Five-class classification problem

---

# Data Preprocessing

To improve invariance and reduce irrelevant variation, several preprocessing techniques were applied.

### Shape Normalization

Each image is

- thresholded
- cropped using its bounding box
- padded
- resized back to 32×32

This removes unnecessary translation and scale variation.

---

### Edge Augmentation

A Sobel edge detector is applied to every image.

The resulting edge map is concatenated with the normalized image, producing a **two-channel input**.

```
Channel 1 → Normalized image

Channel 2 → Sobel edge map
```

This encourages the network to focus on object boundaries rather than texture.

---

### Data Augmentation

During training

- Random horizontal flips
- Random vertical flips

were applied.

During inference

- Test-Time Augmentation (TTA)
- Multiple transformed views
- Prediction averaging

were used to improve robustness.

---

# Model Architecture

The proposed model consists of a **shared-weight Siamese CNN encoder**.

```
Input (2 × 32 × 32)

↓

Conv (2 → 16)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv (16 → 32)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv (32 → 16)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Fully Connected

↓

58-D Embedding
```

All five images pass through the **same encoder**, ensuring a common embedding space.

---

# Outlier Detection

Instead of learning a classifier head, the model predicts the odd image using **average pairwise L2 distances**.

For every embedding

```
e₀, e₁, e₂, e₃, e₄
```

the average distance to the remaining embeddings is computed.

The image with the **largest average distance** is predicted as the outlier.

This parameter-free decision rule closely matches the structure of the problem while reducing overfitting. :contentReference[oaicite:0]{index=0}

---

# Training

Optimizer

- Adam

Loss Function

- Cross Entropy Loss

Learning Rate

- 1e−3

Batch Size

- 64

Embedding Dimension

- 58

Maximum Epochs

- 400

Early Stopping

- Patience = 150

Total Trainable Parameters

- **24,718**

---

# Experiments

Several architectures were evaluated.

- Logistic Regression Baseline
- Siamese CNN with classifier head
- Hybrid CNN with handcrafted features
- Siamese CNN using Group Mean L2
- Siamese CNN using Average Pairwise L2
- Siamese CNN with Triplet Loss
- Final Siamese CNN with Shape Normalization, Sobel Edge Augmentation and Test-Time Augmentation

An extensive ablation study was conducted to quantify the contribution of each design decision. :contentReference[oaicite:1]{index=1}

---

# Results

| Model | Public Kaggle Accuracy |
|------------------------------|----------------:|
| Logistic Regression | 20.9% |
| Siamese CNN + Classifier Head | 23–25% |
| Hybrid CNN | 54–59% |
| Siamese CNN (Group Mean L2) | 62.2% |
| Siamese CNN (Pairwise L2) | 61.8% |
| Siamese CNN + Triplet Loss | 62.1% |
| **Final Model** | **67.10%** |

The final architecture achieved a **67.10% public Kaggle accuracy**, outperforming the logistic regression baseline by more than **46 percentage points** while remaining within the strict **25,000 parameter limit**. :contentReference[oaicite:2]{index=2}

---

# Technologies

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib

---

# Repository Structure

```
.
├── data/
├── figures/
├── models/
├── src/
├── report/
├── requirements.txt
└── README.md
```

---

# Key Learning Outcomes

- Implemented a Siamese CNN using shared weights
- Learned embedding-based metric learning
- Designed a parameter-free distance-based inference mechanism
- Applied image preprocessing and Sobel edge extraction
- Performed ablation studies
- Implemented Test-Time Augmentation
- Built a compact deep learning model under strict parameter constraints

---

