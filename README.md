
# Handcrafted Feature-Based Object Classification on the COIL-20 Dataset

## Overview

This project implements a classical computer vision pipeline for multi-class object classification using the COIL-20 dataset. 
Instead of deep learning approaches, the system relies on handcrafted feature extraction and traditional machine learning techniques.

The implemented pipeline:

HOG → Feature Scaling → PCA / LDA → Linear SVM → Evaluation

The objective is to demonstrate that well-designed classical methods can still achieve strong performance on structured image datasets.

---

## Dataset

- Dataset: COIL-20 (Columbia Object Image Library)
- Total Images: 1440
- Classes: 20 objects
- Images per Class: 72 (5-degree rotation increments)
- Image Resolution: 128 × 128

Each image is labeled based on object identity and rotation angle.

---

## Methodology

### 1. Preprocessing
- Convert images to grayscale
- Normalize pixel intensities to [0,1]
- Split dataset into 80% training and 20% testing

### 2. Feature Extraction
- Histogram of Oriented Gradients (HOG)
- 9 orientation bins
- 8×8 pixels per cell
- 2×2 cells per block
- L2-Hys normalization

### 3. Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

### 4. Classification
- Linear Support Vector Machine (SVM)
- One-vs-rest multi-class strategy

---

## Results

| Method        | Accuracy |
|---------------|----------|
| PCA + SVM     | 93.9%    |
| LDA + SVM     | 96.1%    |

LDA slightly outperforms PCA due to its supervised nature and class-discriminative projections.

---

## Repository Structure

- dataset_loader.py – Dataset loading and label extraction
- preprocessing.py – Image normalization and train-test split
- hog_extraction.py – HOG feature computation
- pca_pipeline.py – PCA-based classification pipeline
- lda_pipeline.py – LDA-based classification pipeline
- main.py – Complete pipeline execution script
- COIL20_Handcrafted_Project.ipynb – Google Colab notebook version

---

## How to Run

Install dependencies:

pip install numpy opencv-python scikit-image scikit-learn matplotlib

Update the dataset path inside main.py and run:

python main.py

---

## Conclusion

This project shows that classical handcrafted feature-based approaches remain highly effective on controlled datasets such as COIL-20. 
The combination of HOG features, dimensionality reduction, and linear SVM provides strong classification performance with minimal computational cost.

---

Department of Computer Science and Engineering  
SSN College of Engineering, Chennai, India
