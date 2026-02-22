Handcrafted Feature-Based Object Classification on the COIL-20 Dataset
📌 Project Overview

This project implements a classical machine learning pipeline for object classification using the COIL-20 dataset. Instead of deep learning, we use handcrafted features combined with dimensionality reduction and a linear classifier.

The complete pipeline:

HOG → Feature Scaling → PCA / LDA → Linear SVM → Evaluation

The goal is to demonstrate that traditional computer vision techniques can still achieve high accuracy on controlled datasets like COIL-20.

📂 Repository Structure
.
├── COIL20_Handcrafted_Project.ipynb
├── coil-20-proc.zip
├── dataset_loader.py
├── preprocessing.py
├── hog_extraction.py
├── pca_pipeline.py
├── lda_pipeline.py
├── main.py
└── README.md
File Description

dataset_loader.py
Loads images from the dataset directory and extracts labels from filenames.

preprocessing.py
Handles grayscale conversion, normalization, and train-test splitting.

hog_extraction.py
Extracts Histogram of Oriented Gradients (HOG) features from images.

pca_pipeline.py
Implements PCA-based dimensionality reduction and SVM classification.

lda_pipeline.py
Implements LDA-based dimensionality reduction and SVM classification.

main.py
Executes the full pipeline and prints final results.

COIL20_Handcrafted_Project.ipynb
Google Colab version of the full experiment.

coil-20-proc.zip
Processed COIL-20 dataset (if included).

🗂 Dataset Information

Dataset: COIL-20
Total Images: 1440
Classes: 20 objects
Images per Class: 72
Resolution: 128 × 128

Each image filename follows:

objX__Y.png

Where:

X → object number (class label)

Y → rotation index

⚙️ Methodology
1️⃣ Preprocessing

Convert images to grayscale

Normalize pixel values to [0,1]

Split dataset into:

80% Training (1152 images)

20% Testing (288 images)

2️⃣ Feature Extraction – HOG

We compute:

9 orientation bins

8×8 pixels per cell

2×2 cells per block

L2-Hys normalization

Final HOG feature vector size: 8100 dimensions

HOG captures edge direction and local shape structure, which is critical for object recognition.

3️⃣ Dimensionality Reduction
🔹 PCA

Retains 95% variance

~150 principal components

Unsupervised method

🔹 LDA

Uses class labels

Maximum 19 components (20 classes → C−1)

Maximizes class separability

4️⃣ Classification – Linear SVM

We train a one-vs-rest linear SVM:

min
⁡
𝑤
,
𝑏
1
2
∣
∣
𝑤
∣
∣
2
+
𝐶
∑
𝜉
𝑖
w,b
min
	​

2
1
	​

∣∣w∣∣
2
+C∑ξ
i
	​


SVM finds the optimal separating hyperplane in the reduced feature space.

📊 Results
Method	Accuracy
PCA + SVM	93.9%
LDA + SVM	96.1%
Key Observations

LDA performs slightly better because it uses label information.

Most errors occur between visually similar objects.

Classical methods are highly efficient (training under seconds).

Demonstrates strong performance without deep learning.

▶️ How to Run
Option 1 – Run Locally

Install dependencies:

pip install numpy opencv-python scikit-image scikit-learn matplotlib

Update dataset path inside main.py

Run:

python main.py
Option 2 – Run on Google Colab

Open the notebook:

COIL20_Handcrafted_Project.ipynb

Mount Google Drive and update dataset path:

dataset_path = '/content/drive/MyDrive/dataset'

Run all cells.

🧠 Key Takeaways

Handcrafted features remain powerful on controlled datasets.

HOG effectively captures object shape information.

LDA significantly improves class separation.

Linear SVM is sufficient in reduced feature space.

Classical pipelines are computationally lightweight.

🚀 Future Work

Compare with CNN-based deep learning models

Experiment with other descriptors (SURF, SIFT)

Perform cross-validation instead of single split

Analyze robustness to noise and occlusion

👨‍💻 Authors

Chandru S

Harish R

Sri Krishna O S (Corresponding Author)
Department of Computer Science and Engineering
SSN College of Engineering, Chennai, India
