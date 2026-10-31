# Machine Learning Assignment 2 — Classification & Deployment

## a. Problem Statement
Build and evaluate an end-to-end binary classification system to predict whether a breast mass is **Malignant (0)** or **Benign (1)** based on digitized Fine Needle Aspirate (FNA) cell nuclei characteristics, and deploy an interactive model evaluation application on Streamlit Community Cloud.

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Machine Learning Repository / Kaggle)
* **Number of Instances:** 569 (Exceeds minimum requirement of 500 instances)
* **Number of Features:** 30 continuous morphological features (Exceeds minimum requirement of 12 features), including mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
* **Target Variable:** `target` (Binary: `0` = Malignant [212 instances], `1` = Benign [357 instances])
* **Train/Test Split:** 80% Training (455 instances) / 20% Testing (`test_data.csv`, 114 instances, stratified).

## c. GitHub Repository Link
* **Repository URL:** `<INSERT_YOUR_GITHUB_REPO_LINK_HERE>`
* **Live Streamlit App URL:** `<INSERT_YOUR_STREAMLIT_CLOUD_LINK_HERE>`

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.96 | 1.0 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.995 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved top-tier performance (Accuracy: 0.9825, AUC: 0.9954, MCC: 0.9623) because standardized FNA morphological features exhibit strong linear separability in high-dimensional space. |
| **Decision Tree** | Showed the lowest generalization performance among the 5 models (Accuracy: 0.9211, AUC: 0.9163, MCC: 0.8341) due to axis-aligned orthogonal splits struggling with smooth continuous boundaries. |
| **KNN** | Delivered strong accuracy (0.9737) and high recall (1.0) after StandardScaler normalization, effectively capturing local neighborhood clusters. |
| **Naive Bayes** | Achieved 0.9298 accuracy and 0.9868 AUC; slight performance drop compared to linear/kernel models is caused by strong multicollinearity between radius, perimeter, and area features violating the conditional independence assumption. |
| **Support Vector Machine** | Matched the highest overall accuracy (0.9825) and MCC (0.9623) with an AUC of 0.995, as the RBF kernel effectively handles non-linear margins across all 30 features. |
| **Overall, Winner for your dataset?** | **Logistic Regression & Support Vector Machine (Tie for Accuracy/F1/MCC; Logistic Regression edges out on AUC at 0.9954)** — Both achieve 98.25% accuracy, 0.9861 F1-score, and 0.9623 MCC on the held-out test set. |
