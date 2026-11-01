# Machine Learning Assignment 2 — Classification & Deployment

## a. Problem Statement

Build and evaluate an end-to-end binary classification system to predict whether a breast mass is **Malignant (0)** or **Benign (1)** based on digitized Fine Needle Aspirate (FNA) cell nuclei characteristics, and deploy an interactive model evaluation application on Streamlit Community Cloud.

## b. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Machine Learning Repository / Kaggle)
- **Number of Instances:** 569 (exceeds the minimum requirement of 500 instances)
- **Number of Features:** 30 continuous morphological features (exceeds the minimum requirement of 12 features), including mean, standard error, and worst values of radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- **Target Variable:** `target` (Binary: `0` = Malignant [212 instances], `1` = Benign [357 instances])
- **Train/Test Split:** 80% Training (455 instances) / 20% Testing (`test_data.csv`, 114 instances, stratified).

## c. GitHub Repository Link

- **Repository URL:** https://github.com/RamanathanSks/2026ak05475_ml_assignment_2
- **Live Streamlit App URL:** `<INSERT_YOUR_STREAMLIT_CLOUD_LINK_HERE>`

## d. Models Used & Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9737 | 0.9960 | 0.9859 | 0.9722 | 0.9790 | 0.9439 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Very strong (Accuracy 0.9737, AUC 0.9960 — the highest AUC of all models, MCC 0.9439). Standardized FNA features are close to linearly separable, so a linear boundary works well. It made only 3 errors (1 false positive, 2 false negatives) on 114 test samples. |
| **Decision Tree** | Weakest model (Accuracy 0.9211, AUC 0.9163, MCC 0.8341) with 9 errors, 6 of them benign cases predicted as malignant. A depth-5 tree uses axis-aligned splits, which approximate the smooth class boundary poorly and generalize less well. |
| **KNN** | Strong (Accuracy 0.9737, MCC 0.9442) with perfect recall (1.0000): it missed no benign case, but produced 3 false positives. It benefits from standardization, since distance-based methods are scale sensitive. |
| **Naive Bayes** | Moderate (Accuracy 0.9298, MCC 0.8492) but a high AUC of 0.9868, so its probability ranking is good while its hard decisions are weaker. Radius, perimeter and area are strongly correlated, which violates the conditional-independence assumption. |
| **Support Vector Machine** | Best overall (Accuracy 0.9825, F1 0.9861, MCC 0.9623) with only 2 errors (1 false positive, 1 false negative). The RBF kernel captures non-linear structure in the 30-dimensional feature space. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — highest Accuracy (0.9825), Precision, Recall, F1 (0.9861) and MCC (0.9623) on the held-out test set, with AUC (0.9950) just behind Logistic Regression (0.9960). Logistic Regression and KNN follow (Accuracy 0.9737). |
