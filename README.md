# Credit Card Fraud Detection

This project develops a highly effective machine learning model to detect fraudulent credit card transactions from a dataset characterized by a severe class imbalance (only 0.17% of transactions were fraudulent).

---

## Project Overview
The project focuses on detecting fraud in credit card transactions using gradient boosting algorithms. Key challenges included:

- **Severe class imbalance** (fraudulent transactions are extremely rare).  
- **Outliers** in the `Time` and `Amount` features.  
- **Large dataset size**, which limited local computational resources.

---

## Key Methodologies and Findings

### 1. Robust Preprocessing
- `Time` and `Amount` features were scaled using **RobustScaler**, which handles outliers effectively.  
- The dataset was split into training and testing sets using **stratification** to preserve the fraud-to-non-fraud ratio.

### 2. Addressing Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on **training data only** to prevent data leakage.  
- The test set remained imbalanced to ensure realistic model evaluation.

### 3. Model Selection and Performance
- Two gradient boosting algorithms were trained: **XGBoost** and **LightGBM**.  
- **XGBoost** delivered superior performance, particularly on the minority class (fraud).

**XGBoost Results on Test Data:**
- **Overall Accuracy:** 99.9%  
- **Fraud Recall:** 86% (correctly identified 86 out of 98 actual fraud cases)  
- **Fraud Precision:** 72% (of flagged transactions, 72% were actually fraud)  
- **ROC AUC Score:** 0.98  

> These metrics indicate excellent ability to distinguish between fraudulent and legitimate transactions.

---

## Limitations and Future Work
- Hyperparameter tuning was limited due to **computational constraints**.  
- Further improvements could be achieved with **GridSearchCV** or **Bayesian optimization** on more powerful hardware.  
- Wrapping preprocessing and modeling steps into a **scikit-learn Pipeline** would help productionalize the solution.

---

## Business Impact 📈
- High **fraud recall** allows early detection of fraudulent activities, mitigating financial losses.  
- Strong **precision** ensures minimal impact on legitimate customers.  
- This project serves as a **proof-of-concept** for real-world fraud detection systems.

---

## Dataset Information
- The **original dataset is not included** in this repository due to GitHub file size limitations.  
- You can download the dataset from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Place the file `creditcard.csv` in the **project root directory** before running the notebooks or scripts.

---

## Usage
1. Download the dataset and place it in the project root.  
2. Run the notebooks (`.ipynb`) or scripts (`.py`) to preprocess the data, train models, and evaluate performance.  

---

## Project Files
- `*.ipynb` – Jupyter notebooks with code and analysis  
- `*.py` – Python scripts for preprocessing and modeling  
- `README.md` – Project description  
- `.gitignore` – Ignores large files, virtual environments, and temporary files  

---

## Notes
- All preprocessing, SMOTE application, and model training are included in the notebooks.  
- The project demonstrates end-to-end handling of **imbalanced datasets** and **fraud detection modeling**.
