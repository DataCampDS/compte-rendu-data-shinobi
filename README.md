# suivi-datacamp
Les points hebdomadaires se feront sur ce README.md

# Week 1 & 2 tasks:
## Setup environment

We have cloned the repo on our local machine and we have also created a virtual environment individually.

```bash
python3 -m venv env

```

## Download single-cell RNA sequencing data

We worked on the provided data by download_data.py:

```bash
python download_data.py

```
## Initial workflow

We have discussed about the workflow that we should follow for this project. We have devided each members a task in order to progress the process. The First thing we need to do is to do Data Preprocessing and Descriptive Statistics. 
Then we do feature engineering (normalization, standard scaler, PCA), and we can work on baseline models for classification like Logistic Regression, Random Forest, KNN, XGBoost,...etc.
After that, we will work on deep learning models.

## Each member tasks:

Ratanakmuny:
- Data cleaning
- descriptive statistics
  
Kimmeng:
- Data preprocessing (log normalization)
- Feature Engineering (Standard Scaler, PCA)
- Train and test Logistics Regression

Tito:
- Data preprocessing (normalization)
- Feature Engineering (Standard Scaler, PCA)
- Train and test Random Forest Regression

We have uploaded the notebook  "model_training.ipynb" for data preprocessing and model training.
