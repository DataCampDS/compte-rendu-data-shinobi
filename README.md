# suivi-datacamp
Les points hebdomadaires se feront sur ce README.md

# Week 1 & 2 tasks:

- Each member try to understand the Single Cell RNA Sequencing dataset 
- Divide the task for each member to do data cleaning, descriptive statistics analysis, and data preprocessing.
- Discuss the possbile workflows of the project.

## Setup environment

We have cloned the repo on our local machine and we have also created a virtual environment individually.

```bash
python3 -m venv env

```

## Download single-cell RNA sequencing data

We are going to work on the provided data by download_data.py:

```bash
python download_data.py

```
## Initial workflow

We have devided each members a task in order to progress the process. The First thing we need to do is to do Data Preprocessing and Descriptive Statistics. 

Next thing is to do literature review on existing similar tasks. Then we can work on baseline models for classification like KNN, XGBoost,...etc.

After that we can work on Deep Learning models.  

## Results

We have tried to use first 2 models, Logistic Regression and Random Forest, and we have evaluated themusing balanced accuracy and confusion matrix. 

### Logistic Regression

- Train balanced accuracy: 1.000
- Test balanced accuracy: 0.749

### Random Forest Classifier

- Train balanced accuracy: 1.000
- Test balanced accuracy: 0.655
