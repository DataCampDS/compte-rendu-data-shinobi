# suivi-datacamp
Les points hebdomadaires se feront sur ce README.md

# Week 1 & 2 tasks:

We have discussed about the workflow of this project by divided for each members tasks to ensure steady progress. We started with Exploratory Data Analysis (Descriptive Statistics,...), and Data Preprocessing. This step is important because it helps us clearly understand the nature of the data and gives us an initial direction for improving performance along the way.

After that, we moved on to feature engineering (normalization, standard scaler, PCA), and now we are experimenting with baseline classification models such as Logistic Regression, Random Forest, KNN, XGBoost, etc. Here, we are trying every model's setup as many as possible that could give us a good performance with minimal time run.  

Following these steps, we will proceed to work on deep learning models if needed.

Ratanakmuny:
- Exploratory Data Analysis
- Data preprocessing
  
Kimmeng:
- Data preprocessing (log normalization)
- Feature Engineering (Standard Scaler, PCA)
- Train and test Logistics Regression

Tito:
- Data preprocessing (normalization)
- Feature Engineering (Standard Scaler, PCA)
- Train and test Random Forest Regression

# Week 3 & 4 tasks:

These two weeks due to many tasks of project from other courses, our progress is a little bit slow. In this stage, we are still working on the modeling by exploring all the possible models that we know and from the assumption that from did EDA. From the modeling that we have done, the accuracy of the baseline model seems to be around 0.8 on the test dataset. 

So, we are working on training new models, apply feature engineerings to see whether the accuracy can be achived higher than the result we have or not.

Our purpose is to find serveral best performance models then try to apply the feature engineering and hyperparamter in the next stage. 

The tasks were divided:

Ratanakmuny:
- Data Preprocessing (Still working on it because the result didn't improve like what we expected it to be)
- Worked on handle the imbalanced class with SMOTE (Not Good Idea!)
- Trained on Logistic, Random Forest, SVM and Gradient Boosting

Kimmeng & Tito:
- Research on High Variance Gene Selection
- Perform High Variance Gene selection
- Encounter overfitting problem when doing this

Result (end of Week 4):
After Week 4, we observed that our baseline Logistic Regression achieved around 0.8 accuracy on the test dataset. However, when we applied High Variance Gene Selection, the models started to overfit, and the performance on our local validation/training setup dropped below 0.8. This suggested that the added complexity was not helping and that something in the pipeline (preprocessing, splitting, or feature selection strategy) might be causing instability.

# Week 5 & 6 tasks:

After obtaining worse results than the baseline in Week 4, our main objective became: **identify why the more advanced models performed worse than the baseline**.

The tasks were divided:

Ratanakmuny:
- Investigate and validate the data preprocessing pipeline (since the improvements did not match expectations)
- Check whether preprocessing choices could explain the performance drop

Tito:
- Study reasonable ways to improve the current models without overcomplicating the pipeline
- Focus on approaches that improve performance while keeping runtime acceptable

Kimmeng:
- Work on a stacking classification to combine several good baseline models.

# Week 7 until the end of the challenge

From Week 7 onward, we shifted into an optimization phase. All team members are now focused on improving performance by iterating on the best candidate models, applying feature engineering and hyperparameter tuning, and selecting a strong final strategy with the goal of reaching the top ranking.


---

# Introduction

In this project, we are working on classification of cell-types of the scMARK dataset. There exist 1000 observations on training dataset and 500 observation on testing dataset, but the main problem is that this is a high-dimensional dataset where the number of features is higher than the number of observations. Additionally, the number of each cell-type is also imbalanced with T_cells_CD8+,T_cells_CD4+, Cancer_cells and NK_cells, 342, 336, 237 and 85 respectively.

The goal is to train models that are best for classification, specifically cell-types classification with high-dimensional and imbalance dataset.

# Data preprocessing

Before training our models, we started with data preprocessing below, follow by the problem and the solutions: 

- Large variability in total gene expression across cells -> normalization is necessary.
- Gene expression is highly right-skewed with outliers -> log transformation is applied.
- Most genes show low variance -> select highly variable genes to reduce noise.
- Keeping only variable genes preserves informative biological signals.
- High dimensionality and correlated genes -> feature standardization is required.
- Strong redundancy and rapid variance capture -> PCA is used for dimensionality reduction.

# Preprocessing Pipeline

Preprocessing pipelines are evaluated progressively, starting from raw data and gradually incorporating normalization, variance stabilization, feature selection, scaling, and dimensionality reduction to assess their impact on model performance.

* Pipeline 1: Raw baseline
    - No preprocessing.
    - Used as a reference to measure the benefit of all transformations.
* Pipeline 2: Normalization only
    - Cells are normalized by their total expression to correct for sequencing depth differences.
* Pipeline 3: Normalization + log transform
    - Log transformation is added to reduce skewness and stabilize variance after normalization.
* Pipeline 4: Normalization + log transform + HVG selection
    - Feature space is reduced by keeping only highly variable genes, removing uninformative genes.
* Pipeline 5: Normalization + log transform + HVG selection + scaling
    - Features are standardized to ensure comparable scales, which is important for linear models and PCA.
* Pipeline 6: Normalization + log transform + HVG selection + scaling + PCA
    - Dimensionality is further reduced using PCA to remove redundancy and retain the main sources of variation.

# Models Evaluation

We have worked with a lot of models like Logistic Regression, Linear SVM, Gaussian Navies Bayes, Random Forest, KNN, Decision Tree, Extra Tree, GradientBoosting, HistGradient, XGBoosting, AdaBoost, LightGBM, and ensemble methods like Bagging, Voting and Stacking.

The top 3 highest performances among those models are:
1. Ensemble model of Stacking:
    - Balanced accuracy: 0.86
    - Training and validation time: 137.802195s and 3.352885s
    - Base models: Bagging, HistGradient and LightGBM
    - Meta model: Logistic regression
    - Pipeline: 

2. LightGBM:
    - Balanced accuracy: 0.86
    - Training and validation time: 43.036889s and 1.301543s
    - Pipeline 4: Normalization + log transform + HVG selection

3. Gradient Boosting:
    - Balanced accuracy: 0.85
    - Training and validation time: 590.340776s and	1.214318s
    - Pipeline 4: Normalization + log transform + HVG selection

From the training and testing scores before the submission, we notice that Gradient Boosting and LightGBM perform worse after scaling and PCA because tree-based models do not benefit from these transformations and can lose important information when they are applied.

# Ranking


