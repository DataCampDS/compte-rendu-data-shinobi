Notes finales : 

- Tito : 15.5/20
- Ratanakmuny : 16.5/20
- Kimmeng : 15/20


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

This project focuses on **cell-type classification** using the scMARK single-cell RNA-seq dataset. The goal is to predict one of four cell types from gene expression counts:

- Cancer_cells  
- NK_cells  
- T_cells_CD4+  
- T_cells_CD8+  

The training set contains **1000 cells** and the test set contains **500 cells**, with **13551 genes** measured per cell. This creates a challenging setting where the number of features is much larger than the number of observations (high-dimensional data).

In addition, the classes are **imbalanced** in the training set:
- T_cells_CD8+: 342  
- T_cells_CD4+: 336  
- Cancer_cells: 237  
- NK_cells: 85  

Because scRNA-seq data is sparse, noisy, and high-dimensional, the main objective is to build a robust machine learning pipeline that improves classification performance while keeping runtime acceptable. Our work explores preprocessing (normalization, log transform, highly variable gene selection, scaling, PCA) and a wide range of classical machine learning models and ensemble strategies, evaluated using **balanced accuracy**.


# Exploration Data Analysis

You can view our EDA notebook here: [exploratory_data_analysis.ipynb](https://github.com/DataCampDS/scmark-classification-data-shinobi/blob/main/exploratory_data_analysis.ipynb).

### Dataset overview

* **Task:** supervised classification of 4 cell types from scRNA-seq gene count data. 
* **Shapes and formats:**

  * `X_train`: CSR sparse matrix, **(1000, 13551)** genes 
  * `X_test`: CSR sparse matrix, **(500, 13551)** genes 
  * `y_train`: categorical labels with 4 classes, **no missing labels** 

### Class distribution (imbalance)

* Train label counts and proportions: 

  * **T_cells_CD8+**: 342 (0.342)
  * **T_cells_CD4+**: 336 (0.336)
  * **Cancer_cells**: 237 (0.237)
  * **NK_cells**: 85 (0.085)
* Takeaway: the dataset is **imbalanced**, mainly due to the smaller NK_cells class. 

### Cell-level QC and sparsity (why preprocessing is needed)

Computed per cell: total counts, number of detected genes, and percent zeros. Summary stats: 

* **Total counts per cell**

  * mean: ~3334, median: ~1995, max: 37679
* **Detected genes per cell**

  * mean: ~1091, median: ~840, max: 5628
* **Sparsity per cell (% zeros)**

  * mean: ~91.95%, median: ~93.80%

Sparsity patterns: 

* Most cells have **~90% to 98% zeros**, meaning each cell expresses only a small fraction of genes.
* This is expected for scRNA-seq, but it makes modeling harder and motivates **normalization** and **dimension reduction / feature selection**.

### Gene-level sparsity and variance (why HVG helps)

* For genes, the majority have **~95% to 100% zeros**, meaning many genes are rarely expressed and likely uninformative for classification. 
* Gene variance distribution is **highly skewed**: a small subset of genes has very high variance, while most genes are low-variance. 
* Ranked variances (log-scale) show a sharp drop, and cumulative variance indicates **diminishing returns** when adding more genes, supporting a **Highly Variable Gene (HVG)** selection approach. 

### Total expression distribution (normalization signal)

* Total counts per cell are strongly **right-skewed**:

  * Most cells are around **500 to 3000** counts
  * A minority forms a long tail up to **~35,000** counts 
* Takeaway: sequencing depth varies a lot across cells, so **normalization is necessary** before PCA and classification. 

### Low-dimensional visualization (separability check)

* **PCA (2 components)** on standardized data shows **partial separation**: 

  * Cancer_cells form a broader cluster with high spread along PC1
  * NK_cells appear more compact and shifted relative to Cancer_cells
  * T_cells_CD4+ and T_cells_CD8+ overlap more (biologically similar), making them harder to separate
* **UMAP** embedding was also computed to visualize structure (used as an additional qualitative check). 

### Correlation structure between genes (redundancy evidence)

* Correlation heatmap on the **top 200 most variable genes** shows clear **block structures** (co-expression modules) and anti-correlated regions. 
* Takeaway: many genes are redundant, reinforcing the value of **PCA** and/or careful feature selection. 

### EDA conclusions that guided the pipeline

From the observed properties of the data: 

* Data is **ultra sparse** and **high-dimensional** (1000 × 13551).
* Many genes are uninformative (mostly zeros, low variance).
* Total counts vary strongly across cells, so **normalization** is needed.
* PCA and UMAP show meaningful structure (signal exists), but CD4/CD8 separation is intrinsically harder.
* Strong correlation blocks suggest redundancy, motivating **HVG selection** and **dimensionality reduction**.

### Suggested next steps from EDA

* Normalize counts across cells (sequencing depth correction). 
* Apply log transform to stabilize skewness. 
* Select HVGs (for example top 500 to 2000 genes) and/or apply PCA before classification. 
* Handle imbalance using **class weights** first; oversampling such as SMOTE was considered but may generate unrealistic samples for gene expression. 

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

* Pipeline 0: Raw baseline
    - No preprocessing.
    - Used as a reference to measure the benefit of all transformations.
* Pipeline 1: Normalization only
    - Cells are normalized by their total expression to correct for sequencing depth differences.
* Pipeline 2: Normalization + log transform
    - Log transformation is added to reduce skewness and stabilize variance after normalization.
* Pipeline 3: Normalization + log transform + HVG selection
    - Feature space is reduced by keeping only highly variable genes, removing uninformative genes.
* Pipeline 4: Normalization + log transform + HVG selection + scaling
    - Features are standardized to ensure comparable scales, which is important for linear models and PCA.
* Pipeline 5: Normalization + log transform + HVG selection + scaling + PCA
    - Dimensionality is further reduced using PCA to remove redundancy and retain the main sources of variation.

# Models Evaluation

You can view our modeling notebook here: [model_training.ipynb](https://github.com/DataCampDS/scmark-classification-data-shinobi/blob/main/model_training_before_submission.ipynb).

We have worked with a lot of models like Logistic Regression, Linear SVM, Gaussian Navies Bayes, Random Forest, KNN, Decision Tree, Extra Tree, GradientBoosting, HistGradient, XGBoosting, AdaBoost, LightGBM, and ensemble methods like Bagging, Voting and Stacking.

The top 3 highest performances among those models are:

1. LightGBM:
    - Balanced accuracy: 0.86
    - Training and validation time: 43.036889s and 1.301543s
    - Pipeline 3: Normalization + log transform + HVG selection

2. Ensemble model of Stacking:
    - Balanced accuracy: 0.86
    - Training and validation time: 137.802195s and 3.352885s
    - Base models: Bagging, HistGradient and LightGBM
    - Meta model: Logistic regression
    - Pipeline 3: Normalization + log transform + HVG selection

3. Gradient Boosting:
    - Balanced accuracy: 0.85
    - Training and validation time: 590.340776s and	1.214318s
    - Pipeline 3: Normalization + log transform + HVG selection

From the training and testing scores before the submission, we notice that Gradient Boosting and LightGBM perform worse after scaling and PCA because tree-based models do not benefit from these transformations and can lose important information when they are applied.

# Improvement

You can view our improvement modeling notebook here: [model_training_improvement.ipynb](https://github.com/DataCampDS/scmark-classification-data-shinobi/blob/main/model_training_improvement.ipynb).

After experimenting with our initial preprocessing pipelines (P0 to P5) and multiple models, we observed a clear pattern:

- **Logistic Regression performed best with P4** (HVG + scaling), which suggests the linear model is sensitive to feature scaling.
- **LightGBM performed best with P3** (HVG without scaling), which is expected since tree-based models do not benefit from standardization.
- However, even when we changed models or added feature engineering (PCA, additional transforms), the performance tended to plateau around the same range. This indicated that the limitation was not only the classifier, but also the **representation of the gene expression matrix**.

Because of that, we took a step back and explored a different interpretation of the data.

## New idea: Treat gene expression as a "bag of genes" problem

We reframed the single-cell matrix in a way similar to text classification:

- Each **cell** is treated as a **document**
- Each **gene** is treated as a **word**
- Each **count** is treated as a **word frequency**

This viewpoint motivated two changes:
1. Use a fast **unsupervised gene selection** method to keep only informative genes (HVG-like selection).
2. Use **TF-IDF weighting** to downweight genes that appear in many cells and upweight genes that are more specific to certain cell types.

This approach improves class separation because:

- Dispersion-based selection keeps genes that vary meaningfully across cells, reducing noise and runtime.
- TF-IDF reduces the impact of genes expressed in many cell types (common genes) and highlights genes that are more specific to one class.
- Logistic Regression is efficient and effective on sparse high-dimensional TF-IDF features.

This combination led to a significant improvement in balanced accuracy and faster runtime compared to our earlier pipelines.

## Result

- Balanced accuracy: 0.88
- Training and validation time: 3.645396s and 0.155921s

# Ranking

Based on the current public leaderboard:

| Rank | Team        | Best balanced accuracy (bal_acc) | Train time (s)  | Validation time (s) |
|------|-------------|----------------------------------|-----------------|---------------------|
| 1    | Team Avenger| 0.89                             | 41.145948       | 8.252714            |
| 2    | Team A      | 0.88                             | 13.048182       | 0.953971            |
| 3    | Team Shinobi (our team) | 0.88                   | 3.645396        | 0.155921            |

# Conclusion

Despite trying our best to test every models and finetuning for the best parameter, we are still at the back compare to other team. We should have focused around the strategy of the modeling by anaylzing the result of the confusion matrix which cells that models confused.

However, our cost it seems to be less than other team. As data science students, our perspective is trying to get the best performance out of the model with the affordable cost. By that we mean, the performance and the runtime has to balance with each other.

If we have more times, I think we can use more techniques:
- **Specialist models for hard pairs:** Most remaining errors come from biologically similar classes (especially **NK vs T_cells_CD8+**, and sometimes **T_cells_CD4+ vs T_cells_CD8+**). A practical extension is to add a second-stage binary classifier for these pairs and apply it only on uncertain predictions.

- **Supervised feature selection after TF-IDF:** After dispersion-based gene selection, we can apply a lightweight supervised filter (for example chi-square) on TF-IDF features to keep the most discriminative genes for the labels.

- **Calibration and confidence analysis:** Improve decision rules by calibrating probabilities and using confidence-based thresholds (entropy or max probability) to reduce misclassification on ambiguous cells.

- **Ensembling with minimal overhead:** Combine the final Logistic Regression model with one additional fast model (for example Multinomial Naive Bayes on TF-IDF) by averaging predicted probabilities, aiming to improve recall for the weakest class without increasing runtime too much.

- **More robust evaluation:** Use cross-validation and systematic error analysis (per-class recall stability) to confirm that improvements generalize beyond a single split.

