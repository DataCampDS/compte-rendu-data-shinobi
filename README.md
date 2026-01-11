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
