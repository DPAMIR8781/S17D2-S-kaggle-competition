House Prices Prediction | End-to-End ML Pipeline

This project focuses on predicting house prices using advanced data preprocessing, feature engineering, and machine learning techniques.

It was developed as part of a hands-on Kaggle-style competition and demonstrates a full end-to-end data science workflow.

 Project Overview

The goal is to predict SalePrice using structured housing data by building a robust and scalable machine learning pipeline.

The project includes:

Data cleaning & preprocessing
Feature engineering
Feature selection
Model comparison
Final prediction & submission

Key Concepts Applied
🔹 Feature Engineering
Ordinal encoding for quality-based features
One-Hot Encoding for categorical variables
Cyclical encoding for time-based features (MoSold)
Log transformation of target (SalePrice)
🔹 Feature Selection
SelectPercentile (Mutual Information)
SelectFromModel (Random Forest)
VarianceThreshold
Custom correlation filtering (Pearson)
🔹 Pipeline Design
Built using sklearn.pipeline
Modular and reusable preprocessing structure
Fully integrated transformation + modeling


 Models Used
Model	Type	CV Score (RMSE - log)
Ridge	Linear	~0.146
Random Forest	Tree-based	~0.143

✔ Random Forest achieved the best performance.

 Evaluation Metric
RMSLE (Root Mean Squared Log Error)
Target transformed using log1p
Predictions reverted using expm1
 Project Structure
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission_final.csv
├── tests/
├── notebook.ipynb
└── README.md

 Workflow
Data Loading
Feature Analysis (categorical vs numerical)
Preprocessing Pipeline
Feature Selection
Model Training (CV)
Final Model Selection
Prediction & Submission

Example Pipeline
preproc_final = make_pipeline(
    ColumnTransformer([
        ("num", num_pipeline, feat_numerical),
        ("ord", preproc_ordinal, ordinal_cols),
        ("nom", preproc_nominal, nominal_cols)
    ]),
    SelectPercentile(mutual_info_regression, percentile=50),
    VarianceThreshold()
)


 Results
Baseline Score: ~0.21
Final Score: ~0.14

 Significant improvement through feature engineering and model tuning.

 Learnings
Importance of feature encoding strategies
Pipeline design best practices
Handling multicollinearity and redundancy
Why log transformation matters in skewed targets
Model comparison and evaluation

 Future Improvements
XGBoost / LightGBM integration
Hyperparameter tuning (GridSearch / Optuna)
Advanced feature engineering
Stacking / ensembling models

Author

Doruk Pamir
Aspiring Data Analyst / Data Scientist
