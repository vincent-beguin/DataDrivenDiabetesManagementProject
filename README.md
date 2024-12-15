# DataDrivenDiabetesManagementProject
Semester Project for the course Data Driven Diabetes Management. Group of Amir Meymandinezhad, Hamidreza Madi and Vincent Béguin. Project number 8

# Introduction
This project aims to evaluate different Machine learning methodologies to predict Type II diabetes from normal routine health data among Pima Indian women over 21 who've already had a pregnancy. The Pima Indians Diabetes Database (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) was used as training dataset.

# Methodology
six machine learning models were used:
* Logistic Regression
* Random forest
* FNN
* XGBoost
* CatBoost
* LightGBM

Preprossessing was applied when necessary. The first three models were used with data imputation and tested with and without normalisation and SMOTE. The latter three models didn't require any preprocessing.

# Results
| Model                      | Processing                                              | AUC  | Accuracy | F1-Score |
|----------------------------|---------------------------------------------------------|------|----------|----------|
| logistic regression        | Full Preprocessing (Imputation + Normalization + SMOTE) | 0.84 | 0.76     | 0.68     |
|                            | With Partial Preprocessing (Imputation Only)            | 0.84 | 0.77     | 0.65     |
| Random forest              | Full Preprocessing (Imputation + Normalization + SMOTE) | 0.84 | 0.76     | 0.63     |
|                            | With Partial Preprocessing (Imputation Only)            | 0.82 | 0.76     | 0.67     |
| FNN                        | Full Preprocessing (Imputation + Normalization + SMOTE) | 0.84 | 0.76     | 0.70     |
|                            | With Partial Preprocessing (Imputation Only)            | 0.75 | 0.70     | 0.42     |
| XGBoost                    | Without Preprocessing (Raw Data)                        | 0.84 | 0.75     | 0.63     |
| CatBoost                   | Without Preprocessing (Raw Data)                        | 0.84 | 0.77     | 0.64     |
| LightGBM                   | Without Preprocessing (Raw Data)                        | 0.85 | 0.73     | 0.46     |
| Ensemble gradient boosting | Without Preprocessing (Raw Data)                        | 0.84 | 0.76     | 0.63     |
