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
