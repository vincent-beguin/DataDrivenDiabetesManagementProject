# DataDrivenDiabetesManagementProject
Semester Project for the course Data Driven Diabetes Management. Group of Amir Meymandinezhad, Hamidreza Madi and Vincent Béguin. Project number 8

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
