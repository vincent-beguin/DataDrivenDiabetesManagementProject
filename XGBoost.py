# Data manipulation and preprocessing
import pandas as pd
import numpy as np

# XGBoost model
from xgboost import XGBClassifier

# Model selection and evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score

# Visualization
import matplotlib.pyplot as plt


# Load the dataset
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Replace zeros with NaN for missing values in relevant columns
columns_to_replace_zeros = ['Age', 'BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'Glucose', 'BloodPressure']
data[columns_to_replace_zeros] = data[columns_to_replace_zeros].replace(0, np.nan)

# Splitting into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_auc = []
fold_f1 = []
fold_accuracy = [] 


# Best parameters storage
best_params = None
best_auc = 0

# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing Fold {fold + 1}...")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Select the best model and parameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    print(f"Best parameters for Fold {fold + 1}: {grid_search.best_params_}")

    # Evaluate the model
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)  # Add this line


    fold_auc.append(roc_auc)
    fold_f1.append(f1)
    fold_accuracy.append(accuracy)  # Store accuracy


    if roc_auc > best_auc:
        best_auc = roc_auc
        best_params = grid_search.best_params_

    # Plot ROC curve for each fold
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

# Finalize the ROC plot
mean_auc = np.mean(fold_auc)
mean_f1 = np.mean(fold_f1)
mean_accuracy = np.mean(fold_accuracy)

plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"XGBoost ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Boxplot of AUC scores
plt.figure(figsize=(6, 4))
plt.boxplot(fold_auc, vert=False)
plt.title("AUC Scores Across Folds - XGBoost")
plt.xlabel("AUC Score")
plt.show()

print(f"\nMean AUC across all folds: {mean_auc:.4f}")
print(f"Mean F1-score across all folds: {mean_f1:.4f}")
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
print(f"Best Overall Hyperparameters: {best_params}")
