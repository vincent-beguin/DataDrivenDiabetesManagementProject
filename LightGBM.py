# Data manipulation and preprocessing
import pandas as pd
import numpy as np

# LightGBM model
import lightgbm as lgb

# Model selection and evaluation
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score

# Visualization
import matplotlib.pyplot as plt


# Load the dataset
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Replace zeros with NaN for missing values in relevant columns
columns_to_replace_zeros = ['Age', 'BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'Glucose', 'BloodPressure']
data[columns_to_replace_zeros] = data[columns_to_replace_zeros].replace(0, np.nan)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data[columns_to_replace_zeros] = imputer.fit_transform(data[columns_to_replace_zeros])

# Splitting into features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Define hyperparameter grid
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 9],
    "n_estimators": [100, 300, 500],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

# Perform hyperparameter tuning
print("Performing hyperparameter tuning...")
grid_search = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=3,  # This is only for hyperparameter tuning
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X, y)

# Select the best hyperparameters
best_params = grid_search.best_params_
print(f"\nBest hyperparameters found during tuning: {best_params}")

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_auc = []
fold_f1 = []
fold_accuracy = []

# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Perform 5-Fold Cross-Validation with the best hyperparameters
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing Fold {fold + 1}...")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Train LightGBM model using the best hyperparameters
    model = lgb.LGBMClassifier(
        random_state=42,
        **best_params
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # Evaluate the model
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)  # Add this line


    fold_auc.append(roc_auc)
    fold_f1.append(f1)
    fold_accuracy.append(accuracy)  # Store accuracy


    # Plot ROC curve for each fold
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

# Finalize the ROC plot
mean_auc = sum(fold_auc) / len(fold_auc)
#mean_f1 = sum(fold_f1) / len(fold_f1)
mean_f1 = np.mean(fold_f1)
mean_accuracy = np.mean(fold_accuracy)

plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"LightGBM ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print best parameters and metrics
print(f"\nBest Parameters: {best_params}")
print(f"\nMean AUC across folds: {mean_auc:.4f}")
print(f"Mean F1-score across folds: {mean_f1:.4f}")
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
