# Data manipulation
import pandas as pd
import numpy as np

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Oversampling
from imblearn.over_sampling import SMOTE

# Model evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score

# Visualization
import matplotlib.pyplot as plt

#logistic regression all

# Load the dataset
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Replace zeros with NaN for imputation in relevant columns
columns_to_replace_zeros = ['Age', 'BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
data[columns_to_replace_zeros] = data[columns_to_replace_zeros].replace(0, np.nan)

# Splitting into features and target
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_auc = []
fold_f1 = []
fold_accuracy = []  


# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Perform 5-fold cross-validation
best_params = None
best_mean_auc = -1

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing Fold {fold + 1}...")

    # Split the data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Iterative imputation 
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    X_train = iterative_imputer.fit_transform(X_train)
    X_test = iterative_imputer.transform(X_test)


    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0.01, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=3,
        scoring='roc_auc'
    )
    grid_search.fit(X_train, y_train)

    # Train Logistic Regression Model with best parameters
    best_logistic_model = grid_search.best_estimator_
    best_logistic_model.fit(X_train, y_train)

    # Predict probabilities and calculate metrics
    y_pred_proba = best_logistic_model.predict_proba(X_test)[:, 1]
    y_pred = best_logistic_model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    fold_auc.append(roc_auc)
    fold_f1.append(f1)
    fold_accuracy.append(accuracy)


    # Plot ROC curve for each fold
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

    # Track the best model across folds
    if np.mean(fold_auc) > best_mean_auc:
        best_mean_auc = np.mean(fold_auc)
        best_params = grid_search.best_params_

# Finalize the ROC plot
mean_auc = np.mean(fold_auc)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"Logistic Regression ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Boxplot of AUC scores
plt.figure(figsize=(6, 4))
plt.boxplot(fold_auc, vert=False)
plt.title("AUC Scores Across Folds - Logistic Regression")
plt.xlabel("AUC Score")
plt.show()

print(f"\nMean AUC across all folds: {mean_auc:.4f}")
print(f"Mean F1-Score across all folds: {np.mean(fold_f1):.4f}")
mean_accuracy = np.mean(fold_accuracy)
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
print(f"Best Hyperparameters: {best_params}")
