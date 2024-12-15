# Data manipulation
import pandas as pd
import numpy as np

# Imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Oversampling
from imblearn.over_sampling import SMOTE

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score

# Visualization
import matplotlib.pyplot as plt

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
fold_auc = []  # Store AUC for each fold
fold_f1 = []   # Store F1-score for each fold
fold_accuracy = []  # To store accuracy for each fold


# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_params = None  # To store the best parameters

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing Fold {fold + 1}...")

    # Split the data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Iterative Imputation for all features
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

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Best parameters for fold {fold + 1}: {best_params}")

    # Train the model with best parameters
    rf_model = RandomForestClassifier(random_state=42, **best_params)
    rf_model.fit(X_train, y_train)

    # Predict probabilities and calculate AUC
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = rf_model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracy.append(accuracy)
    fold_auc.append(roc_auc)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred)
    fold_f1.append(f1)

    # Plot ROC curve for each fold
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

# Finalize the ROC plot
mean_auc = np.mean(fold_auc)
mean_f1 = np.mean(fold_f1)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"Random Forest ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Boxplot of AUC scores
plt.figure(figsize=(6, 4))
plt.boxplot(fold_auc, vert=False)
plt.title("AUC Scores Across Folds - Random Forest")
plt.xlabel("AUC Score")
plt.show()

print(f"\nMean AUC across all folds: {mean_auc:.4f}")
print(f"Mean F1-score across all folds: {mean_f1:.4f}")
mean_accuracy = np.mean(fold_accuracy)
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
print(f"Best Hyperparameters: {best_params}")
