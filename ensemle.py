# Data manipulation and preprocessing
import pandas as pd
import numpy as np

# Machine learning models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

# Model evaluation and splitting
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score

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

# Initialize base learners with the best hyperparameters
xgb = XGBClassifier(
    colsample_bytree=0.8,
    learning_rate=0.01,
    max_depth=3,
    n_estimators=300,
    subsample=0.8,
    random_state=42,
    eval_metric='logloss'
)

lgb = LGBMClassifier(
    subsample=0.6,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    colsample_bytree=0.6,
    random_state=42
)

cat = CatBoostClassifier(
    depth=4,
    iterations=300,
    learning_rate=0.05,
    random_state=42,
    verbose=0
)

# Ensemble Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat)],
    voting='soft'
)

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_auc = []
fold_f1 = []
fold_accuracy = []

# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Perform 5-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    print(f"\nProcessing Fold {fold + 1}...")

    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Train Ensemble Model
    ensemble_model.fit(X_train, y_train)

    # Predict probabilities and calculate metrics
    y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)  # Generate binary predictions
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)  # Calculate F1 score
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy

    fold_auc.append(roc_auc)
    fold_f1.append(f1)
    fold_accuracy.append(accuracy)

    # Plot ROC curve for each fold
    plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

# Finalize the ROC plot
mean_auc = np.mean(fold_auc)
mean_accuracy = np.mean(fold_accuracy)
mean_f1 = np.mean(fold_f1)

plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"Ensemble Gradient Boosting ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Boxplot of AUC scores
plt.figure(figsize=(6, 4))
plt.boxplot(fold_auc, vert=False)
plt.title("AUC Scores Across Folds - Ensemble Gradient Boosting")
plt.xlabel("AUC Score")
plt.show()

print(f"\nMean AUC across all folds: {mean_auc:.4f}")
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
print(f"Mean F1-score across folds: {mean_f1:.4f}")
