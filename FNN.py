# Data manipulation
import pandas as pd
import numpy as np

# Deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Preprocessing and imputation
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Oversampling
from imblearn.over_sampling import SMOTE

# Model evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score

# Visualization
import matplotlib.pyplot as plt


# Define the Feedforward Neural Network (FNN)
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.4):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Load dataset
file_path = "diabetes.csv"
data = pd.read_csv(file_path)

# Replace zeros with NaN for imputation in relevant columns
columns_to_replace_zeros = ['Age', 'BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
data[columns_to_replace_zeros] = data[columns_to_replace_zeros].replace(0, np.nan)

X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# Initialize imputation strategies
mode_imputer = SimpleImputer(strategy='most_frequent')
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
knn_imputer = KNNImputer(n_neighbors=5)
mean_imputer = SimpleImputer(strategy='mean')

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_auc = []
fold_f1 = []
fold_accuracy = []

# Prepare for ROC plots
plt.figure(figsize=(10, 8))

# Hyperparameter grid
hidden_layer_sizes = [[128, 64, 32], [256, 128, 64], [64, 32, 16]]
learning_rates = [0.001, 0.0005, 0.0001]
best_params = None
best_auc = 0

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

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Hyperparameter tuning
    for hidden_layers in hidden_layer_sizes:
        for lr in learning_rates:
            model = FNN(input_size=X_train_tensor.shape[1], hidden_sizes=hidden_layers)
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Train the model
            model.train()
            for epoch in range(100):
                for batch_X, batch_y in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True):
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).squeeze().numpy()
                roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2])
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    best_params = {'hidden_layers': hidden_layers, 'learning_rate': lr}

    # Final Model with Best Parameters
    model = FNN(input_size=X_train_tensor.shape[1], hidden_sizes=best_params['hidden_layers'])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    model.train()
    for epoch in range(100):
        for batch_X, batch_y in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate with Best Parameters
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).squeeze().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        fold_auc.append(roc_auc)

        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracy.append(accuracy) 


        f1 = f1_score(y_test, y_pred)
        fold_f1.append(f1)

        # Plot ROC curve for each fold
        plt.plot(fpr, tpr, label=f"Fold {fold + 1} (AUC = {roc_auc:.2f})")

# Finalize the ROC plot
mean_auc = np.mean(fold_auc)
mean_f1 = np.mean(fold_f1)
mean_accuracy = np.mean(fold_accuracy)

plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title(f"FNN ROC Curve (Mean AUC = {mean_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Boxplot of AUC scores
plt.figure(figsize=(6, 4))
plt.boxplot(fold_auc, vert=False)
plt.title("AUC Scores Across Folds - FNN")
plt.xlabel("AUC Score")
plt.show()

print(f"\nMean AUC across all folds: {mean_auc:.4f}")
print(f"Mean F1-score across all folds: {mean_f1:.4f}")
print(f"Mean Accuracy across all folds: {mean_accuracy:.4f}")
print(f"Best Hyperparameters: {best_params}")
