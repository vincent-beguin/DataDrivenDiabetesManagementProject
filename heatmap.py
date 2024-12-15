# Data manipulation and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load your dataset
file_path = "diabetes.csv"  # Update the path to your dataset
data = pd.read_csv(file_path)

# Replace zeros with NaN for imputation (optional, based on your dataset's preprocessing needs)
columns_to_replace_zeros = ['Age', 'BMI', 'Pregnancies', 'SkinThickness', 'Insulin', 'Glucose', 'BloodPressure']
data[columns_to_replace_zeros] = data[columns_to_replace_zeros].replace(0, np.nan)  # Use np.nan instead of pd.NA

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Adjust size as needed
sns.heatmap(
    correlation_matrix,
    annot=True,          # Show correlation values
    fmt=".2f",           # Limit to 2 decimal places
    cmap="coolwarm",     # Color scheme
    cbar=True            # Show color bar
)
plt.title("Correlation Heatmap of Features")
plt.show()
