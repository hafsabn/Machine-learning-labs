from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch dataset
hepatitis = fetch_ucirepo(id=46)

# Data (as pandas dataframes)
X = hepatitis.data.features
y = hepatitis.data.targets

# Combine X and y for easier manipulation
hepatitis_data = pd.concat([X, y], axis=1)

# 1. Print the first 10 samples
print("First 10 samples:\n", hepatitis_data.head(10))

# 2. Print 3 random samples
print("\n3 random samples:\n", hepatitis_data.sample(3))

# 3. Print the last 5 samples
print("\nLast 5 samples:\n", hepatitis_data.tail(5))

# 4. Display column names, number of missing values, and data types in one command
dataset_info = hepatitis_data.isnull().sum().to_frame(name='Missing Values').join(hepatitis_data.dtypes.to_frame(name='Data Type'))
print("\nDataset Info (Missing values and Data types):\n", dataset_info)

# 5. Display only columns and their types (numerical or categorical)
columns_types = hepatitis_data.dtypes.apply(lambda x: 'Numerical' if pd.api.types.is_numeric_dtype(x) else 'Categorical')
print("\nColumns and their types:\n", columns_types)

# 6. Display number of samples and features
num_samples, num_features = hepatitis_data.shape
print(f"\nNumber of samples: {num_samples}, Number of features: {num_features}")

# 7. Display statistics for numerical features
numerical_stats = hepatitis_data.select_dtypes(include=[np.number]).describe()
print("\nStatistics for numerical features:\n", numerical_stats)

# 8. Check if dataset is balanced using a count and a plot
class_count = hepatitis_data['Class'].value_counts()
print("\nClass distribution:\n", class_count)

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=hepatitis_data)
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
plt.close()

# 9. Line plot, histogram, and box plot for continuous values only
continuous_columns = ['Age', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime']

# Line plot
plt.figure(figsize=(10, 6))
for col in continuous_columns:
    plt.plot(hepatitis_data[col].dropna(), label=col)
plt.legend()
plt.title('Line Plot for Continuous Values')
plt.savefig('line_plot.png')
plt.close()

# Histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(continuous_columns):
    row = i // 3
    col_idx = i % 3
    hepatitis_data[col].hist(bins=15, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(col)
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# Box plots
plt.figure(figsize=(10, 6))
hepatitis_data[continuous_columns].boxplot()
plt.title('Box Plot for Continuous Values')
plt.savefig('box_plot.png')
plt.close()

# Explanation for 'Protime' discontinuity
protime_non_missing = hepatitis_data['Protime'].notnull().sum()
protime_null_count = hepatitis_data['Protime'].isnull().sum()
print(f"\n'Protime' has {protime_null_count} missing values out of {len(hepatitis_data)} total samples.")
print("This causes the discontinuity in the line plot.")