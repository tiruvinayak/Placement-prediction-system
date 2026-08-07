import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
import matplotlib.pyplot as plt

file_path = "/Users/padaltiruvinayak/Desktop/Placement_predict/dataset/placement_predict_50K_Raw.csv"
df = pd.read_csv(file_path)

print( " orignal dataset")
print("------------------------")
print(df.head())

print("Dataset :",df.shape)

print("\n Data type ")
print("---------------------")
print(df.dtypes)

print("\n missing values ")
print("---------------------")
print(df.isnull().sum())

print("\nDupilcant values",df.duplicated().sum())

#Step-2  Remove dupilcants

df = df.drop_dupilcantes()

#step-3  Handle Missing values

numerical_cols = df.select_dtypes(include=['int 64','float64']).columns

#filling the numerical values missing in the mean
for column in numerical_cols:
    df[column] = df[column].fillna(df[column].mode())

# Categorical Columns
categorical_columns = df.select_dtypes(include=['object']).columns


# Fill missing values in categorical columns with mode
for column in categorical_columns:
   df[column] = df[column].fillna(df[column].mode()[0])


# ---------------------------------------------------
# Step 4: Remove Leading and Trailing Spaces (Extra spaces from text columns)
# ---------------------------------------------------
for column in categorical_columns:
   df[column] = df[column].str.strip()




# ------------------------------------------------------------
# Select Numeric Columns
# ------------------------------------------------------------
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns


print("\nNumeric Columns:")
print(list(numeric_columns))


# ------------------------------------------------------------
# Standardization (Z-score)
# Mean = 0, Standard Deviation = 1
# ------------------------------------------------------------
standard_scaler = StandardScaler()


standardized = standard_scaler.fit_transform(df[numeric_columns])


for i, col in enumerate(numeric_columns):
   df[col + "_Standardized"] = standardized[:, i]


# ------------------------------------------------------------
# Feature Scaling (Min-Max Scaling)
# Values between 0 and 1
# ------------------------------------------------------------
minmax_scaler = MinMaxScaler()


scaled = minmax_scaler.fit_transform(df[numeric_columns])


for i, col in enumerate(numeric_columns):
   df[col + "_Scaled"] = scaled[:, i]


# ------------------------------------------------------------
# Normalization (L2 Normalization)
# Each row becomes a unit vector
# ------------------------------------------------------------
normalizer = Normalizer(norm='l2')


normalized = normalizer.fit_transform(df[numeric_columns])


for i, col in enumerate(numeric_columns):
   df[col + "_Normalized"] = normalized[:, i]




# ------------------------------------------------------------
# Display Results after pre-processing (Verify Dataset)
# ------------------------------------------------------------
print("\n Display Results after Preprocessed Dataset")
print(df.head())


print("\nDataset Shape:", df.shape)


print("\nDataset Information")
print(df.info())


print("\nColumns in Dataset:")
print(df.columns)


print("\nMissing Values After Preprocessing")
print(df.isnull().sum())


print("\nDuplicate Records After Preprocessing")
print(df.duplicated().sum())


# ---------------------------------------------------
# Step 8: Save Preprocessed Dataset
# ---------------------------------------------------
df.to_csv("/Users/padaltiruvinayak/Desktop/Placement_predict/dataset/clean_minmax_stand_norma_M2.csv", index=False)


print("\nPreprocessed dataset saved successfully.")


# to display histogram of preprocessed data
pf = pd.read_csv(" /Users/padaltiruvinayak/Desktop/Placement_predict/dataset/clean_minmax_stand_norma_M2.csv")
# Display histograms
pf.hist(figsize=(12, 10), bins=10, edgecolor='black')


plt.suptitle("Histogram of Preprocessed Placement Dataset")
plt.tight_layout()
plt.show()
