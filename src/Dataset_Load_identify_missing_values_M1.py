import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Load Dataset
# ============================================

print("1. Load the Dataset")

file_path = "/Users/padaltiruvinayak/Desktop/Placement_predict/dataset/placement_predict_50K_Raw.csv"

try:
    # Read CSV
    df = pd.read_csv(file_path)

    # ============================================
    # Display Dataset
    # ============================================

    print("-----------------------------------")
    print("1. Dataset Contents:")
    print("-----------------------------------")
    print(df)

    print("-----------------------------------")
    print("2. Number of Rows and Columns")
    print("-----------------------------------")
    print(df.shape)

    print("-----------------------------------")
    print("3. Column Names")
    print("-----------------------------------")
    print(df.columns.tolist())

    # Display settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("-----------------------------------")
    print("4. First 10 Records")
    print("-----------------------------------")
    print(df.head(10))

    print("-----------------------------------")
    print("Last 10 Records")
    print("-----------------------------------")
    print(df.tail(10))

    # ============================================
    # Dataset Understanding
    # ============================================

    print("-----------------------------------")
    print("5. Data Types")
    print("-----------------------------------")
    print(df.dtypes)

    print("=" * 60)

    print("6. Column Names with Data Types")
    print("-" * 40)

    for column in df.columns:
        print(f"{column:<25}{df[column].dtype}")

    print("-----------------------------------")
    print("7. Dataset Information")
    print("-----------------------------------")
    df.info()

    print("\n" + "=" * 60)

    # ============================================
    # Numeric Columns
    # ============================================

    print("-----------------------------------")
    print("8. Numerical Columns")
    print("-----------------------------------")

    numeric_df = df.select_dtypes(include=["int64", "float64"])

    print(numeric_df)

    print("-----------------------------------")
    print("9. Missing Values in Numerical Columns")
    print("-----------------------------------")
    print(numeric_df.isnull().sum())

    print("Total Missing Numerical Values:",
          numeric_df.isnull().sum().sum())

    # ============================================
    # Float Columns
    # ============================================

    float_columns = df.select_dtypes(include=["float64"]).columns

    print("-----------------------------------")
    print("10. Float Column Names")
    print("-----------------------------------")

    for col in float_columns:
        print(col)

    print("-----------------------------------")
    print("11. Missing Values in Float Columns")
    print("-----------------------------------")

    print(df[float_columns].isnull().sum())

    print("Total Missing Float Values:",
          df[float_columns].isnull().sum().sum())

    # ============================================
    # Categorical Columns
    # ============================================

    categorical_df = df.select_dtypes(include=["object"])

    print("-----------------------------------")
    print("12. Categorical Columns")
    print("-----------------------------------")
    print(categorical_df)

    print("-----------------------------------")
    print("Missing Values in Categorical Columns")
    print("-----------------------------------")
    print(categorical_df.isnull().sum())

    print("Total Missing Categorical Values:",
          categorical_df.isnull().sum().sum())

    # ============================================
    # Missing Values
    # ============================================

    print("-----------------------------------")
    print("13. Missing Values in Each Column")
    print("-----------------------------------")
    print(df.isnull().sum())

    total_missing = df.isnull().sum().sum()

    print("-----------------------------------")
    print("14. Total Missing Values")
    print("-----------------------------------")
    print(total_missing)

    # ============================================
    # Duplicate Records
    # ============================================

    duplicate_count = df.duplicated().sum()

    print("-----------------------------------")
    print("15. Duplicate Records")
    print("-----------------------------------")
    print(duplicate_count)

    # ============================================
    # Statistical Summary
    # ============================================

    print("-----------------------------------")
    print("16. Statistical Overview")
    print("-----------------------------------")
    print(df.describe())

    # ============================================
    # Missing Values Heatmap
    # ============================================

    print("-----------------------------------")
    print("17. Missing Values Heatmap")
    print("-----------------------------------")

    plt.figure(figsize=(12,6))

    sns.heatmap(
        df.isnull(),
        cmap="viridis",
        cbar=False,
        yticklabels=False,
        xticklabels=True
    )

    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File not found: {file_path}")

except Exception as e:
    print(f"An error occurred: {e}")