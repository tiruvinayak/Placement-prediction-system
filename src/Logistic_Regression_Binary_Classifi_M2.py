# ============================================================
# LOGISTIC REGRESSION - BINARY CLASSIFICATION
# Placement Prediction
#
# Features:
#   1. CGPA
#   2. HistoryOfBacklogs
#   3. Internships
#
# Target:
#   PlacementStatus
#
# -1 -> 0 -> Not Placed
#  0 -> 1 -> Placed
#
# Manual implementation using:
#   - Sigmoid
#   - Cross-Entropy Loss
#   - Gradient Descent
#   - Confusion Matrix
#   - Decision Boundary
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET = "/Users/padaltiruvinayak/Desktop/Placement_predict/dataset/final_preprocess_M2.csv"

OUTPUT_FOLDER = "/Users/padaltiruvinayak/Desktop/Placement_predict/outputs/Logistic_Regression_Binary_Classify_M2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD PLACEMENT DATASET
# ============================================================

df = pd.read_csv(DATASET)

print("\n========== DATASET ==========")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

FEATURES = [
    "CGPA",
    "HistoryOfBacklogs",
    "Internships"
]

TARGET = "PlacementStatus"


X = df[FEATURES].values

y = df[TARGET].values.astype(int)


# ============================================================
# 4. CONVERT TARGET TO BINARY 0/1
# ============================================================

# Your dataset contains:
#
# -1 = Not Placed
#  0 = Placed
#
# Logistic Regression binary classification is easier to
# handle with:
#
# 0 = Not Placed
# 1 = Placed
#
# Therefore:
# -1 -> 0
#  0 -> 1

y = np.where(y == -1, 0, 1)


print("\nFeatures:")
print(FEATURES)

print("\nTarget:")
print(TARGET)

print("\nTarget distribution:")
print(pd.Series(y).value_counts().sort_index())


# ============================================================
# 5. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\n========== STANDARDIZATION ==========")

print("Training data standardized.")
print("Testing data standardized.")


# ============================================================
# 7. SIGMOID FUNCTION
# ============================================================

def sigmoid(z):
    """
    Sigmoid function:

                 1
    σ(z) = -------------
             1 + e^(-z)

    Converts the linear model output into probability.
    """

    # Prevent overflow in exp()
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# ============================================================
# 8. SIGMOID GRAPH
# ============================================================

z_values = np.linspace(-10, 10, 500)

sigmoid_values = sigmoid(z_values)


plt.figure(figsize=(8, 6))

plt.plot(
    z_values,
    sigmoid_values,
    linewidth=3
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Threshold = 0.5"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("z")

plt.ylabel("Sigmoid(z)")

plt.title("Sigmoid Function")

plt.legend()

plt.grid(alpha=0.3)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_sigmoid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

# IMPORTANT:
# Do not use plt.show()
# Otherwise the program waits for the window.
plt.close()


# ============================================================
# 9. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(y_true, y_probability):
    """
    Binary Cross-Entropy Loss:

    L = -1/m Σ [
            y log(p) +
            (1-y) log(1-p)
        ]
    """

    epsilon = 1e-15

    y_probability = np.clip(
        y_probability,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_probability)
        +
        (1 - y_true) * np.log(1 - y_probability)
    )

    return loss


# ============================================================
# 10. INITIALIZE LOGISTIC REGRESSION
# ============================================================

number_of_features = X_train_scaled.shape[1]

weights = np.zeros(number_of_features)

bias = 0.0

learning_rate = 0.05

epochs = 3000

loss_history = []


print("\n========== TRAINING ==========")

print("Number of features:", number_of_features)

print("Learning rate:", learning_rate)

print("Epochs:", epochs)


# ============================================================
# 11. TRAIN USING GRADIENT DESCENT
# ============================================================

m = len(y_train)


for epoch in range(epochs):

    # --------------------------------------------------------
    # Linear model
    #
    # z = w1*x1 + w2*x2 + w3*x3 + b
    # --------------------------------------------------------

    z = np.dot(
        X_train_scaled,
        weights
    ) + bias


    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    probability = sigmoid(z)


    # --------------------------------------------------------
    # Cross-entropy loss
    # --------------------------------------------------------

    loss = cross_entropy_loss(
        y_train,
        probability
    )

    loss_history.append(loss)


    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    error = probability - y_train


    dw = (
        1 / m
    ) * np.dot(
        X_train_scaled.T,
        error
    )


    db = (
        1 / m
    ) * np.sum(error)


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    weights -= learning_rate * dw

    bias -= learning_rate * db


    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if (epoch + 1) % 500 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {loss:.6f}"
        )


# ============================================================
# 12. DISPLAY MODEL PARAMETERS
# ============================================================

print("\n========== MODEL PARAMETERS ==========")


for feature, weight in zip(FEATURES, weights):

    print(
        f"{feature}: {weight:.6f}"
    )


print(
    f"Bias: {bias:.6f}"
)


# ============================================================
# 13. CROSS-ENTROPY LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Cross-Entropy Loss")

plt.title(
    "Logistic Regression - Cross-Entropy Loss"
)

plt.grid(alpha=0.3)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_cross_entropy_loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

# Prevent plt.show() from blocking the program
plt.close()


# ============================================================
# 14. PREDICTION FUNCTIONS
# ============================================================

def predict_probability(X):
    """
    Calculate placement probability.
    """

    z = np.dot(
        X,
        weights
    ) + bias

    return sigmoid(z)


def predict(X, threshold=0.5):
    """
    Convert probability into binary class.

    probability >= 0.5 -> Placed
    probability < 0.5  -> Not Placed
    """

    probability = predict_probability(X)

    return (
        probability >= threshold
    ).astype(int)


# ============================================================
# 15. TEST SET PREDICTION
# ============================================================

test_probability = predict_probability(
    X_test_scaled
)


y_pred = predict(
    X_test_scaled
)


# ============================================================
# 16. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== MODEL PERFORMANCE ==========")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Placed",
            "Placed"
        ]
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n========== CONFUSION MATRIX ==========")

print(cm)


plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.colorbar()


plt.xticks(
    [0, 1],
    [
        "Not Placed",
        "Placed"
    ]
)


plt.yticks(
    [0, 1],
    [
        "Not Placed",
        "Placed"
    ]
)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

# Prevent blocking
plt.close()


# ============================================================
# 19. DECISION BOUNDARY AS HYPERPLANE
# ============================================================

# Logistic regression equation:
#
# z = w1*x1 + w2*x2 + w3*x3 + b
#
# Decision boundary:
#
# probability = 0.5
#
# sigmoid(z) = 0.5
#
# Therefore:
#
# z = 0
#
# Hence:
#
# w1*x1 +
# w2*x2 +
# w3*x3 +
# b = 0
#
# This is a HYPERPLANE.
#
# Since we have 3 features, the complete boundary
# exists in 3-dimensional feature space.
#
# We visualize a 2D cross-section by fixing
# Internships = 0.


# ============================================================
# 20. CREATE CGPA GRID
# ============================================================

cgpa_values = np.linspace(
    df["CGPA"].min() - 0.2,
    df["CGPA"].max() + 0.2,
    300
)


historyofbacklogs_values = np.linspace(
    df["HistoryOfBacklogs"].min() - 1,
    df["HistoryOfBacklogs"].max() + 1,
    300
)


CGPA, HistoryOfBacklogs = np.meshgrid(
    cgpa_values,
    historyofbacklogs_values
)


# ============================================================
# 21. FIX INTERNSHIPS = 0
# ============================================================

INTERNSHIP = np.zeros_like(CGPA)


# ============================================================
# 22. CREATE GRID
# ============================================================

grid = np.column_stack(
    [
        CGPA.ravel(),
        HistoryOfBacklogs.ravel(),
        INTERNSHIP.ravel()
    ]
)


# ============================================================
# 23. STANDARDIZE GRID
# ============================================================

grid_scaled = scaler.transform(
    grid
)


# ============================================================
# 24. CALCULATE PROBABILITY
# ============================================================

grid_probability = predict_probability(
    grid_scaled
)


grid_probability = grid_probability.reshape(
    CGPA.shape
)


# ============================================================
# 25. PLOT DECISION BOUNDARY
# ============================================================

plt.figure(figsize=(10, 7))


# ------------------------------------------------------------
# Probability regions
# ------------------------------------------------------------

contour = plt.contourf(
    CGPA,
    HistoryOfBacklogs,
    grid_probability,
    levels=50,
    cmap="RdYlGn",
    alpha=0.35
)


plt.colorbar(
    contour,
    label="Placement Probability"
)


# ------------------------------------------------------------
# Hyperplane cross-section
# ------------------------------------------------------------

plt.contour(
    CGPA,
    HistoryOfBacklogs,
    grid_probability,
    levels=[0.5],
    colors="black",
    linewidths=3
)


# ------------------------------------------------------------
# Original data
# ------------------------------------------------------------

plt.scatter(
    df[df[TARGET] == 0]["CGPA"],
    df[df[TARGET] == 0]["HistoryOfBacklogs"],
    color="red",
    edgecolor="black",
    s=20,
    alpha=0.4,
    label="Not Placed"
)


plt.scatter(
    df[df[TARGET] == 1]["CGPA"],
    df[df[TARGET] == 1]["HistoryOfBacklogs"],
    color="green",
    edgecolor="black",
    s=20,
    alpha=0.4,
    label="Placed"
)


plt.xlabel("CGPA")

plt.ylabel("HistoryOfBacklogs")


plt.title(
    "Logistic Regression Decision Boundary\n"
    "w₁(CGPA) + w₂(HistoryOfBacklogs) + "
    "w₃(Internships) + b = 0"
)


plt.legend()

plt.grid(alpha=0.2)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_decision_boundary_hyperplane.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

# Prevent blocking
plt.close()


# ============================================================
# 26. PREDICT A NEW STUDENT
# ============================================================

# Example student:
#
# CGPA              = 8.0
# HistoryOfBacklogs = 0
# Internships       = 1


new_student = np.array(
    [
        [8.0, 0, 1]
    ]
)


# ============================================================
# 27. STANDARDIZE NEW STUDENT
# ============================================================

new_student_scaled = scaler.transform(
    new_student
)


# ============================================================
# 28. CALCULATE NEW STUDENT PROBABILITY
# ============================================================

new_probability = predict_probability(
    new_student_scaled
)[0]


# ============================================================
# 29. NEW STUDENT PREDICTION
# ============================================================

new_prediction = int(
    new_probability >= 0.5
)


# ============================================================
# 30. DISPLAY NEW STUDENT RESULT
# ============================================================

print("\n========== NEW STUDENT PREDICTION ==========")

print(
    "CGPA:",
    new_student[0][0]
)

print(
    "HistoryOfBacklogs:",
    new_student[0][1]
)

print(
    "Internships:",
    new_student[0][2]
)


print(
    "Placement Probability:",
    f"{new_probability * 100:.2f}%"
)


if new_prediction == 1:

    print(
        "Prediction: PLACED"
    )

else:

    print(
        "Prediction: NOT PLACED"
    )


# ============================================================
# 31. SAVE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "Actual": y_test,
        "Predicted": y_pred,
        "Placement_Probability": test_probability
    }
)


results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "prediction_results.csv"
    ),
    index=False
)


# ============================================================
# 32. SAVE MODEL PARAMETERS
# ============================================================

parameters = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Weight": weights
    }
)


parameters.loc[len(parameters)] = [
    "Bias",
    bias
]


parameters.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_parameters.csv"
    ),
    index=False
)


# ============================================================
# 33. FINISH
# ============================================================

print("\n============================================")

print("PROGRAM COMPLETED SUCCESSFULLY")

print("============================================")


print(
    "\nAll output files are stored in:"
)


print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print("\nGenerated images:")

print("1. 01_sigmoid.png")

print("2. 02_cross_entropy_loss.png")

print("3. 03_confusion_matrix.png")

print("4. 04_decision_boundary_hyperplane.png")


print("\nGenerated CSV files:")

print("5. prediction_results.csv")

print("6. model_parameters.csv")


print("\n============================================")