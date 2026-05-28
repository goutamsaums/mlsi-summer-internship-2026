# ============================================================
# DAY 3 — EXERCISE 3C : CLASSIFICATION FUNDAMENTALS
# Foundations of Supervised Learning
# ============================================================

# Topics Covered:
# 1. Logistic Regression from Scratch
# 2. Sigmoid Function
# 3. Gradient Descent Optimization
# 4. Classification Metrics
# 5. Confusion Matrix
# 6. Precision, Recall, F1-Score
# 7. ROC Curve and AUC
# 8. Comparison with Scikit-Learn
# 9. Bias-Variance Concepts
#
# Dataset:
# Breast Cancer Dataset
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

np.random.seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = load_breast_cancer()

X = data.data
y = data.target

feature_names = data.feature_names

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)

print("\nClasses:")
print(data.target_names)


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)


# ============================================================
# 4. FEATURE SCALING
# ============================================================

"""
Logistic Regression performs better
when features are scaled.
"""

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ============================================================
# 5. ADD BIAS TERM
# ============================================================

X_train_bias = np.c_[

    np.ones(X_train.shape[0]),
    X_train

]

X_test_bias = np.c_[

    np.ones(X_test.shape[0]),
    X_test

]


# ============================================================
# 6. SIGMOID FUNCTION
# ============================================================

"""
Sigmoid Function:

σ(z) = 1 / (1 + e^(-z))

Converts values into probabilities.
"""

def sigmoid(z):

    return 1 / (1 + np.exp(-z))


# ============================================================
# 7. INITIALIZE PARAMETERS
# ============================================================

theta = np.zeros(X_train_bias.shape[1])

learning_rate = 0.01

iterations = 5000

m = len(y_train)


# ============================================================
# 8. COST FUNCTION
# ============================================================

"""
Binary Cross Entropy Loss
"""

def compute_cost(X, y, theta):

    predictions = sigmoid(X.dot(theta))

    epsilon = 1e-10

    cost = -(1 / m) * np.sum(

        y * np.log(predictions + epsilon)

        +

        (1 - y) * np.log(1 - predictions + epsilon)

    )

    return cost


# ============================================================
# 9. GRADIENT DESCENT
# ============================================================

print("\n==============================")
print("TRAINING LOGISTIC REGRESSION")
print("==============================")

cost_history = []

for i in range(iterations):

    predictions = sigmoid(X_train_bias.dot(theta))

    gradient = (1 / m) * X_train_bias.T.dot(

        predictions - y_train

    )

    theta = theta - learning_rate * gradient

    cost = compute_cost(
        X_train_bias,
        y_train,
        theta
    )

    cost_history.append(cost)

    if i % 500 == 0:

        print(f"Iteration {i} | Cost = {cost:.4f}")


# ============================================================
# 10. FINAL PARAMETERS
# ============================================================

print("\n==============================")
print("MODEL PARAMETERS")
print("==============================")

print(theta)


# ============================================================
# 11. PREDICTIONS
# ============================================================

probabilities = sigmoid(

    X_test_bias.dot(theta)

)

y_pred_manual = (probabilities >= 0.5).astype(int)


# ============================================================
# 12. PERFORMANCE METRICS
# ============================================================

print("\n==============================")
print("PERFORMANCE METRICS")
print("==============================")

accuracy = accuracy_score(
    y_test,
    y_pred_manual
)

precision = precision_score(
    y_test,
    y_pred_manual
)

recall = recall_score(
    y_test,
    y_pred_manual
)

f1 = f1_score(
    y_test,
    y_pred_manual
)

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(
    y_test,
    y_pred_manual
)

print(cm)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(

    classification_report(
        y_test,
        y_pred_manual
    )

)


# ============================================================
# 15. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities
)

auc_score = roc_auc_score(
    y_test,
    probabilities
)

print("\n==============================")
print("ROC-AUC")
print("==============================")

print("AUC Score :", auc_score)


# ============================================================
# 16. PLOT ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 17. COST FUNCTION PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(cost_history)

plt.xlabel("Iterations")

plt.ylabel("Cost")

plt.title("Gradient Descent Cost Reduction")

plt.grid(True)

plt.show()


# ============================================================
# 18. SCIKIT-LEARN COMPARISON
# ============================================================

print("\n==============================")
print("SCIKIT-LEARN COMPARISON")
print("==============================")

sklearn_model = LogisticRegression(
    max_iter=5000
)

sklearn_model.fit(
    X_train,
    y_train
)

y_pred_sklearn = sklearn_model.predict(
    X_test
)

accuracy_sklearn = accuracy_score(
    y_test,
    y_pred_sklearn
)

precision_sklearn = precision_score(
    y_test,
    y_pred_sklearn
)

recall_sklearn = recall_score(
    y_test,
    y_pred_sklearn
)

f1_sklearn = f1_score(
    y_test,
    y_pred_sklearn
)

print("\nScikit-Learn Logistic Regression")
print("--------------------------------")

print(f"Accuracy  : {accuracy_sklearn:.4f}")

print(f"Precision : {precision_sklearn:.4f}")

print(f"Recall    : {recall_sklearn:.4f}")

print(f"F1 Score  : {f1_sklearn:.4f}")


# ============================================================
# 19. BIAS-VARIANCE CONCEPT
# ============================================================

print("\n==============================")
print("BIAS-VARIANCE TRADEOFF")
print("==============================")

print("""
1. High Bias
   - Model too simple
   - Underfitting
   - Poor training performance

2. High Variance
   - Model too complex
   - Overfitting
   - Poor generalization

3. Logistic Regression
   - Usually low variance
   - Can underfit nonlinear data

4. Regularization
   - Reduces variance
   - Improves generalization
""")


# ============================================================
# 20. PRECISION, RECALL, F1 EXPLANATION
# ============================================================

print("\n==============================")
print("CLASSIFICATION METRICS")
print("==============================")

print("""
1. Accuracy
   Correct Predictions / Total Predictions

2. Precision
   TP / (TP + FP)

   Measures correctness of positive predictions.

3. Recall
   TP / (TP + FN)

   Measures ability to find positives.

4. F1 Score
   Harmonic mean of Precision and Recall.

5. ROC Curve
   Shows tradeoff between TPR and FPR.

6. AUC
   Higher AUC means better classifier.
""")


# ============================================================
# 21. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0, 1], ["Malignant", "Benign"])

plt.yticks([0, 1], ["Malignant", "Benign"])

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

print("\n==============================")
print("FEATURE COEFFICIENTS")
print("==============================")

coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": theta[1:]

})

print(coefficients.sort_values(
    by="Coefficient",
    ascending=False
))


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("DAY 3 — EXERCISE 3C COMPLETE")
print("==============================")

print("""
Topics Completed:

✓ Logistic Regression from Scratch
✓ Sigmoid Function
✓ Gradient Descent
✓ Binary Classification
✓ Confusion Matrix
✓ Precision
✓ Recall
✓ F1 Score
✓ ROC Curve
✓ AUC Score
✓ Scikit-Learn Comparison
✓ Bias-Variance Tradeoff

Key Concepts Learned:

✓ Probabilistic Classification
✓ Optimization
✓ Decision Boundary
✓ Performance Evaluation
✓ Generalization
✓ Classification Metrics
""")

