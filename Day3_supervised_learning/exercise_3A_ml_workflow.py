# ============================================================
# DAY 3 — EXERCISE 3A : ML WORKFLOW
# Foundations of Supervised Learning
# ============================================================

# Topics Covered:
# 1. Train / Validation / Test Split
# 2. K-Fold Cross Validation (From Scratch)
# 3. Performance Metrics
# 4. Bias-Variance Tradeoff
# 5. Learning Curves
#
# Dataset:
# California Housing Dataset
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score

np.random.seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

housing = fetch_california_housing()

X = housing.data
y = housing.target

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Feature Shape :", X.shape)
print("Target Shape  :", y.shape)

print("\nFeature Names:")
print(housing.feature_names)


# ============================================================
# 3. TRAIN - VALIDATION - TEST SPLIT
# ============================================================

"""
Workflow:
- Training Set   -> Train model
- Validation Set -> Tune model
- Test Set       -> Final evaluation
"""

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("\n==============================")
print("DATA SPLIT")
print("==============================")

print("Training Set   :", X_train.shape)
print("Validation Set :", X_val.shape)
print("Test Set       :", X_test.shape)


# ============================================================
# 4. TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)


# ============================================================
# 5. PERFORMANCE METRICS (MANUAL)
# ============================================================

# ------------------------------------------------------------
# Mean Squared Error
# ------------------------------------------------------------

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ------------------------------------------------------------
# Root Mean Squared Error
# ------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


# ------------------------------------------------------------
# Mean Absolute Error
# ------------------------------------------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


# ------------------------------------------------------------
# R² Score
# ------------------------------------------------------------

def r2_manual(y_true, y_pred):

    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)

    ss_residual = np.sum((y_true - y_pred) ** 2)

    return 1 - (ss_residual / ss_total)


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nTRAINING PERFORMANCE")
print("--------------------")
print("MSE  :", mse(y_train, train_predictions))
print("RMSE :", rmse(y_train, train_predictions))
print("MAE  :", mae(y_train, train_predictions))
print("R²   :", r2_manual(y_train, train_predictions))

print("\nVALIDATION PERFORMANCE")
print("--------------------")
print("MSE  :", mse(y_val, val_predictions))
print("RMSE :", rmse(y_val, val_predictions))
print("MAE  :", mae(y_val, val_predictions))
print("R²   :", r2_manual(y_val, val_predictions))

print("\nTEST PERFORMANCE")
print("--------------------")
print("MSE  :", mse(y_test, test_predictions))
print("RMSE :", rmse(y_test, test_predictions))
print("MAE  :", mae(y_test, test_predictions))
print("R²   :", r2_manual(y_test, test_predictions))


# ============================================================
# 7. SCIKIT-LEARN VERIFICATION
# ============================================================

print("\n==============================")
print("SKLEARN VERIFICATION")
print("==============================")

print("R² Score (sklearn):",
      r2_score(y_test, test_predictions))


# ============================================================
# 8. K-FOLD CROSS VALIDATION FROM SCRATCH
# ============================================================

"""
Cross Validation:
- Split dataset into K folds
- Train on K-1 folds
- Validate on remaining fold
- Repeat K times
- Average scores
"""

print("\n==============================")
print("K-FOLD CROSS VALIDATION")
print("==============================")

indices = np.random.permutation(len(X))

X_shuffled = X[indices]
y_shuffled = y[indices]

k = 5

fold_size = len(X) // k

cv_scores = []

for i in range(k):

    start = i * fold_size
    end = start + fold_size

    # Validation Fold
    X_val_fold = X_shuffled[start:end]
    y_val_fold = y_shuffled[start:end]

    # Training Fold
    X_train_fold = np.concatenate(
        (X_shuffled[:start], X_shuffled[end:])
    )

    y_train_fold = np.concatenate(
        (y_shuffled[:start], y_shuffled[end:])
    )

    # Train Model
    model_cv = LinearRegression()

    model_cv.fit(X_train_fold, y_train_fold)

    predictions_fold = model_cv.predict(X_val_fold)

    fold_mse = mse(y_val_fold, predictions_fold)

    cv_scores.append(fold_mse)

    print(f"Fold {i+1} MSE :", fold_mse)

print("\nAverage CV MSE :", np.mean(cv_scores))


# ============================================================
# 9. BIAS - VARIANCE TRADEOFF
# ============================================================

"""
Bias:
- Model too simple
- Underfitting

Variance:
- Model too complex
- Overfitting
"""

print("\n==============================")
print("BIAS - VARIANCE TRADEOFF")
print("==============================")

# Synthetic Nonlinear Dataset
x = np.linspace(0, 10, 100)

y_curve = np.sin(x) + np.random.normal(0, 0.2, 100)

x = x.reshape(-1, 1)

degrees = [1, 3, 15]

plt.figure(figsize=(12, 6))

plt.scatter(x, y_curve, color='black', label='Data')

for degree in degrees:

    model_poly = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model_poly.fit(x, y_curve)

    y_pred_curve = model_poly.predict(x)

    plt.plot(
        x,
        y_pred_curve,
        label=f'Degree {degree}'
    )

plt.title("Bias-Variance Tradeoff")
plt.xlabel("X")
plt.ylabel("Y")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 10. LEARNING CURVES
# ============================================================

"""
Learning Curves Help Diagnose:
- Underfitting
- Overfitting
- Need for More Data
"""

print("\n==============================")
print("LEARNING CURVES")
print("==============================")

train_sizes, train_scores, val_scores = learning_curve(

    estimator=LinearRegression(),

    X=X,

    y=y,

    cv=5,

    scoring='neg_mean_squared_error',

    train_sizes=np.linspace(0.1, 1.0, 10)

)

# Convert Negative MSE to Positive
train_scores_mean = -np.mean(train_scores, axis=1)

val_scores_mean = -np.mean(val_scores, axis=1)


# ============================================================
# 11. PLOT LEARNING CURVES
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    train_sizes,
    train_scores_mean,
    marker='o',
    label='Training Error'
)

plt.plot(
    train_sizes,
    val_scores_mean,
    marker='o',
    label='Validation Error'
)

plt.title("Learning Curves")

plt.xlabel("Training Set Size")

plt.ylabel("Mean Squared Error")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 12. INTERPRETATION
# ============================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

print(\"\"\"
1. High Bias (Underfitting)
   - Training error high
   - Validation error high
   - Model too simple

2. High Variance (Overfitting)
   - Training error low
   - Validation error high
   - Model too complex

3. Good Generalization
   - Both errors low
   - Small gap between curves

4. Cross Validation
   - Better estimate of generalization
   - Reduces dependency on one split

5. Learning Curves
   - Diagnose model behavior
   - Show if more data helps
\"\"\")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("DAY 3 — EXERCISE 3A COMPLETE")
print("==============================")

print(\"\"\"
Topics Completed:

✓ Train / Validation / Test Split
✓ Cross Validation from Scratch
✓ Regression Metrics
✓ Bias-Variance Tradeoff
✓ Learning Curves
✓ Model Evaluation Workflow

Key Concepts Learned:

✓ Underfitting
✓ Overfitting
✓ Generalization
✓ Validation Strategy
✓ Model Complexity
✓ Workflow Design
\"\"\")

