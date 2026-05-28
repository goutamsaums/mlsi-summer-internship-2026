# ============================================================
# DAY 3 — EXERCISE 3A : ML WORKFLOW
# Foundations of Supervised Learning
# ============================================================

# Topics Covered:
# 1. Train-Validation-Test Split
# 2. Cross Validation from Scratch
# 3. Performance Metrics
# 4. Bias-Variance Tradeoff
# 5. Learning Curves
# 6. Model Evaluation Workflow
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

from sklearn.model_selection import (
    train_test_split,
    learning_curve
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

np.random.seed(42)


# ============================================================
# 2. LOAD DATASET
# ============================================================

housing = fetch_california_housing()

X = housing.data
y = housing.target

feature_names = housing.feature_names

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("Feature Shape :", X.shape)

print("Target Shape  :", y.shape)

print("\nFeature Names:")
print(feature_names)


# ============================================================
# 3. TRAIN - VALIDATION - TEST SPLIT
# ============================================================

"""
ML Workflow:

1. Train Set
   Used to train the model

2. Validation Set
   Used for tuning and model selection

3. Test Set
   Used for final evaluation
"""

# First split:
# 80% train+validation
# 20% test

X_temp, X_test, y_temp, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# Second split:
# 60% train
# 20% validation

X_train, X_val, y_train, y_val = train_test_split(

    X_temp,
    y_temp,

    test_size=0.25,

    random_state=42

)

print("\n==============================")
print("TRAIN-VALIDATION-TEST SPLIT")
print("==============================")

print("Training Set Shape   :", X_train.shape)

print("Validation Set Shape :", X_val.shape)

print("Testing Set Shape    :", X_test.shape)


# ============================================================
# 4. TRAIN BASIC MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\n==============================")
print("MODEL TRAINING")
print("==============================")

print("Linear Regression Model Trained")


# ============================================================
# 5. PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)

y_val_pred = model.predict(X_val)

y_test_pred = model.predict(X_test)


# ============================================================
# 6. PERFORMANCE METRICS
# ============================================================

"""
Basic Performance Metrics:

1. MSE
2. RMSE
3. MAE
4. R² Score
"""

print("\n==============================")
print("PERFORMANCE METRICS")
print("==============================")


def evaluate_model(y_true, y_pred, dataset_name):

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)

    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name}")

    print("-" * 30)

    print(f"MSE   : {mse:.4f}")

    print(f"RMSE  : {rmse:.4f}")

    print(f"MAE   : {mae:.4f}")

    print(f"R²    : {r2:.4f}")

    return mse, rmse, mae, r2


train_metrics = evaluate_model(
    y_train,
    y_train_pred,
    "Training Performance"
)

val_metrics = evaluate_model(
    y_val,
    y_val_pred,
    "Validation Performance"
)

test_metrics = evaluate_model(
    y_test,
    y_test_pred,
    "Testing Performance"
)


# ============================================================
# 7. CROSS VALIDATION FROM SCRATCH
# ============================================================

"""
K-Fold Cross Validation

Process:

1. Divide data into K folds
2. Train on K-1 folds
3. Validate on remaining fold
4. Repeat K times
5. Average the results
"""

print("\n==============================")
print("K-FOLD CROSS VALIDATION")
print("==============================")

k = 5

fold_size = len(X_train) // k

indices = np.arange(len(X_train))

np.random.shuffle(indices)

cv_scores = []

for fold in range(k):

    start = fold * fold_size

    end = start + fold_size

    val_indices = indices[start:end]

    train_indices = np.concatenate(

        (indices[:start], indices[end:])

    )

    X_fold_train = X_train[train_indices]

    y_fold_train = y_train[train_indices]

    X_fold_val = X_train[val_indices]

    y_fold_val = y_train[val_indices]

    fold_model = LinearRegression()

    fold_model.fit(
        X_fold_train,
        y_fold_train
    )

    fold_predictions = fold_model.predict(
        X_fold_val
    )

    fold_rmse = np.sqrt(

        mean_squared_error(
            y_fold_val,
            fold_predictions
        )

    )

    cv_scores.append(fold_rmse)

    print(f"Fold {fold + 1} RMSE : {fold_rmse:.4f}")


print("\nAverage Cross Validation RMSE")

print("--------------------------------")

print(np.mean(cv_scores))


# ============================================================
# 8. BIAS-VARIANCE TRADEOFF
# ============================================================

"""
Bias:
- Underfitting
- Model too simple

Variance:
- Overfitting
- Model too complex
"""

print("\n==============================")
print("BIAS-VARIANCE TRADEOFF")
print("==============================")

train_r2 = train_metrics[3]

val_r2 = val_metrics[3]

print(f"\nTraining R²   : {train_r2:.4f}")

print(f"Validation R² : {val_r2:.4f}")


difference = abs(train_r2 - val_r2)

print(f"Difference     : {difference:.4f}")


if difference < 0.05:

    print("\nModel Generalizes Well")

elif train_r2 > val_r2:

    print("\nPossible Overfitting (High Variance)")

else:

    print("\nPossible Underfitting (High Bias)")


# ============================================================
# 9. LEARNING CURVES
# ============================================================

"""
Learning Curves:

Plots:
1. Training Error
2. Validation Error

Used to analyze:
- Overfitting
- Underfitting
- Generalization
"""

print("\n==============================")
print("LEARNING CURVES")
print("==============================")

train_sizes, train_scores, val_scores = learning_curve(

    estimator=LinearRegression(),

    X=X_train,

    y=y_train,

    train_sizes=np.linspace(0.1, 1.0, 10),

    cv=5,

    scoring='neg_mean_squared_error'

)

train_errors = -train_scores

val_errors = -val_scores

train_mean = np.mean(train_errors, axis=1)

val_mean = np.mean(val_errors, axis=1)


# ============================================================
# 10. PLOT LEARNING CURVES
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(

    train_sizes,
    train_mean,

    label="Training Error"

)

plt.plot(

    train_sizes,
    val_mean,

    label="Validation Error"

)

plt.xlabel("Training Set Size")

plt.ylabel("Mean Squared Error")

plt.title("Learning Curves")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# 11. OVERFITTING VS UNDERFITTING
# ============================================================

print("\n==============================")
print("MODEL INTERPRETATION")
print("==============================")

print("""

1. Underfitting
   - High Training Error
   - High Validation Error
   - Model too simple

2. Overfitting
   - Very Low Training Error
   - High Validation Error
   - Model memorizes data

3. Good Generalization
   - Similar train and validation error
   - Balanced performance

4. Learning Curves Help Detect:
   - Bias
   - Variance
   - Data sufficiency
""")


# ============================================================
# 12. MODEL COMPLEXITY DEMONSTRATION
# ============================================================

print("\n==============================")
print("MODEL COMPLEXITY")
print("==============================")

print("""

Low Complexity Model:
- High Bias
- Underfitting

High Complexity Model:
- High Variance
- Overfitting

Goal:
Find balance between:
- Bias
- Variance
""")


# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

print("\n==============================")
print("FINAL TEST EVALUATION")
print("==============================")

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

print(f"\nTest RMSE : {test_rmse:.4f}")

print(f"Test R²   : {test_r2:.4f}")


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n==============================")
print("MODEL COEFFICIENTS")
print("==============================")

coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": model.coef_

})

print(coefficients)


# ============================================================
# 15. RESIDUAL ANALYSIS
# ============================================================

residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid(True)

plt.show()


# ============================================================
# 16. COMPLETE ML WORKFLOW SUMMARY
# ============================================================

print("\n==============================")
print("COMPLETE ML WORKFLOW")
print("==============================")

print("""

Step 1 : Load Dataset

Step 2 : Split Data
         Train / Validation / Test

Step 3 : Train Model

Step 4 : Evaluate Performance

Step 5 : Cross Validation

Step 6 : Analyze Bias-Variance

Step 7 : Create Learning Curves

Step 8 : Final Test Evaluation

Step 9 : Improve Model
""")


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("DAY 3 — EXERCISE 3A COMPLETE")
print("==============================")

print("""

Topics Completed:

✓ Train-Validation-Test Split
✓ Cross Validation from Scratch
✓ Performance Metrics
✓ Bias-Variance Tradeoff
✓ Learning Curves
✓ Residual Analysis
✓ Model Evaluation Workflow

Key Concepts Learned:

✓ Underfitting
✓ Overfitting
✓ Generalization
✓ Model Validation
✓ Cross Validation
✓ Learning Curves
✓ Error Analysis
✓ Workflow Design

""")
