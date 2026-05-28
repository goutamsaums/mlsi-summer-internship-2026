# ============================================================
# DAY 3 — EXERCISE 3B : LINEAR REGRESSION
# Foundations of Supervised Learning
# ============================================================

# Topics Covered:
# 1. Linear Regression using Normal Equation
# 2. Ridge Regression (L2 Regularization)
# 3. Lasso Regression (L1 Regularization)
# 4. Comparison with Scikit-Learn
# 5. Residual Analysis
# 6. Model Assumptions
# 7. Bias-Variance Tradeoff
# 8. Learning Curves
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

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================")

print("Training Shape :", X_train.shape)

print("Testing Shape  :", X_test.shape)


# ============================================================
# 4. ADD BIAS COLUMN
# ============================================================

"""
Normal Equation:

Theta = (XᵀX)^(-1)Xᵀy

Bias term added manually.
"""

X_train_bias = np.c_[

    np.ones(X_train.shape[0]),
    X_train

]

X_test_bias = np.c_[

    np.ones(X_test.shape[0]),
    X_test

]


# ============================================================
# 5. LINEAR REGRESSION FROM SCRATCH
# ============================================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

theta = np.linalg.inv(

    X_train_bias.T.dot(X_train_bias)

).dot(

    X_train_bias.T

).dot(y_train)

print("\nTheta Parameters:")
print(theta)


# ============================================================
# 6. PREDICTIONS
# ============================================================

y_pred_manual = X_test_bias.dot(theta)


# ============================================================
# 7. PERFORMANCE METRICS
# ============================================================

print("\n==============================")
print("MANUAL MODEL PERFORMANCE")
print("==============================")

mse_manual = mean_squared_error(
    y_test,
    y_pred_manual
)

rmse_manual = np.sqrt(mse_manual)

mae_manual = mean_absolute_error(
    y_test,
    y_pred_manual
)

r2_manual = r2_score(
    y_test,
    y_pred_manual
)

print(f"MSE   : {mse_manual:.4f}")

print(f"RMSE  : {rmse_manual:.4f}")

print(f"MAE   : {mae_manual:.4f}")

print(f"R²    : {r2_manual:.4f}")


# ============================================================
# 8. SCIKIT-LEARN COMPARISON
# ============================================================

print("\n==============================")
print("SCIKIT-LEARN COMPARISON")
print("==============================")

sklearn_model = LinearRegression()

sklearn_model.fit(
    X_train,
    y_train
)

y_pred_sklearn = sklearn_model.predict(
    X_test
)

mse_sklearn = mean_squared_error(
    y_test,
    y_pred_sklearn
)

rmse_sklearn = np.sqrt(mse_sklearn)

mae_sklearn = mean_absolute_error(
    y_test,
    y_pred_sklearn
)

r2_sklearn = r2_score(
    y_test,
    y_pred_sklearn
)

print("\nScikit-Learn Linear Regression")
print("--------------------------------")

print(f"MSE   : {mse_sklearn:.4f}")

print(f"RMSE  : {rmse_sklearn:.4f}")

print(f"MAE   : {mae_sklearn:.4f}")

print(f"R²    : {r2_sklearn:.4f}")


# ============================================================
# 9. RIDGE REGRESSION (L2)
# ============================================================

"""
Ridge Regression:

Cost Function:

J(θ) = MSE + λΣθ²

Reduces Overfitting
Controls Large Coefficients
"""

print("\n==============================")
print("RIDGE REGRESSION")
print("==============================")

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train,
    y_train
)

y_pred_ridge = ridge_model.predict(
    X_test
)

ridge_mse = mean_squared_error(
    y_test,
    y_pred_ridge
)

ridge_rmse = np.sqrt(ridge_mse)

ridge_r2 = r2_score(
    y_test,
    y_pred_ridge
)

print(f"MSE  : {ridge_mse:.4f}")

print(f"RMSE : {ridge_rmse:.4f}")

print(f"R²   : {ridge_r2:.4f}")


# ============================================================
# 10. LASSO REGRESSION (L1)
# ============================================================

"""
Lasso Regression:

Cost Function:

J(θ) = MSE + λΣ|θ|

Performs Feature Selection
Some coefficients become zero.
"""

print("\n==============================")
print("LASSO REGRESSION")
print("==============================")

lasso_model = Lasso(alpha=0.01)

lasso_model.fit(
    X_train,
    y_train
)

y_pred_lasso = lasso_model.predict(
    X_test
)

lasso_mse = mean_squared_error(
    y_test,
    y_pred_lasso
)

lasso_rmse = np.sqrt(lasso_mse)

lasso_r2 = r2_score(
    y_test,
    y_pred_lasso
)

print(f"MSE  : {lasso_mse:.4f}")

print(f"RMSE : {lasso_rmse:.4f}")

print(f"R²   : {lasso_r2:.4f}")


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

comparison = pd.DataFrame({

    "Model": [
        "Manual Linear Regression",
        "Scikit-Learn Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "RMSE": [
        rmse_manual,
        rmse_sklearn,
        ridge_rmse,
        lasso_rmse
    ],

    "R² Score": [
        r2_manual,
        r2_sklearn,
        ridge_r2,
        lasso_r2
    ]

})

print(comparison)


# ============================================================
# 12. RESIDUAL ANALYSIS
# ============================================================

"""
Residual = Actual - Predicted

Good Model:
- Residuals randomly distributed
- No visible pattern
"""

print("\n==============================")
print("RESIDUAL ANALYSIS")
print("==============================")

residuals = y_test - y_pred_sklearn


# ============================================================
# 13. RESIDUAL PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred_sklearn,
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

plt.savefig("residual_plot_linear_regression.png")

plt.show()


# ============================================================
# 14. HISTOGRAM OF RESIDUALS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    residuals,
    bins=30
)

plt.xlabel("Residual")

plt.ylabel("Frequency")

plt.title("Distribution of Residuals")

plt.grid(True)

plt.savefig("residual_distribution.png")

plt.show()


# ============================================================
# 15. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred_sklearn,
    alpha=0.5
)

plt.plot(

    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],

    'r--'

)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.savefig("actual_vs_predicted.png")

plt.show()


# ============================================================
# 16. LEARNING CURVES
# ============================================================

"""
Learning Curves:

1. Training Error
2. Validation Error

Used to detect:
- Overfitting
- Underfitting
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

print("\nTraining Error Mean:")
print(train_mean)

print("\nValidation Error Mean:")
print(val_mean)


# ============================================================
# 17. PLOT LEARNING CURVES
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

plt.savefig("learning_curves_linear_regression.png")

plt.show()


# ============================================================
# 18. MODEL ASSUMPTIONS
# ============================================================

print("\n==============================")
print("LINEAR REGRESSION ASSUMPTIONS")
print("==============================")

print("""

1. Linearity
   Relationship between X and y should be linear.

2. Independence
   Observations should be independent.

3. Homoscedasticity
   Constant variance of residuals.

4. Normality
   Residuals approximately normally distributed.

5. No Multicollinearity
   Features should not be highly correlated.

6. No Extreme Outliers
   Outliers can distort predictions.
""")


# ============================================================
# 19. BIAS-VARIANCE TRADEOFF
# ============================================================

print("\n==============================")
print("BIAS-VARIANCE TRADEOFF")
print("==============================")

print("""

High Bias:
- Underfitting
- Model too simple

High Variance:
- Overfitting
- Model too complex

Regularization helps:
- Reduce variance
- Improve generalization
""")


# ============================================================
# 20. VISUAL BIAS-VARIANCE DEMONSTRATION
# ============================================================

np.random.seed(0)

X_demo = np.sort(np.random.rand(30, 1) * 5, axis=0)

y_demo = np.sin(X_demo).ravel() + np.random.randn(30) * 0.2

degrees = [1, 3, 10]

plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):

    model_poly = make_pipeline(

        PolynomialFeatures(degree),
        LinearRegression()

    )

    model_poly.fit(X_demo, y_demo)

    X_plot = np.linspace(0, 5, 100).reshape(-1, 1)

    y_plot = model_poly.predict(X_plot)

    plt.subplot(1, 3, i + 1)

    plt.scatter(X_demo, y_demo)

    plt.plot(X_plot, y_plot)

    plt.title(f"Polynomial Degree {degree}")

plt.tight_layout()

plt.savefig("bias_variance_tradeoff.png")

plt.show()


# ============================================================
# 21. FEATURE COEFFICIENTS
# ============================================================

print("\n==============================")
print("FEATURE COEFFICIENTS")
print("==============================")

coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": sklearn_model.coef_

})

print(coefficients)


# ============================================================
# 22. REGULARIZATION EFFECT
# ============================================================

print("\n==============================")
print("REGULARIZATION EFFECT")
print("==============================")

ridge_coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Ridge Coefficient": ridge_model.coef_

})

lasso_coefficients = pd.DataFrame({

    "Feature": feature_names,

    "Lasso Coefficient": lasso_model.coef_

})

print("\nRidge Coefficients")
print(ridge_coefficients)

print("\nLasso Coefficients")
print(lasso_coefficients)


# ============================================================
# 23. COEFFICIENT COMPARISON PLOT
# ============================================================

x = np.arange(len(feature_names))

width = 0.25

plt.figure(figsize=(12, 6))

plt.bar(
    x - width,
    sklearn_model.coef_,
    width,
    label='Linear Regression'
)

plt.bar(
    x,
    ridge_model.coef_,
    width,
    label='Ridge'
)

plt.bar(
    x + width,
    lasso_model.coef_,
    width,
    label='Lasso'
)

plt.xticks(x, feature_names, rotation=45)

plt.ylabel("Coefficient Value")

plt.title("Coefficient Comparison")

plt.legend()

plt.grid(True)

plt.savefig("coefficient_comparison.png")

plt.show()


# ============================================================
# 24. INTERPRETATION
# ============================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

print("""

1. Linear Regression
   Finds best-fit relationship.

2. Normal Equation
   Closed-form analytical solution.

3. Ridge Regression
   Prevents overfitting using L2 penalty.

4. Lasso Regression
   Performs feature selection using L1 penalty.

5. Residual Analysis
   Detects model problems and assumption violations.

6. Learning Curves
   Detect bias and variance problems.

7. Regularization
   Improves generalization performance.
""")


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("DAY 3 — EXERCISE 3B COMPLETE")
print("==============================")

print("""

Topics Completed:

✓ Linear Regression from Scratch
✓ Normal Equation
✓ Ridge Regression
✓ Lasso Regression
✓ Scikit-Learn Comparison
✓ Residual Analysis
✓ Learning Curves
✓ Model Assumptions
✓ Bias-Variance Tradeoff
✓ Regularization

Key Concepts Learned:

✓ Underfitting
✓ Overfitting
✓ Generalization
✓ Residual Diagnostics
✓ Regularization
✓ Model Complexity
✓ Learning Curves

""")
