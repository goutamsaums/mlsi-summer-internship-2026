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

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.metrics import mean_squared_error, r2_score


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
print("DATA SPLIT")
print("==============================")

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)


# ============================================================
# 4. ADD BIAS COLUMN
# ============================================================

"""
Normal Equation:
Theta = (XᵀX)^(-1) Xᵀy

Need bias column of ones
"""

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]

X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]


# ============================================================
# 5. LINEAR REGRESSION FROM SCRATCH
# ============================================================

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")

# Normal Equation
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

mse_manual = mean_squared_error(y_test, y_pred_manual)

rmse_manual = np.sqrt(mse_manual)

r2_manual = r2_score(y_test, y_pred_manual)

print("\nManual Linear Regression Performance")
print("------------------------------------")

print("MSE  :", mse_manual)

print("RMSE :", rmse_manual)

print("R²   :", r2_manual)


# ============================================================
# 8. SCIKIT-LEARN COMPARISON
# ============================================================

print("\n==============================")
print("SCIKIT-LEARN COMPARISON")
print("==============================")

sklearn_model = LinearRegression()

sklearn_model.fit(X_train, y_train)

y_pred_sklearn = sklearn_model.predict(X_test)

mse_sklearn = mean_squared_error(
    y_test,
    y_pred_sklearn
)

r2_sklearn = r2_score(
    y_test,
    y_pred_sklearn
)

print("\nScikit-Learn Linear Regression")
print("--------------------------------")

print("MSE :", mse_sklearn)

print("R²  :", r2_sklearn)


# ============================================================
# 9. RIDGE REGRESSION (L2)
# ============================================================

"""
Ridge Regression:
Adds penalty term

Cost Function:

J(θ) = MSE + λΣθ²

Reduces overfitting
Controls large coefficients
"""

print("\n==============================")
print("RIDGE REGRESSION")
print("==============================")

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

ridge_mse = mean_squared_error(
    y_test,
    y_pred_ridge
)

ridge_r2 = r2_score(
    y_test,
    y_pred_ridge
)

print("\nRidge Regression Performance")
print("--------------------------------")

print("MSE :", ridge_mse)

print("R²  :", ridge_r2)


# ============================================================
# 10. LASSO REGRESSION (L1)
# ============================================================

"""
Lasso Regression:
Adds absolute coefficient penalty

Cost Function:

J(θ) = MSE + λΣ|θ|

Can shrink coefficients to zero
Useful for feature selection
"""

print("\n==============================")
print("LASSO REGRESSION")
print("==============================")

lasso_model = Lasso(alpha=0.01)

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

lasso_mse = mean_squared_error(
    y_test,
    y_pred_lasso
)

lasso_r2 = r2_score(
    y_test,
    y_pred_lasso
)

print("\nLasso Regression Performance")
print("--------------------------------")

print("MSE :", lasso_mse)

print("R²  :", lasso_r2)


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

    "MSE": [
        mse_manual,
        mse_sklearn,
        ridge_mse,
        lasso_mse
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

Good Linear Regression:
- Residuals randomly distributed
- No visible pattern
- Constant variance
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

plt.show()


# ============================================================
# 16. MODEL ASSUMPTIONS
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
   Residuals should be approximately normal.

5. No Multicollinearity
   Features should not be highly correlated.

6. No Extreme Outliers
   Outliers can distort regression results.
""")


# ============================================================
# 17. COEFFICIENT ANALYSIS
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
# 18. REGULARIZATION EFFECT
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
# 19. INTERPRETATION
# ============================================================

print("\n==============================")
print("INTERPRETATION")
print("==============================")

print("""
1. Linear Regression
   - Finds best-fit line
   - Uses least squares optimization

2. Normal Equation
   - Closed-form solution
   - No iterative optimization needed

3. Ridge Regression
   - Penalizes large coefficients
   - Helps reduce overfitting

4. Lasso Regression
   - Performs feature selection
   - Some coefficients become zero

5. Residual Analysis
   - Detects model problems
   - Checks assumptions

6. R² Score
   - Measures explained variance
   - Higher is better
""")


# ============================================================
# 20. FINAL SUMMARY
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
✓ Model Assumptions
✓ Coefficient Interpretation
✓ Regularization Concepts

Key Concepts Learned:

✓ Underfitting
✓ Overfitting
✓ Bias-Variance Tradeoff
✓ Regularization
✓ Generalization
✓ Residual Diagnostics
""")

