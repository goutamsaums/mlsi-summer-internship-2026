# Day 3 — Foundations of Supervised Learning Notes

## Overview

Day 3 focuses on the practical foundations of supervised machine learning.
The major goal is understanding how models learn from labeled data and how to evaluate whether a model generalizes well or simply memorizes training data.

This day combines:

* Mathematical foundations
* Python implementations
* Model evaluation workflows
* Bias-variance analysis
* Regularization concepts
* Cross-validation strategies

---

# Supervised Learning

Supervised learning is a machine learning paradigm where:

* Input features (X) are mapped to target outputs (Y)
* The model learns patterns from labeled examples

Two major supervised learning tasks:

1. Regression
2. Classification

---

# Regression

Regression predicts continuous numerical values.

Examples:

* House price prediction
* Temperature forecasting
* Sales prediction

Common algorithms:

* Linear Regression
* Ridge Regression
* Lasso Regression

---

# Classification

Classification predicts categorical labels.

Examples:

* Spam detection
* Disease prediction
* Sentiment analysis

Common algorithms:

* Logistic Regression
* Support Vector Machines
* Decision Trees

---

# Machine Learning Workflow

A proper ML workflow generally follows:

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Train/Validation/Test Split
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Final Testing

---

# Train / Validation / Test Split

Datasets are divided into three subsets.

| Dataset        | Purpose                   |
| -------------- | ------------------------- |
| Training Set   | Learn model parameters    |
| Validation Set | Tune hyperparameters      |
| Test Set       | Final unbiased evaluation |

Typical split:

* 70% Training
* 15% Validation
* 15% Test

---

# Why Validation Sets Matter

Without a validation set:

* Models may overfit training data
* Hyperparameters may be chosen incorrectly
* Performance estimates become biased

Validation sets help:

* Choose model complexity
* Select regularization strength
* Compare multiple models

---

# Bias and Variance

## High Bias

High bias means:

* Model is too simple
* Underfits the data
* Cannot capture patterns

Example:

* Linear model on nonlinear data

Characteristics:

* High training error
* High validation error

---

## High Variance

High variance means:

* Model is too complex
* Overfits the training data
* Learns noise instead of patterns

Example:

* Very high-degree polynomial

Characteristics:

* Very low training error
* High validation error

---

# Bias-Variance Tradeoff

Goal:
Find a balance between:

* Underfitting
* Overfitting

Increasing complexity:

* Reduces bias
* Increases variance

Regularization helps balance both.

---

# Regularization

Regularization prevents overfitting by penalizing large parameter values.

General idea:

[
Cost = Loss + Regularization
]

---

# Ridge Regression (L2)

Adds squared weight penalty.

[
\theta = (X^TX + \lambda I)^{-1}X^Ty
]

Characteristics:

* Shrinks coefficients
* Keeps all features
* Reduces variance

---

# Lasso Regression (L1)

Adds absolute weight penalty.

Characteristics:

* Shrinks some coefficients to zero
* Performs feature selection
* Creates sparse models

---

# Cross Validation

Cross-validation improves model evaluation.

Instead of using one validation split:

* Data is repeatedly split
* Multiple evaluations are averaged

Benefits:

* Better generalization estimate
* Efficient use of data
* Useful for small datasets

---

# K-Fold Cross Validation

Workflow:

1. Split dataset into K folds
2. Train on K−1 folds
3. Validate on remaining fold
4. Repeat K times
5. Average results

Typical choice:

* K = 5
* K = 10

---

# Leave-One-Out Cross Validation

Extreme version of K-Fold CV.

If:
[
K = N
]

Then:

* One sample used for validation
* Remaining used for training

Very expensive computationally.

Used only for very small datasets.

---

# Learning Curves

Learning curves visualize:

* Training performance
* Validation performance
* Dataset size effects

They help diagnose:

* High bias
* High variance
* Need for more data

---

# Performance Metrics

## Regression Metrics

### Mean Squared Error (MSE)

[
MSE = \frac{1}{n}\sum(y_i - \hat{y}_i)^2
]

---

### Root Mean Squared Error (RMSE)

[
RMSE = \sqrt{MSE}
]

---

### Mean Absolute Error (MAE)

[
MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|
]

---

### R² Score

Measures goodness of fit.

[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
]

---

# Logistic Regression

Used for binary classification.

Outputs probabilities using sigmoid function.

Sigmoid:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

Decision rule:

* Probability > 0.5 → Class 1
* Otherwise → Class 0

---

# Classification Metrics

## Accuracy

[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
]

---

## Precision

[
Precision = \frac{TP}{TP + FP}
]

Measures correctness of positive predictions.

---

## Recall

[
Recall = \frac{TP}{TP + FN}
]

Measures ability to detect positives.

---

## F1 Score

[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
]

Useful for imbalanced datasets.

---

# Confusion Matrix

A confusion matrix summarizes:

* True Positives
* False Positives
* True Negatives
* False Negatives

Used to analyze classifier behavior.

---

# ROC Curve

ROC Curve:

* Plots TPR vs FPR
* Evaluates classifier thresholds

AUC (Area Under Curve):

* Higher AUC = better classifier

---

# Feature Selection

Feature selection identifies the most useful features.

Benefits:

* Reduces overfitting
* Improves interpretability
* Reduces computation

---

# Forward Selection

Workflow:

1. Start with no features
2. Add best feature
3. Continue adding features greedily
4. Stop when performance no longer improves

---

# Key Practical Insights

1. More complex models are not always better.
2. Validation performance matters more than training performance.
3. Cross-validation improves reliability.
4. Regularization is essential for controlling variance.
5. Learning curves diagnose model problems.
6. Proper workflow prevents biased evaluation.

---

# Python Libraries Used

```python
numpy
pandas
matplotlib
scikit-learn
```

---

# Day 3 Outcome

By the end of Day 3, you should be able to:

* Build supervised learning workflows
* Split datasets correctly
* Evaluate models properly
* Detect overfitting and underfitting
* Implement regression and classification models
* Use cross-validation effectively
* Understand regularization concepts
* Analyze learning curves and performance metrics

---

# Suggested GitHub Structure

```text
Day3_supervised_learning/
│
├── notebooks/
├── src/
├── data/
├── images/
├── references/
├── notes.md
└── README.md
```
