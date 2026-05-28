# Day 3 — Foundations of Supervised Learning

## Google Colab Notebook

Complete Day 3 implementation notebook:

https://colab.research.google.com/drive/1tVmW_bydZyI9JsIw3Ol9Jj-USZPdpRVf?usp=sharing

The notebook contains:

* Exercise 3A — ML Workflow
* Exercise 3B — Linear Regression
* Exercise 3C — Classification Fundamentals

---

# Overview

Day 3 focuses on the practical foundations of supervised learning.
The primary objective is to understand how machine learning models are trained, validated, evaluated, regularized, and improved using proper workflows and statistical reasoning.

This day combines:

* ML workflow fundamentals
* Bias-variance tradeoff
* Cross-validation
* Learning curves
* Linear Regression
* Regularization techniques
* Logistic Regression
* Classification metrics
* Model evaluation methods

The exercises are divided into:

* **Exercise 3A** → ML Workflow
* **Exercise 3B** → Linear Regression
* **Exercise 3C** → Classification Fundamentals

---

# Learning Resources

## Video References

1. Foundations of Supervised Learning
2. Bias vs Variance
3. Regularization and Validation
4. Logistic Regression and Classification

---

# Core Concepts Covered

---

# 1. Supervised Learning

Supervised learning learns a mapping:

[
X \rightarrow y
]

Where:

* (X) = Input features
* (y) = Target/output variable

The model learns from labeled data and attempts to generalize to unseen data.

---

## Types of Supervised Learning

### Regression

Predicts continuous numerical values.

Examples:

* House price prediction
* Temperature forecasting
* Sales prediction

---

### Classification

Predicts categorical labels.

Examples:

* Spam detection
* Disease classification
* Fraud detection

---

# 2. Bias-Variance Tradeoff

The bias-variance tradeoff is one of the most important concepts in machine learning.

---

## High Bias → Underfitting

The model is too simple to capture the true relationship in the data.

Characteristics:

* High training error
* High validation error
* Poor learning capability

Example:

* Linear model applied to nonlinear data

---

## High Variance → Overfitting

The model memorizes training data instead of learning generalized patterns.

Characteristics:

* Very low training error
* High validation/test error
* Poor generalization

---

## Good Generalization

A properly balanced model:

* Learns meaningful patterns
* Avoids memorization
* Performs well on unseen data

---

# 3. Train / Validation / Test Split

Datasets are divided into:

| Split                | Purpose                   |
| -------------------- | ------------------------- |
| Training Set         | Learn model parameters    |
| Validation (Dev) Set | Tune hyperparameters      |
| Test Set             | Final unbiased evaluation |

Typical implementation:

```python
from sklearn.model_selection import train_test_split
```

Example split:

* 70% Training
* 15% Validation
* 15% Test

---

# 4. Cross Validation

Cross-validation is used for reliable model evaluation.

---

## K-Fold Cross Validation

The dataset is divided into (K) folds.

Workflow:

1. Train on (K-1) folds
2. Validate on remaining fold
3. Repeat (K) times
4. Average results

Advantages:

* Better use of limited data
* Reliable validation
* Reduced evaluation bias

---

# 5. Learning Curves

Learning curves compare:

* Training performance
* Validation performance

Used to analyze:

* Underfitting
* Overfitting
* Model complexity
* Data sufficiency

---

# Exercise 3A — ML Workflow

## Objectives

Implemented:

* Train-validation-test split
* Cross-validation from scratch
* Performance metrics
* Bias-variance analysis
* Learning curves

---

# 1. Train-Validation-Test Split

Used:

```python
train_test_split()
```

Purpose:

* Separate training and evaluation data
* Prevent information leakage
* Improve model generalization evaluation

---

# 2. Cross Validation from Scratch

Implemented manually:

* K-Fold splitting
* Training loops
* Validation averaging

Concept learned:

* Robust evaluation methodology

---

# 3. Performance Metrics

Implemented:

* MAE
* MSE
* RMSE
* Accuracy

Purpose:

* Quantitative performance measurement
* Model comparison

---

# 4. Bias-Variance Tradeoff

Observed:

* Underfitting
* Overfitting
* Proper generalization

Using:

* Polynomial regression models
* Validation error analysis

---

# 5. Learning Curves

Generated:

* Training score curves
* Validation score curves

Used to analyze:

* Model learning behavior
* Data sufficiency
* Overfitting trends

---

# Exercise 3B — Linear Regression

## Objectives

Implemented complete Linear Regression workflow.

---

# 1. Linear Regression from Scratch

Used the Normal Equation:

[
\theta = (X^TX)^{-1}X^Ty
]

Purpose:

* Closed-form parameter estimation

Advantages:

* Exact analytical solution
* No iterative optimization required

---

# 2. Regularization

Implemented:

* Ridge Regression (L2)
* Lasso Regression (L1)

---

## Ridge Regression (L2)

Penalty term:

[
\lambda \sum \theta^2
]

Effect:

* Reduces variance
* Prevents overfitting
* Shrinks coefficients smoothly

---

## Lasso Regression (L1)

Penalty term:

[
\lambda \sum |\theta|
]

Effect:

* Performs feature selection
* Produces sparse coefficients

---

# 3. Scikit-Learn Comparison

Compared custom implementation with:

```python
LinearRegression()
Ridge()
Lasso()
```

Purpose:

* Validate correctness
* Compare outputs and performance

---

# 4. Residual Analysis

Residual:

[
Residual = y - \hat{y}
]

Analyzed:

* Residual distributions
* Prediction errors
* Model assumptions

---

# Linear Regression Assumptions

## Linearity

Relationship between variables should be linear.

## Independence

Observations should be independent.

## Homoscedasticity

Residual variance should remain constant.

## Normality

Residuals should be approximately normally distributed.

---

# Exercise 3C — Classification Fundamentals

## Objectives

Implemented Logistic Regression and classification evaluation methods.

---

# 1. Logistic Regression from Scratch

Used the Sigmoid Function:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

Purpose:

* Convert linear outputs into probabilities

---

# 2. Gradient Descent

Parameter update rule:

[
\theta = \theta - \alpha \nabla J(\theta)
]

Concepts learned:

* Optimization
* Cost minimization
* Convergence behavior

---

# 3. Classification Metrics

Implemented:

* Accuracy
* Precision
* Recall
* F1-score

---

## Accuracy

[
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
]

---

## Precision

[
Precision = \frac{TP}{TP + FP}
]

Measures positive prediction quality.

---

## Recall

[
Recall = \frac{TP}{TP + FN}
]

Measures ability to identify positive cases.

---

## F1 Score

Harmonic mean of Precision and Recall.

---

# 4. Confusion Matrix

Visualized classification outcomes:

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

Purpose:

* Understand prediction errors
* Evaluate classifier quality

---

# 5. ROC Curve and AUC

ROC Curve:

* Plots TPR vs FPR

AUC:

* Measures overall classification quality

Higher AUC:

* Better classifier performance

---

# 6. Scikit-Learn Comparison

Compared custom implementation with:

```python
LogisticRegression()
```

Purpose:

* Validate implementation
* Compare classification performance

---

# Python Implementations

The Day 3 implementation includes the following Python programs:

---

## exercise_3A_ml_workflow.py

Contains:

* Dataset splitting
* Cross-validation implementation
* Performance metrics
* Bias-variance analysis
* Learning curves

---

## exercise_3B_linear_regression.py

Contains:

* Linear Regression from scratch
* Normal Equation
* Ridge Regression
* Lasso Regression
* Residual analysis
* Scikit-learn comparison

---

## exercise_3C_classification_fundamentals.py

Contains:

* Logistic Regression from scratch
* Gradient Descent
* Confusion Matrix
* ROC Curve
* Precision / Recall / F1-score
* Scikit-learn comparison

---

# GitHub Execution Workflow

## Step 1 — Clone Repository

```bash
git clone <repository_link>
cd Day3_supervised_learning
```

---

## Step 2 — Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Step 3 — Run Python Programs

### Execute Exercise 3A

```bash
python exercise_3A_ml_workflow.py
```

---

### Execute Exercise 3B

```bash
python exercise_3B_linear_regression.py
```

---

### Execute Exercise 3C

```bash
python exercise_3C_classification_fundamentals.py
```

---

# Outputs Generated

The programs generate:

* Training metrics
* Validation metrics
* Learning curves
* Residual plots
* Confusion matrices
* ROC curves
* Accuracy reports
* Model comparison outputs

---

# Libraries Used

```python
numpy
pandas
matplotlib
scikit-learn
```

---

# Datasets Used

| Exercise    | Dataset                    |
| ----------- | -------------------------- |
| Exercise 3A | Synthetic datasets         |
| Exercise 3B | California Housing Dataset |
| Exercise 3C | Breast Cancer Dataset      |

---

# File Structure

```text
Day3_supervised_learning/

│
├── notes.md
├── exercise_3A_ml_workflow.py
├── exercise_3B_linear_regression.py
├── exercise_3C_classification_fundamentals.py

```

---

# Practical Skills Developed

By completing Day 3 exercises, the following practical skills were developed:

* Building ML workflows
* Splitting datasets correctly
* Evaluating models properly
* Understanding overfitting and underfitting
* Using regularization
* Implementing regression algorithms
* Implementing classification algorithms
* Creating learning curves
* Applying validation techniques
* Comparing manual ML with Scikit-learn

---

# Final Summary

Day 3 established the core supervised learning workflow used throughout machine learning:

1. Prepare data
2. Split datasets
3. Train models
4. Validate performance
5. Tune hyperparameters
6. Prevent overfitting
7. Evaluate properly
8. Generalize to unseen data

These concepts form the foundation for:

* Decision Trees
* Random Forests
* Support Vector Machines (SVMs)
* Neural Networks
* Deep Learning
* Advanced machine learning systems
