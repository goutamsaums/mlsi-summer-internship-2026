# Day 3 — Foundations of Supervised Learning

## Overview

Day 3 focused on the practical foundations of supervised learning and model evaluation.

The goal was to understand how machine learning models are:

* Trained
* Validated
* Evaluated
* Regularized
* Improved using proper ML workflows

This day combined theoretical understanding with practical Python implementations using NumPy, Matplotlib, and Scikit-learn.

Topics covered included:

* Supervised Learning
* Train / Validation / Test Splits
* Cross Validation
* Bias-Variance Tradeoff
* Learning Curves
* Linear Regression
* Ridge Regression
* Lasso Regression
* Logistic Regression
* Classification Metrics
* Confusion Matrix
* ROC Curve
* Model Evaluation

The exercises were divided into:

* Exercise 3A → ML Workflow
* Exercise 3B → Linear Regression
* Exercise 3C → Classification Fundamentals

---

# Learning Resources

## Videos

1. Foundations of Supervised Learning
2. Bias vs Variance
3. Regularization and Validation
4. Logistic Regression and Classification

---

# Core Concepts Covered

# 1. Supervised Learning

Supervised learning learns a mapping:

[
X \rightarrow y
]

Where:

* (X) = input features
* (y) = target/output

The model learns patterns from labeled data and predicts outputs for unseen examples.

---

## Types of Supervised Learning

### Regression

Used when the output variable is continuous.

Examples:

* House price prediction
* Temperature forecasting
* Stock price prediction

---

### Classification

Used when the output variable is categorical.

Examples:

* Spam detection
* Disease classification
* Sentiment analysis

---

# 2. Bias-Variance Tradeoff

One of the most important concepts in machine learning.

It explains the balance between:

* Simplicity of the model
* Ability to generalize to unseen data

---

## High Bias → Underfitting

Occurs when the model is too simple.

Characteristics:

* High training error
* High validation error
* Poor learning capability

Example:

* Linear model for nonlinear data

---

## High Variance → Overfitting

Occurs when the model memorizes training data.

Characteristics:

* Very low training error
* High validation/test error
* Poor generalization

---

## Good Generalization

A balanced model:

* Learns patterns
* Avoids memorization
* Performs well on unseen data

---

# 3. Train / Validation / Test Split

Datasets are divided into separate subsets.

| Dataset Split        | Purpose                |
| -------------------- | ---------------------- |
| Train Set            | Learn model parameters |
| Validation (Dev) Set | Tune hyperparameters   |
| Test Set             | Final evaluation       |

Typical split:

* 70% Training
* 15% Validation
* 15% Testing

Implemented using:

```python
train_test_split()
```

---

# 4. Cross Validation

Used when datasets are small.

## K-Fold Cross Validation

The dataset is divided into K equal parts.

Process:

1. Train on K-1 folds
2. Validate on remaining fold
3. Repeat K times
4. Average the results

Benefits:

* Better use of limited data
* More reliable evaluation
* Reduces variance in estimates

---

# 5. Learning Curves

Learning curves compare:

* Training performance
* Validation performance

Used to detect:

* Underfitting
* Overfitting
* Data sufficiency

---

# Exercise 3A — ML Workflow

## Objectives

Implemented:

1. Train-validation-test split
2. Cross-validation from scratch
3. Performance metrics
4. Bias-variance tradeoff analysis
5. Learning curves

---

# 1. Train-Validation-Test Split

Used:

```python
train_test_split()
```

Purpose:

* Separate training and evaluation data
* Prevent data leakage
* Estimate generalization performance

---

# 2. Cross Validation from Scratch

Implemented manual K-Fold cross validation using Python.

Included:

* Manual splitting
* Training loop
* Validation averaging

Concepts learned:

* Robust model evaluation
* Better use of limited data
* Avoiding overfitting

---

# 3. Performance Metrics

Implemented:

* MAE
* MSE
* RMSE
* R² Score

Purpose:

* Quantify model performance
* Compare models

---

# 4. Bias-Variance Tradeoff

Observed using:

* Training error
* Validation error
* Learning curves
* Model complexity demonstrations

Practiced:

* Underfitting
* Overfitting
* Proper model complexity

---

# 5. Learning Curves

Generated:

* Training score curves
* Validation score curves

Used to analyze:

* Model complexity
* Dataset sufficiency
* Generalization capability

---

# Exercise 3B — Linear Regression

## Objectives

Implemented complete Linear Regression workflow.

Included:

* Linear Regression from scratch
* Ridge Regression
* Lasso Regression
* Residual Analysis
* Scikit-learn comparison

---

# 1. Linear Regression from Scratch

Used the Normal Equation:

[
\theta = (X^TX)^{-1}X^Ty
]

Purpose:

* Closed-form parameter estimation
* Analytical solution

Advantages:

* Exact solution
* No iterative optimization required

---

# 2. Ridge Regression (L2 Regularization)

Penalty term:

[
\lambda \sum \theta^2
]

Effects:

* Reduces variance
* Prevents overfitting
* Shrinks coefficients

---

# 3. Lasso Regression (L1 Regularization)

Penalty term:

[
\lambda \sum |\theta|
]

Effects:

* Performs feature selection
* Produces sparse coefficients

---

# 4. Scikit-Learn Comparison

Compared manual implementation with:

```python
LinearRegression()
Ridge()
Lasso()
```

Purpose:

* Verify implementation correctness
* Compare model performance

---

# 5. Residual Analysis

Residual:

[
Residual = y - \hat{y}
]

Analyzed:

* Residual distributions
* Prediction errors
* Model assumptions

Generated visualizations:

* Residual Plot
* Histogram of Residuals
* Actual vs Predicted Plot
* Learning Curves

---

# Linear Regression Assumptions

## Linearity

Relationship between features and target should be linear.

---

## Independence

Observations should be independent.

---

## Homoscedasticity

Residual variance should remain constant.

---

## Normality

Residuals should be approximately normally distributed.

---

# Exercise 3C — Classification Fundamentals

## Objectives

Implemented Logistic Regression and classification evaluation techniques.

Included:

* Logistic Regression from scratch
* Gradient Descent
* Confusion Matrix
* ROC Curve
* Classification Metrics

---

# 1. Logistic Regression from Scratch

Used Sigmoid Function:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

Purpose:

* Convert outputs into probabilities

---

# 2. Gradient Descent

Parameter update rule:

[
\theta = \theta - \alpha \nabla J(\theta)
]

Learned:

* Optimization
* Cost minimization
* Convergence

---

# 3. Classification Metrics

Implemented:

## Accuracy

[
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
]

Measures overall correctness.

---

## Precision

[
Precision = \frac{TP}{TP+FP}
]

Measures quality of positive predictions.

---

## Recall

[
Recall = \frac{TP}{TP+FN}
]

Measures ability to detect positive cases.

---

## F1 Score

Harmonic mean of precision and recall.

---

# 4. Confusion Matrix

Visualized:

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

Purpose:

* Understand classification errors
* Analyze false positives and false negatives

---

# 5. ROC Curve and AUC

ROC Curve:

* TPR vs FPR

AUC:

* Measures overall classifier quality

Higher AUC indicates:

* Better classification performance

Generated visualizations:

* ROC Curve
* Confusion Matrix Heatmap
* Classification Metrics Bar Chart
* Gradient Descent Cost Plot
* Learning Curves
* Bias-Variance Demonstration

---

# 6. Scikit-Learn Comparison

Compared custom Logistic Regression implementation with:

```python
LogisticRegression()
```

Purpose:

* Validate implementation
* Compare metrics and predictions

---

# Python Files Created

```text
Day3_supervised_learning/

│
├── notes.md
├── exercise_3A_ml_workflow.py
├── exercise_3B_linear_regression.py
└── exercise_3C_classification_fundamentals.py
```

---

# Libraries Used

```python
numpy
pandas
matplotlib
scikit-learn
```

---

# Resources Used

* Foundations of Supervised Learning lecture videos
* Stanford CS229 concepts on Bias-Variance Tradeoff and Regularization
* NumPy documentation
* Scikit-learn documentation
* Matplotlib documentation
* Google Colab for execution and experimentation

---

# Google Colab Practice

Used Google Colab to execute Python implementations directly from the GitHub repository.

Commands used:

```python
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

%cd /content/mlsi-summer-internship-2026/Day3_supervised_learning

!pip install numpy pandas matplotlib scikit-learn

%matplotlib inline

!python exercise_3A_ml_workflow.py

!python exercise_3B_linear_regression.py

!python exercise_3C_classification_fundamentals.py
```

---

# Google Colab Notebook

Colab notebook used for practice and execution:

https://colab.research.google.com/drive/1tVmW_bydZyI9JsIw3Ol9Jj-USZPdpRVf?usp=sharing

---

# Datasets Used

## Exercise 3A — ML Workflow

California Housing Dataset

Used for:

* Train-validation-test split
* Cross-validation
* Bias-variance analysis
* Learning curves
* Performance metrics

---

## Exercise 3B — Linear Regression

California Housing Dataset

Used for:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Residual analysis
* Model comparison

---

## Exercise 3C — Classification Fundamentals

Breast Cancer Wisconsin Dataset

Used for:

* Logistic Regression
* Classification metrics
* Confusion matrix
* ROC curve
* Precision, Recall, and F1-score

---

# Visualizations Generated

## Exercise 3A

* Learning Curves
* Residual Plot
* Bias-Variance Visualization

---

## Exercise 3B

* Residual Plot
* Residual Distribution Histogram
* Actual vs Predicted Plot
* Learning Curves

---

## Exercise 3C

* ROC Curve
* Confusion Matrix Heatmap
* Classification Metrics Bar Chart
* Gradient Descent Cost Reduction Plot
* Learning Curves
* Bias-Variance Demonstration

---

# Practical Understanding Developed

By completing Day 3 exercises, the following practical skills were developed:

* Building ML workflows
* Splitting datasets properly
* Evaluating models correctly
* Understanding overfitting
* Using regularization
* Implementing regression algorithms
* Implementing classification algorithms
* Creating learning curves
* Using validation techniques
* Comparing manual ML implementations with Scikit-learn

---

# Understanding After Day 3

Day 3 helped build a strong understanding of the complete supervised learning workflow used in machine learning systems.

I learned how machine learning models are properly trained, validated, evaluated, and improved using train-validation-test splits and cross-validation techniques.

The concepts of bias and variance became much clearer through practical implementation and visualization using learning curves and polynomial models.

I understood how overfitting happens when models become too complex and how regularization techniques like Ridge and Lasso help reduce variance and improve generalization.

Implementing Linear Regression using the Normal Equation improved understanding of analytical solutions, while Logistic Regression from scratch helped clarify how gradient descent and sigmoid functions work in classification problems.

I also practiced important evaluation metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* AUC Score

Comparing manual implementations with Scikit-learn helped verify correctness and understand how machine learning libraries simplify model development.

I still need more practice with:

* Hyperparameter tuning
* Advanced regularization methods
* Large-scale datasets
* More complex classification problems

But the complete supervised learning workflow and evaluation methodology are now much clearer and more practical to implement.

---

# Final Summary

Day 3 established the complete supervised learning workflow used throughout machine learning:

1. Prepare datasets
2. Split training and evaluation data
3. Train models
4. Validate performance
5. Tune hyperparameters
6. Prevent overfitting
7. Evaluate properly
8. Generalize to unseen data

These concepts form the foundation for:

* Decision Trees
* Random Forests
* Support Vector Machines
* Neural Networks
* Deep Learning
* Advanced Machine Learning Systems
