# Day 4 — Foundations of Deep Learning

## Overview

Day 4 focused on understanding the foundations of Deep Learning and Neural Networks through both theoretical concepts and practical implementations.

The goal was to understand how deep learning models:

* Learn from data
* Perform forward propagation
* Compute prediction errors
* Update parameters using backpropagation
* Optimize using gradient descent
* Scale using modern deep learning frameworks

This day combined:

* Mathematical foundations
* Neural network implementation from scratch using NumPy
* Deep Learning implementation using PyTorch

Topics covered included:

* Artificial Neural Networks (ANN)
* Multi-Layer Perceptrons (MLP)
* Forward Propagation
* Backpropagation
* Activation Functions
* Loss Functions
* Gradient Descent
* Mini-Batch Gradient Descent
* Gradient Checking
* PyTorch Framework
* Convolutional Neural Networks (CNN)
* Data Loaders
* Regularization Techniques
* Learning Rate Scheduling
* GPU Acceleration
* Model Saving and Loading

The exercises were divided into:

* Exercise 4A → Neural Network from Scratch
* Exercise 4B → PyTorch Framework Deep Dive

---

# Learning Resources

## Videos

1. Introduction to Neural Networks
2. Forward Propagation and Backpropagation
3. Activation Functions Explained
4. Deep Learning Fundamentals
5. PyTorch Framework Introduction
6. Convolutional Neural Networks
7. Optimization and Learning Rate Scheduling

---

# Core Concepts Covered

## 1. Artificial Neural Networks

Artificial Neural Networks are computational models inspired by biological neurons.

A neural network consists of:

* Input Layer
* Hidden Layer(s)
* Output Layer

Each neuron performs:

1. Weighted summation
2. Bias addition
3. Activation function

Purpose:

* Learn nonlinear relationships
* Perform classification
* Perform regression
* Extract meaningful patterns from data

---

## 2. Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron is the simplest deep neural network architecture.

Characteristics:

* Fully connected layers
* Feedforward architecture
* Nonlinear activation functions

Applications:

* Binary classification
* Multi-class classification
* Regression

Implemented from scratch in Exercise 4A.

---

## 3. Forward Propagation

Forward propagation computes outputs from inputs.

For each layer:

z = Wx + b

a = g(z)

Where:

* W = weights
* x = input
* b = bias
* g = activation function

Purpose:

* Generate predictions
* Compute network output

Implemented manually using NumPy.

---

## 4. Activation Functions

Activation functions introduce nonlinearity into neural networks.

### Sigmoid

Formula:

σ(z) = 1 / (1 + e⁻ᶻ)

Range:

* 0 to 1

Uses:

* Binary classification
* Probability outputs

Advantages:

* Smooth
* Probabilistic interpretation

Disadvantages:

* Vanishing gradients

---

### ReLU

Formula:

ReLU(z) = max(0, z)

Advantages:

* Fast computation
* Sparse activation
* Reduced vanishing gradient

Disadvantages:

* Dead neurons

Most commonly used activation function.

---

### Tanh

Formula:

tanh(z)

Range:

-1 to 1

Advantages:

* Zero-centered output
* Better convergence than sigmoid

Disadvantages:

* Can still suffer from vanishing gradients

---

## 5. Loss Functions

Loss functions quantify prediction errors.

Used:

### Binary Cross Entropy (BCE)

Purpose:

* Measure classification performance
* Guide optimization

Lower loss indicates better predictions.

---

## 6. Backpropagation

Backpropagation computes gradients using the chain rule.

Steps:

1. Forward pass
2. Compute loss
3. Compute gradients
4. Propagate errors backward
5. Update weights

Purpose:

* Learn optimal parameters
* Reduce prediction error

Implemented step-by-step in Exercise 4A.

---

## 7. Gradient Descent

Optimization algorithm used to minimize loss.

Update Rule:

θ = θ − α∇J(θ)

Where:

* θ = parameter
* α = learning rate
* J = cost function

Purpose:

* Improve model performance
* Minimize loss

---

## 8. Mini-Batch Gradient Descent

Training data is divided into batches.

Advantages:

* Faster training
* Lower memory usage
* Better convergence

Implemented in Exercise 4A.

---

## 9. Gradient Checking

Used to verify correctness of backpropagation.

Method:

Compare:

* Analytical gradients
* Numerical gradients

Purpose:

* Validate implementation
* Detect programming errors

Implemented in Exercise 4A.

---

# Exercise 4A — Neural Network from Scratch

## Objectives

Implemented:

1. Forward Pass
2. Backpropagation
3. Sigmoid Activation
4. ReLU Activation
5. Tanh Activation
6. Binary Cross Entropy Loss
7. Mini-Batch Gradient Descent
8. Gradient Checking
9. Accuracy Evaluation
10. Activation Comparison

---

## Neural Network Architecture

* Input Layer → 2 neurons
* Hidden Layer 1 → 16 neurons
* Hidden Layer 2 → 8 neurons
* Output Layer → 1 neuron

Dataset:

* Moon Dataset (`make_moons()`)

---

## Concepts Practiced

* Matrix multiplication
* Activation functions
* Gradient computation
* Parameter updates
* Classification accuracy
* Loss minimization

---

# Exercise 4B — PyTorch Framework Deep Dive

## Objectives

Implemented:

1. torch.nn.Module
2. CNN using Conv2d
3. MaxPool2d
4. DataLoader
5. Dropout
6. Weight Decay
7. Learning Rate Scheduling
8. GPU Support
9. Model Saving
10. Model Loading

---

## Convolutional Neural Networks (CNN)

CNNs are specialized neural networks for image processing.

Components:

* Convolution Layers
* Pooling Layers
* Fully Connected Layers

Applications:

* Image Classification
* Face Recognition
* Object Detection

---

## PyTorch Features Used

### torch.nn.Module

Used to define custom neural networks.

### Conv2d

Used for feature extraction.

### MaxPool2d

Used for dimensionality reduction.

### DataLoader

Used for batch loading and shuffling.

### Dropout

Used to reduce overfitting.

### Weight Decay

Implements L2 regularization.

### Learning Rate Scheduler

Implemented using StepLR.

### GPU Support

Implemented using:

```python
torch.device("cuda")
```

### Model Saving

```python
torch.save()
```

### Model Loading

```python
load_state_dict()
```

---

# Datasets Used

## Exercise 4A

Moon Dataset

Used for:

* Binary Classification
* Activation Comparison
* Gradient Checking

---

## Exercise 4B

MNIST Dataset

Used for:

* CNN Training
* Digit Classification

Classes:

* Digits 0–9

Total Samples:

* 70,000

---

# Python Files Created

```text
Day4_deep_learning/

│
├── notes.md
├── exercise_4A_neural_network_from_scratch.py
└── exercise_4B_pytorch_framework.py
```

---

# Libraries Used

```python
numpy
matplotlib
scikit-learn
torch
torchvision
```

---

# Resources Used

* Deep Learning lecture videos
* Neural Network tutorials
* NumPy documentation
* PyTorch documentation
* Torchvision documentation
* Google Colab

---

# Google Colab Practice

Commands used:

```python
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

%cd /content/mlsi-summer-internship-2026/Day4_deep_learning

!pip install numpy matplotlib scikit-learn torch torchvision

%matplotlib inline

!python exercise_4A_neural_network_from_scratch.py

!python exercise_4B_pytorch_framework.py
```

---

# Google Colab Notebook

https://colab.research.google.com/drive/176KNWSUxDfppWkT7MtylUSvdqwBOlROi?usp=sharing

---

# Practical Understanding Developed

By completing Day 4 exercises, the following practical skills were developed:

* Building neural networks from scratch
* Implementing forward propagation
* Implementing backpropagation
* Understanding gradient descent
* Comparing activation functions
* Working with mini-batch training
* Performing gradient checking
* Using PyTorch effectively
* Building CNN architectures
* Working with image datasets
* Using DataLoaders
* Applying regularization
* Using learning rate schedulers
* Saving and loading models
* Utilizing GPU acceleration

---

# Final Summary

Day 4 established the foundation of modern Deep Learning:

1. Build Neural Networks
2. Perform Forward Propagation
3. Compute Loss
4. Apply Backpropagation
5. Optimize using Gradient Descent
6. Train using Mini-Batches
7. Validate with Gradient Checking
8. Use Deep Learning Frameworks
9. Build CNNs
10. Apply Regularization
11. Schedule Learning Rates
12. Save and Deploy Models

These concepts form the foundation for:

* Computer Vision
* Natural Language Processing
* Generative AI
* Transformers
* Large Language Models (LLMs)
* Reinforcement Learning
* Modern AI Systems

---

**Day 4 Completed Successfully.**
