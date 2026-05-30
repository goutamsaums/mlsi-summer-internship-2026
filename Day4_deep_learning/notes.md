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

# 1. Artificial Neural Networks

Artificial Neural Networks are computational models inspired by biological neurons.

A neural network consists of:

* Input Layer
* Hidden Layer(s)
* Output Layer

Each neuron receives inputs and performs:

1. Weighted summation
2. Bias addition
3. Activation function

General structure:

Input Layer → Hidden Layers → Output Layer

Purpose:

* Learn nonlinear patterns
* Perform classification
* Perform regression
* Extract meaningful representations from data

---

# 2. Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron is the simplest form of deep neural network.

Characteristics:

* Fully connected layers
* Feedforward architecture
* Nonlinear activations

Applications:

* Binary Classification
* Multi-class Classification
* Regression

Implemented from scratch in Exercise 4A.

---

# 3. Forward Propagation

Forward propagation computes outputs from inputs.

For each layer:

z = Wx + b

a = g(z)

Where:

* W = weights
* x = inputs
* b = bias
* g = activation function

Steps:

1. Compute linear combination
2. Apply activation
3. Pass output to next layer
4. Generate prediction

Purpose:

* Produce predictions
* Compute network output

Implemented manually in Exercise 4A.

---

# 4. Activation Functions

Activation functions introduce nonlinearity.

Without activation functions, neural networks become equivalent to linear models.

---

## Sigmoid

Formula:

σ(z)=1/(1+e⁻ᶻ)

Range:

0 to 1

Uses:

* Binary classification
* Output probabilities

Advantages:

* Smooth
* Interpretable probabilities

Disadvantages:

* Vanishing gradients

---

## ReLU

Formula:

ReLU(z)=max(0,z)

Advantages:

* Fast computation
* Sparse activation
* Reduced vanishing gradient

Disadvantages:

* Dead neurons

Most commonly used activation in deep learning.

---

## Tanh

Formula:

tanh(z)

Range:

-1 to 1

Advantages:

* Zero-centered output
* Better convergence than sigmoid

Disadvantages:

* Vanishing gradient still possible

---

# 5. Loss Functions

Loss functions quantify prediction errors.

For binary classification:

Binary Cross Entropy (BCE)

Purpose:

* Measure model performance
* Guide optimization

Lower loss indicates better predictions.

---

# 6. Backpropagation

Backpropagation calculates gradients using the chain rule.

Process:

1. Forward pass
2. Compute loss
3. Calculate gradients
4. Propagate errors backward
5. Update parameters

Purpose:

* Learn optimal weights
* Reduce loss

Implemented step-by-step in Exercise 4A.

---

# 7. Gradient Descent

Optimization algorithm used to minimize loss.

Update Rule:

θ = θ − α∇J(θ)

Where:

* θ = parameter
* α = learning rate
* J = cost function

Purpose:

* Minimize prediction error
* Improve performance

---

# 8. Mini-Batch Gradient Descent

Training data is divided into batches.

Advantages:

* Faster training
* Reduced memory usage
* Better convergence

Steps:

1. Shuffle data
2. Create batches
3. Train batch-by-batch
4. Update weights

Implemented in Exercise 4A.

---

# 9. Gradient Checking

Used to verify correctness of backpropagation.

Method:

Compare:

* Analytical gradient
* Numerical gradient

Formula:

( J(θ+ε) − J(θ−ε) ) / 2ε

Purpose:

* Validate implementation
* Detect bugs

Implemented in Exercise 4A.

---

# Exercise 4A — Neural Network from Scratch

## Objectives

Implemented:

1. Forward Propagation
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

# Neural Network Architecture

Implemented:

Input Layer:

* 2 neurons

Hidden Layer 1:

* 16 neurons

Hidden Layer 2:

* 8 neurons

Output Layer:

* 1 neuron

Dataset:

Moon Dataset

Generated using:

```python
make_moons()
```

---

# Forward Pass Implementation

Implemented manually using NumPy.

Operations:

* Matrix multiplication
* Bias addition
* Activation functions

Purpose:

* Generate predictions

---

# Backpropagation Implementation

Computed:

* dW1
* db1
* dW2
* db2
* dW3
* db3

Purpose:

* Learn optimal weights

---

# Activation Function Comparison

Compared:

* Sigmoid
* ReLU
* Tanh

Measured:

* Training Loss
* Test Accuracy

Observed:

* ReLU generally converges faster
* Tanh performs well
* Sigmoid trains slower

---

# Exercise 4B — PyTorch Framework Deep Dive

## Objectives

Implemented:

1. torch.nn.Module
2. Convolutional Neural Networks
3. DataLoader
4. Dropout Regularization
5. Weight Decay
6. Learning Rate Scheduling
7. GPU Support
8. Model Saving
9. Model Loading
10. Visualization

---

# 1. PyTorch Framework

PyTorch is an open-source deep learning framework.

Advantages:

* Dynamic computation graph
* GPU acceleration
* Easy debugging
* Large community support

---

# 2. torch.nn.Module

All neural networks inherit from:

```python
torch.nn.Module
```

Purpose:

* Define model architecture
* Manage parameters
* Enable training

Implemented:

```python
class CNN(nn.Module)
```

---

# 3. Convolutional Neural Networks (CNN)

CNNs are specialized neural networks for image processing.

Components:

* Convolution Layers
* Pooling Layers
* Fully Connected Layers

Applications:

* Image Classification
* Object Detection
* Face Recognition

---

# 4. Conv2d Layer

Implemented:

```python
nn.Conv2d()
```

Purpose:

* Extract image features
* Detect patterns

Examples:

* Edges
* Shapes
* Textures

---

# 5. MaxPool2d Layer

Implemented:

```python
nn.MaxPool2d()
```

Purpose:

* Reduce image size
* Reduce computation
* Improve robustness

---

# 6. DataLoader

Implemented:

```python
DataLoader()
```

Purpose:

* Batch processing
* Shuffling
* Efficient loading

Benefits:

* Faster training
* Better memory management

---

# 7. Dropout Regularization

Implemented:

```python
nn.Dropout(0.5)
```

Purpose:

* Prevent overfitting
* Improve generalization

Method:

Randomly deactivate neurons during training.

---

# 8. Weight Decay

Implemented:

```python
weight_decay=1e-4
```

Equivalent to:

L2 Regularization

Benefits:

* Prevents large weights
* Improves generalization

---

# 9. Learning Rate Scheduling

Implemented:

```python
StepLR()
```

Purpose:

Reduce learning rate during training.

Benefits:

* Faster convergence
* Better optimization

---

# 10. GPU Acceleration

Implemented:

```python
torch.device("cuda")
```

Benefits:

* Faster training
* Parallel computation

---

# 11. Model Saving

Implemented:

```python
torch.save()
```

Purpose:

Store trained model parameters.

---

# 12. Model Loading

Implemented:

```python
load_state_dict()
```

Purpose:

Reuse trained models.

---

# Dataset Used

## Exercise 4A

Moon Dataset

Generated using:

```python
make_moons()
```

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
* PyTorch Implementation

Classes:

0–9 handwritten digits

Total Samples:

70,000

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
* Google Colab for execution and experimentation

---

# Google Colab Practice

Used Google Colab to execute all Day 4 exercises directly from GitHub.

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

Colab notebook used for practice and execution:

https://colab.research.google.com/drive/176KNWSUxDfppWkT7MtylUSvdqwBOlROi?usp=sharing

---

# Visualizations Generated

## Exercise 4A

* Training Loss Curve
* Activation Function Comparison
* Decision Boundary Visualization
* Gradient Checking Results

---

## Exercise 4B

* CNN Training Loss Curve
* CNN Accuracy Curve
* Sample Predictions
* Learning Rate Schedule Visualization

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

# Understanding After Day 4

Day 4 provided a complete introduction to modern Deep Learning.

I learned how neural networks process information through multiple layers using forward propagation and how learning occurs through backpropagation.

Implementing a neural network from scratch using NumPy greatly improved understanding of gradients, activation functions, and optimization.

The comparison of Sigmoid, ReLU, and Tanh helped clarify why activation functions are important and how they influence learning.

Using PyTorch demonstrated how professional deep learning systems are developed efficiently using high-level APIs while still relying on the same mathematical foundations.

I also gained practical experience with:

* CNN architectures
* Image classification
* Regularization techniques
* Learning rate scheduling
* GPU training
* Model persistence

Although I still need more practice with:

* Advanced CNN architectures
* Transfer Learning
* Transformers
* Large-scale Deep Learning Systems

the foundational concepts of Deep Learning are now much clearer and easier to implement.

---

# Final Summary

Day 4 established the complete foundation of Deep Learning:

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
