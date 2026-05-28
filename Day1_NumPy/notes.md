# Day 1 Notes - NumPy Fundamentals

## Introduction

Today I started working on NumPy fundamentals for the MLSI Lab Summer Internship preparation tasks.
I practiced basic array operations, mathematical functions, and broadcasting concepts using Python and Google Colab.

The main objective today was to become comfortable with NumPy arrays and understand how numerical operations are performed efficiently in Python.

---

# Topics Learned

## 1. Introduction to NumPy

* NumPy is used for numerical and scientific computing.
* NumPy arrays are faster and more efficient than Python lists.
* Arrays allow vectorized operations and matrix computations.
* NumPy is widely used in machine learning, data science, and scientific applications.

---

# Array Creation Methods

Practiced different ways of creating arrays:

```python
import numpy as np

a = np.array([1, 2, 3])

zeros = np.zeros((2, 3))

ones = np.ones((3, 2))

arr = np.arange(0, 10, 2)

line = np.linspace(0, 1, 5)
```

### Learned

* `array()` creates NumPy arrays
* `zeros()` creates arrays filled with 0
* `ones()` creates arrays filled with 1
* `arange()` creates sequences with intervals
* `linspace()` creates evenly spaced values

---

# Indexing and Slicing

Practiced accessing elements and subarrays.

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])

print(arr[1:4])

print(arr[::-1])
```

### Learned

* Positive indexing
* Negative indexing
* Array slicing
* Step size in slicing
* Reverse indexing

---

# 2D Arrays and Matrix Operations

Worked with multidimensional arrays.

```python
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.shape)

print(matrix[0, 1])
```

### Learned

* Shape of arrays
* Dimensions of arrays
* Row-column indexing
* Matrix representation in NumPy

---

# Reshape and Transpose

Practiced changing array dimensions.

```python
arr = np.arange(1, 13)

reshaped = arr.reshape(3, 4)

transpose = reshaped.T
```

### Learned

* Reshape changes dimensions without changing data
* Transpose swaps rows and columns
* Importance of dimensions in matrix operations

---

# Array Concatenation and Splitting

```python
a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

combined = np.concatenate((a, b))
```

### Learned

* Combining arrays
* Horizontal and vertical stacking
* Splitting arrays into smaller parts

---

# Mathematical Operations

Practiced element-wise operations.

```python
a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

print(a + b)

print(a * b)
```

### Learned

* Addition
* Subtraction
* Multiplication
* Division
* Power operations
* Element-wise calculations

---

# Matrix Multiplication

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A @ B)
```

### Learned

* Difference between element-wise multiplication and matrix multiplication
* Matrix multiplication using `@`
* Importance of matrix dimensions

---

# Statistical Functions

Practiced statistical operations on arrays.

```python
data = np.array([10, 20, 30, 40, 50])

print(np.mean(data))

print(np.std(data))

print(np.min(data))

print(np.max(data))
```

### Learned

* Mean
* Standard deviation
* Minimum and maximum
* Median
* Sum of array elements

---

# Sorting and Searching

```python
arr = np.array([5, 2, 9, 1])

print(np.sort(arr))

print(np.argmax(arr))
```

### Learned

* Sorting arrays
* Finding maximum and minimum positions
* Conditional filtering
* Searching inside arrays

---

# Random Number Generation

```python
np.random.seed(42)

print(np.random.randint(1, 100, 5))
```

### Learned

* Random integer generation
* Random floating point values
* Reproducibility using seed values

---

# Broadcasting

Broadcasting was one of the most important concepts today.

```python
arr = np.array([1, 2, 3])

print(arr + 10)
```

### Learned

* Scalar broadcasting
* Automatic expansion of dimensions
* Broadcasting avoids unnecessary loops

---

# Broadcasting Between Arrays

```python
a = np.array([
    [1],
    [2],
    [3]
])

b = np.array([10, 20, 30])

print(a + b)
```

### Learned

* Broadcasting rules
* Shape compatibility
* Row-wise and column-wise operations

---

# Meshgrid Practice

```python
x = np.linspace(-5, 5, 5)

y = np.linspace(-5, 5, 5)

X, Y = np.meshgrid(x, y)
```

### Learned

* Meshgrids are useful for plotting
* Helpful for mathematical surfaces and visualization

---

# Files Created

* basic_arrays.py
* mathematical_operations.py
* broadcasting_practice.py
* README.md
* notes.md

---

# Resources Used

* MLSI Lab NumPy introductory lecture
* CS231n NumPy Tutorial
* Corey Schafer Python tutorials
* Google Colab for execution and testing

---

# Google Colab Practice

Used Google Colab to execute files directly from GitHub repository.

Commands used:

```python
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

%cd mlsi-summer-internship-2026/Day1_NumPy

!python basic_arrays.py
```

---

# Google Colab Notebook

Colab notebook used for practice and execution:

https://colab.research.google.com/drive/1dsFXSgRI080tspox4NSphmGD3jhLF8gs?usp=sharing

---

# Understanding After Day 1

NumPy makes numerical computation easier and faster in Python.
Broadcasting helps avoid unnecessary loops while working with arrays.
These concepts are important for machine learning, data analysis, and scientific computing.

I still need more practice with advanced broadcasting and multidimensional arrays, but the basics are becoming clearer now.
