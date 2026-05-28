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

Learned:

* `zeros()` creates arrays filled with 0
* `ones()` creates arrays filled with 1
* `arange()` creates sequences
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

Learned:

* Positive and negative indexing
* Slicing syntax
* Step size in slicing

---

# 2D Arrays and Matrix Operations

Worked with matrices and multidimensional arrays.

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(matrix.shape)

print(matrix[0, 1])
```

Learned:

* Shape and dimensions
* Row-column indexing
* Matrix structure

---

# Reshape and Transpose

Practiced changing array dimensions.

```python
arr = np.arange(1, 13)

reshaped = arr.reshape(3, 4)

transpose = reshaped.T
```

Learned:

* Reshape changes dimensions
* Transpose swaps rows and columns

---

# Array Concatenation and Splitting

```python
a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

combined = np.concatenate((a, b))
```

Learned:

* Combining arrays
* Splitting arrays into parts

---

# Mathematical Operations

Practiced element-wise operations.

```python
a = np.array([1, 2, 3])

b = np.array([4, 5, 6])

print(a + b)

print(a * b)
```

Learned:

* Addition
* Subtraction
* Multiplication
* Division
* Power operations

---

# Matrix Multiplication

```python
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print(A @ B)
```

Learned:

* Difference between element-wise multiplication and matrix multiplication

---

# Statistical Functions

Practiced basic statistics.

```python
data = np.array([10, 20, 30, 40, 50])

print(np.mean(data))

print(np.std(data))

print(np.min(data))

print(np.max(data))
```

Learned:

* Mean
* Standard deviation
* Minimum and maximum
* Sum and median

---

# Sorting and Searching

```python
arr = np.array([5, 2, 9, 1])

print(np.sort(arr))

print(np.argmax(arr))
```

Learned:

* Sorting arrays
* Finding max/min positions
* Conditional filtering

---

# Random Number Generation

```python
np.random.seed(42)

print(np.random.randint(1, 100, 5))
```

Learned:

* Random integer generation
* Random floating numbers
* Importance of seed values for reproducibility

---

# Broadcasting

Broadcasting was one of the most important concepts today.

```python
arr = np.array([1, 2, 3])

print(arr + 10)
```

Learned:

* Scalar broadcasting
* Operations between arrays of different shapes
* Automatic expansion of dimensions

---

# Broadcasting Between Arrays

```python
a = np.array([[1],
              [2],
              [3]])

b = np.array([10, 20, 30])

print(a + b)
```

Learned:

* Shape compatibility rules
* Row-wise and column-wise operations

---

# Meshgrid Practice

```python
x = np.linspace(-5, 5, 5)

y = np.linspace(-5, 5, 5)

X, Y = np.meshgrid(x, y)
```

Learned:

* Meshgrids are useful for plotting and mathematical surfaces

---

# Files Created

* basic_arrays.py
* mathematical_operations.py
* broadcasting_practice.py
* README.md
* notes.md

---

# Resources Used

Video:
https://youtu.be/DhxKg3jmiis

Lecture Notes:
https://cs231n.github.io/python-numpy-tutorial/

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

# Understanding After Day 1

NumPy makes numerical computation easier and faster in Python.
Broadcasting helps avoid unnecessary loops while working with arrays.
These concepts are important for machine learning, data analysis, and scientific computing.

I still need more practice with advanced broadcasting and multidimensional arrays, but the basics are becoming clearer now.
