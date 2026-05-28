# Day 1 - Introduction to NumPy

## MLSI Lab Research Readiness Bootcamp 2026

### Intern: Goutam Anand

---

# What I Learned Today

Today I started learning NumPy, which is one of the most important Python libraries used in data science, machine learning, and scientific computing.

I learned why NumPy arrays are preferred over normal Python lists, mainly because they are faster and support efficient mathematical operations.

---

# Topics Covered

## Array Creation

Practiced different methods of creating arrays:

```python
np.array()
np.zeros()
np.ones()
np.arange()
np.linspace()
```

Also learned the difference between these methods and when to use them.

---

## Indexing and Slicing

Practiced:

* accessing elements
* row and column selection
* slicing arrays
* negative indexing

Worked with both 1D and 2D arrays.

---

## Reshape and Transpose

Learned how to:

* reshape arrays into different dimensions
* transpose matrices
* convert dimensions for calculations

Used:

```python
reshape()
transpose()
```

---

## Mathematical Operations

Practiced:

* addition
* subtraction
* multiplication
* division

Learned the difference between:

* element-wise operations
* matrix multiplication

---

## Statistical Functions

Used NumPy statistical functions such as:

```python
np.mean()
np.std()
np.min()
np.max()
```

to perform simple data analysis.

---

## Sorting and Searching

Practiced:

* sorting arrays
* finding indices
* conditional searching

using:

```python
np.sort()
np.where()
np.argmax()
np.argmin()
```

---

## Random Number Generation

Learned how to generate random numbers using:

```python
np.random.randint()
np.random.rand()
```

and also understood the use of:

```python
np.random.seed()
```

for reproducibility.

---

## Broadcasting

Practiced broadcasting operations between arrays of different shapes.

Learned how NumPy automatically adjusts dimensions during operations without using loops.

---

## Meshgrid and Normalization

Tried:

```python
np.meshgrid()
```

and also practiced basic normalization using:

```python
(data - mean) / std
```

---

# Exercises Completed

## Exercise 1A

* Array creation methods
* Indexing and slicing
* Reshape and transpose
* Concatenation and splitting

## Exercise 1B

* Element-wise operations
* Matrix multiplication
* Statistical functions
* Sorting and searching
* Random number generation

## Exercise 1C

* Broadcasting basics
* Operations between arrays of different shapes
* Meshgrid generation
* Array normalization

---

# Practice Work Done

* Created multiple NumPy arrays
* Practiced indexing and slicing
* Performed mathematical operations on arrays
* Worked with reshape and transpose
* Tried broadcasting examples
* Practiced basic statistical analysis
* Executed NumPy scripts from GitHub repository using Google Colab

---

# Resources Used

* Corey Schafer NumPy Tutorial
* CS231n NumPy Tutorial
* NumPy Documentation
* Google Colab

---

# Google Colab Execution

## Colab Notebook Link

https://colab.research.google.com/drive/1dsFXSgRI080tspox4NSphmGD3jhLF8gs?usp=sharing

## Commands Used

```python
!git clone https://github.com/goutamsaums/mlsi-summer-internship-2026.git

%cd mlsi-summer-internship-2026/Day1_NumPy

!python numpy_exercises.py
```

---

# Repository Link

https://github.com/goutamsaums/mlsi-summer-internship-2026

---

# Progress Summary

* Created GitHub repository for internship work
* Organized day-wise folder structure
* Completed Day 1 NumPy exercises
* Added practice codes and notes
* Successfully tested execution in Google Colab

---

# My Understanding

NumPy seems very useful for handling numerical data efficiently. I understood how arrays, broadcasting, and vectorized operations make computations easier and faster compared to normal Python lists.

This gave me a good foundation for upcoming topics like Pandas, Machine Learning, and PyTorch.
