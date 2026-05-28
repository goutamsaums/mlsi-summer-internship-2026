# Day 1 - NumPy Fundamentals
# mathematical_operations.py
# MLSI Lab Summer Internship 2026
# Goutam Anand

import numpy as np

print("========== NUMPY MATHEMATICAL OPERATIONS ==========\n")

# ---------------------------------------------------
# 1. Element-wise Operations
# ---------------------------------------------------

print("1. Element-wise Operations\n")

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("Array A:")
print(a)

print("Array B:")
print(b)
print()

print("Addition:")
print(a + b)
print()

print("Subtraction:")
print(a - b)
print()

print("Multiplication:")
print(a * b)
print()

print("Division:")
print(a / b)
print()

print("Power:")
print(a ** 2)
print()

# ---------------------------------------------------
# 2. Matrix Operations
# ---------------------------------------------------

print("2. Matrix Operations\n")

matrix1 = np.array([[1, 2],
                    [3, 4]])

matrix2 = np.array([[5, 6],
                    [7, 8]])

print("Matrix 1:")
print(matrix1)
print()

print("Matrix 2:")
print(matrix2)
print()

print("Matrix Multiplication:")
print(matrix1 @ matrix2)
print()

print("Dot Product:")
print(np.dot(matrix1, matrix2))
print()

# ---------------------------------------------------
# 3. Statistical Functions
# ---------------------------------------------------

print("3. Statistical Functions\n")

data = np.array([10, 20, 30, 40, 50])

print("Data:")
print(data)
print()

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard Deviation:", np.std(data))
print("Minimum Value:", np.min(data))
print("Maximum Value:", np.max(data))
print("Sum:", np.sum(data))
print()

# ---------------------------------------------------
# 4. Axis-wise Operations
# ---------------------------------------------------

print("4. Axis-wise Operations\n")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("Matrix:")
print(matrix)
print()

print("Column-wise Sum:")
print(np.sum(matrix, axis=0))
print()

print("Row-wise Sum:")
print(np.sum(matrix, axis=1))
print()

print("Column-wise Mean:")
print(np.mean(matrix, axis=0))
print()

# ---------------------------------------------------
# 5. Sorting Arrays
# ---------------------------------------------------

print("5. Sorting Arrays\n")

unsorted_array = np.array([45, 12, 7, 89, 23])

print("Original Array:")
print(unsorted_array)
print()

sorted_array = np.sort(unsorted_array)

print("Sorted Array:")
print(sorted_array)
print()

# ---------------------------------------------------
# 6. Searching Operations
# ---------------------------------------------------

print("6. Searching Operations\n")

search_array = np.array([5, 10, 15, 20, 25])

print("Search Array:")
print(search_array)
print()

print("Index of maximum value:")
print(np.argmax(search_array))
print()

print("Index of minimum value:")
print(np.argmin(search_array))
print()

print("Indices where value > 12:")
print(np.where(search_array > 12))
print()

# ---------------------------------------------------
# 7. Universal Functions
# ---------------------------------------------------

print("7. Universal Functions\n")

angles = np.array([0, np.pi/2, np.pi])

print("Angles:")
print(angles)
print()

print("Sine Values:")
print(np.sin(angles))
print()

print("Cosine Values:")
print(np.cos(angles))
print()

print("Square Root:")
print(np.sqrt(np.array([1, 4, 9, 16])))
print()

# ---------------------------------------------------
# 8. Random Number Generation
# ---------------------------------------------------

print("8. Random Number Generation\n")

np.random.seed(42)

print("Random Integers:")
random_ints = np.random.randint(1, 100, size=5)
print(random_ints)
print()

print("Random Float Values:")
random_floats = np.random.rand(5)
print(random_floats)
print()

print("Random Normal Distribution:")
normal_distribution = np.random.randn(5)
print(normal_distribution)
print()

# ---------------------------------------------------
# 9. Boolean Operations
# ---------------------------------------------------

print("9. Boolean Operations\n")

bool_array = np.array([10, 15, 20, 25, 30])

print("Original Array:")
print(bool_array)
print()

print("Values greater than 18:")
print(bool_array > 18)
print()

print("Filtered Values:")
print(bool_array[bool_array > 18])
print()

# ---------------------------------------------------
# 10. Cumulative Operations
# ---------------------------------------------------

print("10. Cumulative Operations\n")

cum_array = np.array([1, 2, 3, 4, 5])

print("Original Array:")
print(cum_array)
print()

print("Cumulative Sum:")
print(np.cumsum(cum_array))
print()

print("Cumulative Product:")
print(np.cumprod(cum_array))
print()

print("========== END OF MATHEMATICAL OPERATIONS ==========")
