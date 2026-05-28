# Day 1 - NumPy Fundamentals
# basic_arrays.py
# MLSI Lab Summer Internship 2026
# Goutam Anand

import numpy as np

print("========== NUMPY ARRAY FUNDAMENTALS ==========\n")

# ---------------------------------------------------
# 1. Creating Arrays
# ---------------------------------------------------

print("1. Creating Arrays\n")

zeros_array = np.zeros((2, 3))
print("Zeros Array:")
print(zeros_array)
print()

ones_array = np.ones((3, 2))
print("Ones Array:")
print(ones_array)
print()

arange_array = np.arange(1, 11, 2)
print("Arange Array:")
print(arange_array)
print()

linspace_array = np.linspace(0, 1, 5)
print("Linspace Array:")
print(linspace_array)
print()

normal_array = np.array([[1, 2, 3],
                         [4, 5, 6]])

print("Normal NumPy Array:")
print(normal_array)
print()

# ---------------------------------------------------
# 2. Array Shape and Dimensions
# ---------------------------------------------------

print("2. Array Shape and Dimensions\n")

print("Shape of normal_array:", normal_array.shape)
print("Dimensions of normal_array:", normal_array.ndim)
print("Size of normal_array:", normal_array.size)
print("Data type:", normal_array.dtype)
print()

# ---------------------------------------------------
# 3. Indexing and Slicing
# ---------------------------------------------------

print("3. Indexing and Slicing\n")

sample_array = np.array([10, 20, 30, 40, 50, 60])

print("Original Array:")
print(sample_array)
print()

print("First Element:", sample_array[0])
print("Last Element:", sample_array[-1])
print()

print("Elements from index 1 to 4:")
print(sample_array[1:5])
print()

print("Every second element:")
print(sample_array[::2])
print()

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print("2D Matrix:")
print(matrix)
print()

print("Element at row 2 column 3:", matrix[1, 2])
print()

print("First Row:")
print(matrix[0])
print()

print("Second Column:")
print(matrix[:, 1])
print()

# ---------------------------------------------------
# 4. Reshape Operations
# ---------------------------------------------------

print("4. Reshape Operations\n")

reshape_array = np.arange(1, 13)

print("Original Array:")
print(reshape_array)
print()

reshaped = reshape_array.reshape(3, 4)

print("Reshaped Array (3x4):")
print(reshaped)
print()

# ---------------------------------------------------
# 5. Transpose Operation
# ---------------------------------------------------

print("5. Transpose Operation\n")

print("Original Matrix:")
print(reshaped)
print()

transpose_matrix = reshaped.T

print("Transpose Matrix:")
print(transpose_matrix)
print()

# ---------------------------------------------------
# 6. Concatenation
# ---------------------------------------------------

print("6. Array Concatenation\n")

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

concatenated = np.concatenate((a, b))

print("First Array:")
print(a)

print("Second Array:")
print(b)

print("Concatenated Array:")
print(concatenated)
print()

# ---------------------------------------------------
# 7. Splitting Arrays
# ---------------------------------------------------

print("7. Array Splitting\n")

split_array = np.array([1, 2, 3, 4, 5, 6])

split_result = np.split(split_array, 3)

print("Original Array:")
print(split_array)
print()

print("Split Arrays:")
print(split_result)
print()

# ---------------------------------------------------
# 8. Array Copy vs View
# ---------------------------------------------------

print("8. Array Copy vs View\n")

original = np.array([1, 2, 3, 4])

view_array = original.view()
copy_array = original.copy()

original[0] = 100

print("Original Array:")
print(original)

print("View Array:")
print(view_array)

print("Copy Array:")
print(copy_array)
print()

# ---------------------------------------------------
# 9. Array Iteration
# ---------------------------------------------------

print("9. Array Iteration\n")

iteration_array = np.array([[1, 2],
                            [3, 4]])

print("Matrix:")
print(iteration_array)
print()

print("Iterating through elements:")

for element in np.nditer(iteration_array):
    print(element)

print()

# ---------------------------------------------------
# 10. Meshgrid Example
# ---------------------------------------------------

print("10. Meshgrid Example\n")

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

X, Y = np.meshgrid(x, y)

print("X Grid:")
print(X)
print()

print("Y Grid:")
print(Y)
print()

print("========== END OF BASIC ARRAY PRACTICE ==========")
