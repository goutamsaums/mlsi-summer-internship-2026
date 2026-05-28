# Day 1 - NumPy Fundamentals
# broadcasting_practice.py
# MLSI Lab Summer Internship 2026
# Goutam Anand

import numpy as np

print("========== NUMPY BROADCASTING PRACTICE ==========\n")

# ---------------------------------------------------
# 1. Add Scalar to Array
# ---------------------------------------------------

print("1. Add Scalar to Array\n")

array1 = np.array([1, 2, 3, 4, 5])

print("Original Array:")
print(array1)
print()

scalar_added = array1 + 10

print("After Adding Scalar 10:")
print(scalar_added)
print()

# ---------------------------------------------------
# 2. Multiply Array by Scalar
# ---------------------------------------------------

print("2. Multiply Array by Scalar\n")

multiplied = array1 * 5

print("After Multiplication by 5:")
print(multiplied)
print()

# ---------------------------------------------------
# 3. Broadcasting Between Arrays
# ---------------------------------------------------

print("3. Broadcasting Between Arrays\n")

a = np.array([[1],
              [2],
              [3]])

b = np.array([10, 20, 30])

print("Array A:")
print(a)
print()

print("Array B:")
print(b)
print()

broadcast_sum = a + b

print("Broadcasted Addition Result:")
print(broadcast_sum)
print()

# ---------------------------------------------------
# 4. Broadcasting with Different Shapes
# ---------------------------------------------------

print("4. Operations Between Different Shapes\n")

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

vector = np.array([10, 20, 30])

print("Matrix:")
print(matrix)
print()

print("Vector:")
print(vector)
print()

result = matrix + vector

print("Broadcasted Result:")
print(result)
print()

# ---------------------------------------------------
# 5. Row-wise Broadcasting
# ---------------------------------------------------

print("5. Row-wise Broadcasting\n")

row_values = np.array([[100],
                       [200]])

print("Row Values:")
print(row_values)
print()

row_result = matrix + row_values

print("Row-wise Broadcast Result:")
print(row_result)
print()

# ---------------------------------------------------
# 6. Broadcasting for Normalization
# ---------------------------------------------------

print("6. Array Normalization\n")

data = np.array([[10, 20, 30],
                 [40, 50, 60],
                 [70, 80, 90]])

print("Original Data:")
print(data)
print()

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

print("Column Mean:")
print(mean)
print()

print("Column Standard Deviation:")
print(std)
print()

normalized = (data - mean) / std

print("Normalized Data:")
print(normalized)
print()

# ---------------------------------------------------
# 7. Broadcasting with 3D Arrays
# ---------------------------------------------------

print("7. Broadcasting with 3D Arrays\n")

array_3d = np.ones((2, 3, 4))

print("Shape of 3D Array:")
print(array_3d.shape)
print()

broadcast_vector = np.array([1, 2, 3, 4])

print("Broadcast Vector:")
print(broadcast_vector)
print()

broadcast_3d = array_3d + broadcast_vector

print("Broadcasted 3D Result:")
print(broadcast_3d)
print()

# ---------------------------------------------------
# 8. Meshgrid Creation
# ---------------------------------------------------

print("8. Meshgrid Creation\n")

x = np.linspace(-5, 5, 5)
y = np.linspace(-5, 5, 5)

X, Y = np.meshgrid(x, y)

print("X Grid:")
print(X)
print()

print("Y Grid:")
print(Y)
print()

# ---------------------------------------------------
# 9. Distance Calculation using Broadcasting
# ---------------------------------------------------

print("9. Distance Matrix using Broadcasting\n")

points = np.array([[0, 0],
                   [1, 1],
                   [2, 2]])

print("Points:")
print(points)
print()

distance_matrix = np.sqrt(
    np.sum((points[:, np.newaxis] - points) ** 2, axis=2)
)

print("Distance Matrix:")
print(distance_matrix)
print()

# ---------------------------------------------------
# 10. Broadcasting Rules Demonstration
# ---------------------------------------------------

print("10. Broadcasting Rules Example\n")

arr1 = np.array([[1, 2, 3]])
arr2 = np.array([[10],
                 [20],
                 [30]])

print("Array 1 Shape:", arr1.shape)
print("Array 2 Shape:", arr2.shape)
print()

broadcast_example = arr1 + arr2

print("Broadcast Result:")
print(broadcast_example)
print()

print("Result Shape:", broadcast_example.shape)
print()

print("========== END OF BROADCASTING PRACTICE ==========")
