import numpy as np

# creating arrays

a = np.array([1, 2, 3, 4])
b = np.zeros((2,2))
c = np.ones((2,2))
d = np.arange(0, 10, 2)
e = np.linspace(0, 1, 5)

print("Array:")
print(a)

print("\nZeros Array:")
print(b)

print("\nOnes Array:")
print(c)

print("\nArange:")
print(d)

print("\nLinspace:")
print(e)

# basic operations

print("\nAddition:")
print(a + 2)

print("\nMultiplication:")
print(a * 2)

# transpose

x = np.array([[1,2],[3,4]])

print("\nTranspose:")
print(x.T)

# broadcasting

v = np.array([1,0])

print("\nBroadcasting:")
print(x + v)
