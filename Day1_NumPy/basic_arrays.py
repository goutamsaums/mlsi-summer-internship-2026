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

# indexing and slicing

print("\nFirst Element:")
print(a[0])

print("\nSlicing:")
print(a[1:3])

# basic operations

print("\nAddition:")
print(a + 2)

print("\nMultiplication:")
print(a * 2)

# statistical operations

print("\nMean:")
print(np.mean(a))

print("\nStandard Deviation:")
print(np.std(a))

# transpose operation

x = np.array([[1,2],[3,4]])

print("\nTranspose:")
print(x.T)

# broadcasting

v = np.array([1,0])

print("\nBroadcasting:")
print(x + v)

# random array

print("\nRandom Array:")
print(np.random.rand(2,2))
