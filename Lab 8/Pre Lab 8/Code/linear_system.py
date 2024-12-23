import numpy as np

A = np.array([[1, 1, 1, 1],
                [1, -1, -1, 1],
                [1, 1, -1, -1],
                [1, -1, 1, -1]])
b = np.array([1, 1, -1, -1])

W = np.linalg.solve(A, b)
print(W)