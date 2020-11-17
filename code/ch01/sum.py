import numpy as  np

D, N = 8, 7
x = np.array([[1,2,3],[1,2,3]])
print(x)
y = np.sum(x, axis=0, keepdims=True)
print(y)
