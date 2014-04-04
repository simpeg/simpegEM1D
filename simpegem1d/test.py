from numba import jit
import numpy as np
import time
# @jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result
n = 1e4
a = np.arange(n**2).reshape(n,n)

start = time.clock()
print(sum2d(a))
elapsed = (time.clock() - start)
print elapsed