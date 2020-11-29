import numpy as np


# sample locations
def LocationsSampler(size_x, n):

    x = size_x * np.random.rand(n)
    # # to sample in R^2
    # x1 = size_x * np.random.rand(n)
    # x2 = size_x * np.random.rand(n)
    # x = [[x1[i], x2[i]] for i in range(n)]
    return x