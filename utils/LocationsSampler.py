import numpy as np
import scipy


# sample locations
def LocationsSampler(size_x, n):

    # uniform prior
    x = size_x * np.random.rand(n)
    # normal prior
    # x = scipy.stats.norm.rvs(3, 0.1, n)

    # # to sample in R^2
    # x1 = size_x * np.random.rand(n)
    # x2 = size_x * np.random.rand(n)
    # x = [[x1[i], x2[i]] for i in range(n)]
    return x