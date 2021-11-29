import scipy.stats
import numpy as np

# sample locations
def LocationsSampler(size_x, n, type_prior_x, dim_x):

    if type_prior_x == 'uniform':
        x = size_x * scipy.stats.uniform().rvs(n) if dim_x == 1 else scipy.stats.uniform().rvs((n, dim_x))

    if type_prior_x == 'tNormal':
        lower = 0
        upper = size_x
        mu = 0.3
        sigma = 0.3
        # x = scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(n) if dim_x == 1\
        #     else scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs((n, dim_x))
        lower = np.array((0.43, -2.18))
        upper = np.array((0.86, -1.19))
        mu = (upper - lower) / 2
        sigma = 0.3
        x = scipy.stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                                      loc=mu, scale=sigma * np.ones((n, dim_x)))
    if type_prior_x == 'normal':
        x = scipy.stats.norm(3, 0.1).rvs(n) if dim_x == 1 else scipy.stats.norm(3, 0.1).rvs((n, dim_x))

    return x