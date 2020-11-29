import numpy as np
from scipy.stats import poisson


def TruncatedPoisson(lam): # slower version
    k = 1
    t = lam*np.exp(np.negative(lam))/(1-np.exp(np.negative(lam)))
    s = t
    u = np.random.rand(1)[0]
    while s < u:
        k = k+1
        t = t*lam/k
        s = s+t
    return k


def tpoissrnd(lam):
    # lam MUST be an array
    if not np.isscalar(lam):
        x = np.ones(len(lam))
        ind = lam > 1e-5 # below this value x=1 whp
        #ind = ind[:, 0]
        if np.any(ind):
            n_ = sum(ind)
            lam_ = lam[ind]
            x[ind] = poisson.ppf(np.exp(-lam_) + np.multiply(np.random.rand(n_), 1 - np.exp(-lam_)), lam_) #[:, 0]
    else:
        x = 1
        ind = lam > 1e-5
        if np.any(ind):
            n_ = ind
            # lam_ = lam[ind]
            x = poisson.ppf(np.exp(-lam) + np.random.rand() * (1 - np.exp(-lam)), lam)
            #if x == 0:
            #    x = 1

    return x
