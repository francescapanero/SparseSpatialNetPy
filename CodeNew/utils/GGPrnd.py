import numpy as np
import scipy as sp
from scipy.special import gamma


# Sample from truncated generalized gamma process GGP(t; sigma, c) truncated at T (w>T)

def GGPrnd(t, sigma, c, T):

    a = 5
    sigma = float(sigma)
    t = float(t)
    c = float(c)

    if sigma < -1e-8:
        rate = np.exp(np.log(t) - np.log(- sigma) + sigma * np.log(c))
        K = np.random.poisson(rate)
        N = np.random.gamma(-sigma, 1/c, K)
        N = N[N>0]
        T = 0
        return N
    # Use a truncated Pareto on [T,a] and on (a, infty) use truncated exponential (see later)
    if sigma > 0:
        lograte = np.log(t) - np.log(sigma) - c * T - sp.special.gammaln(1-sigma) + np.log(T**(-sigma) - a ** (-sigma))
        Njumps = np.random.poisson(np.exp(lograte))
        # Sample from truncated Pareto
        log_N1 = - 1/sigma * np.log(-(np.random.rand(Njumps) * (a**sigma-T**sigma)-a**sigma)/(a*T)**sigma)
    else:
        lograte = np.log(t) - c*T - np.log(sp.special.gamma(1-sigma)) + np.log(np.log(a) - np.log(T))
        Njumps = np.random.poisson(np.exp(lograte))
        log_N1 = np.random.rand(Njumps) * (np.log(a)-np.log(T)) + np.log(T)
    N1 = np.exp(log_N1)
    ind1 = np.log(np.random.rand(Njumps)) < c*(T - N1)
    N1 = N1[ind1]

    # Use a truncated exponential on (a,+infty) or (T, infty)
    lograte = np.log(t) - c*a - (1+sigma)*np.log(a) - np.log(c) - sp.special.gammaln(1-sigma)
    Njumps = np.random.poisson(np.exp(lograte))
    log_N2 = - np.log(-np.random.rand(Njumps) * np.exp(-c*a) + np.exp(-c*a)) / c
    # log_N2 = np.log(a + np.random.exponential(1/tau, Njumps)) # Sample from truncated exponential
    ind2 = np.log(np.random.rand(Njumps)) < -(1+sigma)*(log_N2 - np.log(a))
    N2 = np.exp(log_N2[ind2])
    N = np.concatenate((N1, N2))
    return N

