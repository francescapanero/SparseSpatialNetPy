import numpy as np
from utils.GGPrnd import GGPrnd
from utils.exptiltBFRY import *


# Sample from GBFRY (Ayed, Lee, Caron 2019)
# w0 can be sampled from finite approximation 'exptiltBFRY' or from infinite generalized gamma process (truncated at T)

def GGPdoublepl(alpha, sigma, tau, **kwargs):

    sigma = float(sigma)
    alpha = float(alpha)
    tau = float(tau)
    T = kwargs['T']
    c = kwargs['c']
    w_type = kwargs['w_type']
    if w_type == 'exptiltBFRY':  # finite approx
        L = kwargs['L']
        w0 = exptiltBFRY(alpha * c ** (sigma * tau - sigma) / (sigma * tau), sigma, c, L)
    if w_type == 'GGP':  # truncated infinite process
        w0 = GGPrnd(alpha * c ** (sigma * tau - sigma) / (sigma * tau), sigma, c, T)
    beta = np.random.beta(sigma * tau, 1, np.size(w0))
    w = w0 / beta
    return w, beta, w0

