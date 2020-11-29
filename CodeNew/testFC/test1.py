# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:22:02 2020

@author: caron
"""
import numpy as np

def exptiltBFRY(alpha, sigma, tau, L):

    # t = (L*sigma/alpha)^(1/sigma)
    t = np.float((L * sigma / alpha) ** (1 / sigma))

    # simulate sociabilities from exponentially tilted BFRY(sigma, tau, t)
    g = np.random.gamma(1 - sigma, 1, L)
    unif = np.random.rand(L)
    w = np.multiply(g, np.power(((t + tau) ** sigma) * (1 - unif) + (tau ** sigma) * unif, -1 / sigma))
    
    return w

alpha = 10
sigma = 0.3
tau = .8
L = 10000

nsamples = 1000
w=np.empty(nsamples)
for i in range(nsamples):
    w[i] = np.sum(exptiltBFRY(alpha, sigma, tau, L))


mu = alpha / tau**(1-sigma)
v = alpha * (1-sigma)/tau**(2-sigma)
print('true mean: ',mu, 'sample mean: ', np.mean(w))
print('true var: ',v, 'sample var: ', np.var(w))
