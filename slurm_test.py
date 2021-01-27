#! /usr/bin/end python

import utils.MCMCNew_fast as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew_fast as aux
import utils.UpdatesNew_fast as up
import numpy as np
# import pymc3 as pm3
import matplotlib.pyplot as plt
import scipy
import pickle

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 2  # exponent distance in the link probability
L_x = 1  # space threshold: [0, L_x]

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

compute_distance = True  # you need distances if you are performing inference on w, n, u
reduce = False  # reduce graph G, locations x and weights w to active nodes. Usually not necessary.
check = False  # to check the log likelihood of the parameters sigma, c, t, tau given w and u in a grid around the
# original parameters

iter = 500000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'

save_every = 2500  # save output every save_every iterations. Must be multiple of 25

# ----------------------------------
# L = 1000
# ----------------------------------

L1 = 1000

w, w0, beta, x, G, L, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
                                            T=T, K=K, L=L1)

#with open('w_slurm', 'wb') as f:
#    w_slurm = pickle.dump(w, f)

