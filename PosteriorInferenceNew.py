import utils.MCMCNew as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew as aux
import utils.UpdatesNew as up
import numpy as np
import pandas as pd
import pymc3 as pm3
import matplotlib.pyplot as plt

# Set parameters for simulating data
t = 100
sigma = 0.3
c = 2
tau = 5  # if singlepl then set it to 0, otherwise > 1
gamma = 2  # exponent distance in the link probability
size_x = 1


K = 100  # number of layers, for layers sampler
T = 0.00001  # threshold simulations weights for GGP and doublepl (with w0 from GGP)
L = 2000  # tot number of nodes in exptiltBFRY

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_alpha = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

compute_distance = True  # you need distances if you are performing inference on w, n, u
reduce = False  # reduce graph G, locations x and weights w to active nodes. Usually not necessary.
check = True  # to check the loglikelihood of the parameters sigma, tau, alpha given w and u

# ----------------------
# SIMULATE DATA
# ----------------------

w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                            T=T, K=K, L=L)

# compute distances
if compute_distance is True and gamma != 0:
    p_ij = aux.space_distance(x, gamma)
    n = up.update_n(w, G, size, p_ij)
if compute_distance is True and gamma == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij)

# compute auxiliary variables and quantities
z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u = tp.tpoissrnd(z * w0)

if reduce is True:
    G, isol = aux.SimpleGraph(G)
    x_red = np.delete(x, isol)
    w_red = np.delete(w, isol)

if check is True:
    check = aux.check_sample_loglik(prior, sigma, c, t, tau, w0, beta, u)

# ---------------------
# posterior inference
# ---------------------

iter = 10000
nburn = int(iter * 0.25)
epsilon = 0.01
R = 3
w_inference = 'gibbs'
a_t = 200
b_t = 1

# # inference on hyperparams (all together), with w0, beta, n, u fixed
# output = mcmc.MCMC(prior, G, p_ij, gamma, size, iter, nburn,
#                    w_inference=w_inference, epsilon=epsilon, R=R,
#                    hyperparams=True,
#                    sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, a_t=a_t, b_t=b_t,
#                    plot=True,
#                    c_init=1.1,
#                    #sigma_init=sigma, tau_init=tau, t_init=t,
#                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)

# inference on w0, all the rest fixed
output = mcmc.MCMC(prior, G, p_ij, gamma, size, iter, nburn,
                   w_inference=w_inference, epsilon=epsilon, R=R,
                   a_t=a_t, b_t=b_t,
                   plot=True,
                   w0=True,
                   #w0_init=w0,
                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)

# df = pd.DataFrame(output[0])
# lags = np.arange(1, 100)
# fig, ax = plt.subplots()
# ax.plot(lags, [pm3.autocorr(df[1], l) for l in lags])
# _ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
# plt.title('Autocorrelation Plot')
# plt.show()
#
