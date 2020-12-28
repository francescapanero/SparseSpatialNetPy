import utils.MCMCNew as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew as aux
import utils.UpdatesNew as up
import numpy as np
# import pandas as pd
# import pymc3 as pm3
import matplotlib.pyplot as plt

# Set parameters for simulating data
t = 100
sigma = 0.4
c = 2
tau = 5  # if singlepl then set it to 0, otherwise > 1
gamma = 0  # exponent distance in the link probability
size_x = 1

K = 100  # number of layers, for layers sampler
T = 0.00001  # threshold simulations weights for GGP and doublepl (with w0 from GGP)
L = 5000  # tot number of nodes in exptiltBFRY

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

compute_distance = True  # you need distances if you are performing inference on w, n, u
reduce = False  # reduce graph G, locations x and weights w to active nodes. Usually not necessary.
check = False  # to check the logposterior of the parameters sigma, tau, alpha given w and u in a grid around the
# original parameters

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
log_post = aux.log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t)

if reduce is True:
    G, isol = aux.SimpleGraph(G)
    x_red = np.delete(x, isol)
    w_red = np.delete(w, isol)

if check is True:
    temp = aux.check_log_likel_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t)
    print('true log likel params = ', aux.log_likel_params(prior, sigma, c, t, tau, w0, beta, u))
    print('log likel in max = ', temp[0])
    if prior == 'singlepl':
        print('params for max (sigma, c, t) = ', temp[1])
    if prior == 'doublepl':
        print('params for max (sigma, c, t, tau) = ', temp[1])

# ---------------------
# posterior inference
# ---------------------

# ----------------
# parameters only
# ----------------

iter = 700000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01

# inference on hyperparams (all together), with w0, beta, n, u fixed
output1 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
                    hyperparams=True,
                    sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
                    sigma_init=sigma+0.2, c_init=c+1, tau_init=tau, t_init=t+50,
                    # sigma_init=sigma, c_init=c, tau_init=tau, t_init=t,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)
output2 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
                    hyperparams=True,
                    sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
                    sigma_init=0.2, c_init=4, tau_init=tau, t_init=t+20,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)
output3 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
                    hyperparams=True,
                    sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
                    sigma_init=0.5, c_init=1.5, tau_init=tau, t_init=t-30,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)

plt.figure()
plt.plot(output1[3], color='blue')
plt.plot(output2[3], color='cornflowerblue')
plt.plot(output3[3], color='navy')
plt.axhline(y=sigma, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/long_exp_no_change_var/sigma1')
plt.figure()
plt.plot(output1[4], color='blue')
plt.plot(output2[4], color='cornflowerblue')
plt.plot(output3[4], color='navy')
plt.axhline(y=c, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/long_exp_no_change_var/c1')
plt.figure()
plt.plot(output1[5], color='blue')
plt.plot(output2[5], color='cornflowerblue')
plt.plot(output3[5], color='navy')
plt.axhline(y=t, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/long_exp_no_change_var/t1')
plt.figure()
plt.plot(output1[9][iter-10000], color='blue')
plt.plot(output2[9][iter-10000], color='cornflowerblue')
plt.plot(output3[9][iter-10000], color='navy')
plt.axhline(y=log_post, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('log posterior')
plt.savefig('images/long_exp_no_change_var/logpost1')

# ----------------
# w only
# ----------------

#iter = 10000
#nburn = int(iter * 0.25)
#epsilon = 0.01
#R = 5
#w_inference = 'HMC'

#output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
#                   w_inference=w_inference, epsilon=epsilon, R=R,
#                   a_t=a_t, b_t=b_t,
#                   plot=True,
#                   w0=True,
#                   w0_init=w0,
#                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)

# df = pd.DataFrame(output[0])
# lags = np.arange(1, 100)
# fig, ax = plt.subplots()
# ax.plot(lags, [pm3.autocorr(df[1], l) for l in lags])
# _ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
# plt.title('Autocorrelation Plot')
# plt.show()
