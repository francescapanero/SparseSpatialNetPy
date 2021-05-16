from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

sigma = 0.4
c = 1.2

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1
tau = 5

size_x = 5

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'naive'  # can be 'layers' or 'naive'

save_every = 2000

# ----------------------
# SIMULATE DATA
# ----------------------

# # ----------
# t = 100
# # ----------
#
# # ----------
# gamma = 0
# # ----------
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma0_L1000',
#                         save_out=False, save_data=False, init=init)
#
#
# # ----------
# gamma = 1
# # ----------
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma1_L1000',
#                         save_out=False, save_data=False, init=init)
#
# # ----------
# gamma = 2
# # ----------
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=500)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma2_L500',
#                         save_out=False, save_data=False, init=init)
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma2_L1000',
#                         save_out=False, save_data=False, init=init)

# ----------
# gamma = 5
# ----------

# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L1000',
#                         save_out=False, save_data=False, init=init)
#
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=2000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=False, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L2000',
#                         save_out=False, save_data=False, init=init)

# ----------
t = 200
gamma = 2
# ----------

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=2000)
deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[0:len(ind)-1]

iter = 300000
nburn = int(iter * 0.25)
init = {}
init[0] = {}
init[0]['sigma'] = 0.8
init[0]['t'] = 300
init[0]['c'] = 2
init[1] = {}
init[1]['sigma'] = sigma
init[1]['t'] = t
init[1]['c'] = c
init[2] = {}
init[2]['sigma'] = 0.2
init[2]['t'] = 100
init[2]['c'] = 3

out = chain.mcmc_chains([G, G, G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        save_every=save_every, plot=True,  path='L2000_gamma2_t200_hyper_w_x',
                        save_out=False, save_data=False, init=init, a_t=200)


# iter = 500000
# nburn = int(iter * 0.25)
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='L5000_gamma5_t300_hyper_w_nu',
#                         save_out=False, save_data=False, init=init)
#
# iter = 700000
# nburn = int(iter * 0.25)
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=True, x=True, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='L5000_gamma5_t300_hyper_w_u_x',
#                         save_out=False, save_data=False, init=init)

#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=3000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L3000_t300_withw',
#                         save_out=False, save_data=False, init=init)
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=False, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L3000_t300_withw_not',
#                         save_out=False, save_data=False, init=init)
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=5000)
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L5000_t300_withw',
#                         save_out=False, save_data=False, init=init)
#
# iter = 100000
# nburn = int(iter * 0.25)
# init = {}
# init[0] = {}
# init[1] = {}
# init[2] = {}
# out = chain.mcmc_chains([G, G, G], iter, nburn, index,
#                         sigma=True, c=True, t=False, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='hyper_gamma5_L5000_t300_withw_not',
#                         save_out=False, save_data=False, init=init)
#

