from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

sigma = 0.4
c = 1.2
gamma = 2

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1
tau = 5

size_x = 5

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

iter = 400000
nburn = int(iter * 0.25)
save_every = 5000

init = {}
init[0] = {}

# ----------------------
# SIMULATE DATA
# ----------------------

# t = 50
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=200)  # 150
# u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
# out = chain.mcmc_chains([G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='whyper_t50_L200',
#                         save_out=False, save_data=False, init=init)
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=500)  # 240
# u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
# out = chain.mcmc_chains([G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='whyper_t50_L500',
#                         save_out=False, save_data=False, init=init)

t = 100

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=500)  # 400
u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[0:len(ind)-1]
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        save_every=save_every, plot=True,  path='whyper_t100_L500',
                        save_out=False, save_data=False, init=init)

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)  # 650
u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[0:len(ind)-1]
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        save_every=save_every, plot=True,  path='whyper_t100_L1000',
                        save_out=False, save_data=False, init=init)

# t = 200
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)  # 880
# u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
# out = chain.mcmc_chains([G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='whyper_t200_L1000',
#                         save_out=False, save_data=False, init=init)
#
# G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=2000)  # 1500
# u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
# deg = np.array(list(dict(G.degree()).values()))
# ind = np.argsort(deg)
# index = ind[0:len(ind)-1]
# out = chain.mcmc_chains([G], iter, nburn, index,
#                         sigma=True, c=True, t=True, tau=False, w0=True, n=False, u=False, x=False, beta=False,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         save_every=save_every, plot=True,  path='whyper_t200_L2000',
#                         save_out=False, save_data=False, init=init)




