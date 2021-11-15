from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain
import scipy.stats
import matplotlib.pyplot as plt
import os

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

sigma = 0.4
c = 1.2

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1
tau = 5

size_x = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'naive'  # can be 'layers' or 'naive'
type_prop_x = 'tNormal'  # or 'tNormal'
type_prior_x = 'tNormal'
dim_x = 2

# ----------------------
# SIMULATE DATA
# ----------------------

# ----------
t = 200
gamma = .2
# ----------

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x, a_t, b_t,
                 T=T, K=K, L=1000)
deg = np.array(list(dict(G.degree()).values()))
x = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
w0 = np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])
n = G.graph['counts']

ind = np.argsort(deg)
index = ind[0:len(ind)-1]
# index = ind[-sum(deg>1):-1]
# index = ind[-10:-1]
p_ij = G.graph['distances']

init = {}
init[0] = {}
init[0]['sigma'] = .6
init[0]['t'] = 100
init[0]['c'] = c
init[0]['x'] = x.copy()
init[0]['counts'] = n.copy()
init[0]['w0'] = w0
# init[0]['x'][index] = size_x * np.random.rand(len(index))
init[1] = {}
#init[1]['sigma'] = 0.8
#init[1]['t'] = 300
#init[1]['c'] = 2
init[1]['x'] = x.copy()
#init[1]['x'][index] = np.random.uniform(0, 1, len(index))
#init[1]['x'][index] = np.random.uniform(0, 1, (len(index), dim_x))
# # init[2] = {}
# # init[2]['sigma'] = 0.2
# # init[2]['t'] = 100
# # init[2]['c'] = 1
# # init[2]['x'] = x.copy()
# # init[2]['x'][index] = size_x * np.random.rand(len(index))

iter = 300000
save_every = 1000
nburn = int(iter * 0.25)
path = 'biv_simulated_updateall'
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=False, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        save_every=save_every, plot=True, path=path,
                        save_out=False, save_data=False, init=init, a_t=200, type_prop_x=type_prop_x)

deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
set_nodes = biggest_deg[-7:]
for m in range(len(set_nodes)):
    plt.figure()
    plt.plot([out[0][12][j][set_nodes[m]][0] for j in range(len(out[0][12]))])
    plt.axhline(y=x[set_nodes[m]][0], color='red')
    plt.title('first coordinate location %i' % set_nodes[m])
    plt.savefig(os.path.join('images', path, 'x0_%i' % set_nodes[m]))
    plt.close()
    plt.figure()
    plt.plot([out[0][12][j][set_nodes[m]][1] for j in range(len(out[0][12]))])
    plt.axhline(y=x[set_nodes[m]][1], color='red')
    plt.title('second coordinate location %i' % set_nodes[m])
    plt.savefig(os.path.join('images', path, 'x1_%i' % set_nodes[m]))
    plt.close()
