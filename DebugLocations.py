from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain
import utils.PlotMCMC as plt_mcmc

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 1  # exponent distance in the link probability
size_x = 10  # space threshold: [0, size_x]

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

# ----------------------
# SIMULATE DATA
# ----------------------

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)

# export true values (will be useful to specify initialisations and select nodes to update based on their degree)
x = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
deg = np.array(list(dict(G.degree()).values()))


# ----------------------
# POSTERIOR INFERENCE FOR LOCATIONS (remember, all the other variables and hyperparams are fixed to their true values)
# ----------------------

# number of iterations, burn in and save_every (save the values of the chain only once every save_every iterations)
iter = 500000
nburn = int(iter * 0.25)
save_every = 1000

# sd of MH proposal for locations (Normal(x, sigma_x^2))
sigma_x = 0.01

# # The experiments I'm running are considering the updates of
# # only a subset of the nodes identified in 'index' (starting with the highest degree one, then the 10 highest...)
ind = np.argsort(deg)
# # - update the n highest deg nodes
# n = 10
# index = ind[len(ind)-n: len(ind)]
# # - update the nodes with deg > 5
a = min(np.where(deg[ind] > 5)[0])
index = ind[a:len(ind)]

# if you want, specify an initialization value you'd like for locations x[index].
# Otherwise, for a random init (from uniform r.v.) simply don't specify init in the function chain.mcmc_debug_x
# (remember that if index != all the nodes, you need to specify that the non updated are fixed to their true value
init = {}
init['x_init'] = x.copy()
# init['x_init'][index] = x[index] + 5

# run the inference
out = chain.mcmc_debug_x(G, iter, nburn, save_every, sigma_x, index, init=init)

# debugging plots. Specify the name of the folder (you'll find it in 'images') in which they're saved.
# outputs:
# - traceplot of log posterior
# - traceplots of x[index]
# - p_ij 95% posterior c.i. wrt true valued for some of x[index] (max 20 of them)
path = 'testspace_deggreater5_trueinit'
plt_mcmc.plot_space_debug(out, G, iter, nburn, save_every, index, path)