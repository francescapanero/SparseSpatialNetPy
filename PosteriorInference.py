from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain
import utils.PlotMCMC as plt_mcmc

# Set parameters for simulating data
t = 200  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 2  # exponent distance in the link probability
size_x = 5  # space threshold: [0, size_x]

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

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1500)
# G1 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=2000)

# recover true values of variables
w = np.array([G.nodes[i]['w'] for i in range(G.number_of_nodes())])
w0 = np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])
beta = np.array([G.nodes[i]['beta'] for i in range(G.number_of_nodes())])
x = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
deg = np.array(list(dict(G.degree()).values()))
n = G.graph['counts']
p_ij = G.graph['distances']
ind = G.graph['ind']
selfedge = G.graph['selfedge']
log_post = G.graph['log_post']


# ----------------------
# POSTERIOR INFERENCE
# ----------------------

# # number of iterations and burn in and save_every (save the values of the chain only once every save_every iterations)
iter = 1000000
nburn = int(iter * 0.25)
save_every = 5000

# fix initaliazation values. Now they are all initialized to their true values.

init = {}

# # first graph
init[0] = {}
# init[0]['w_init'] = w
# init[0]['w0_init'] = w
# init[0]['beta_init'] = beta
# init[0]['n_init'] = n
# init[0]['u_init'] = u
# init[0]['sigma_init'] = sigma + 0.2
# init[0]['c_init'] = c + 1
# init[0]['t_init'] = t + 20
# init[0]['tau_init'] = tau

ind = np.argsort(deg)
# a = min(np.where(deg[ind] > 0)[0])
index = ind[0:len(ind)-1]
init[0]['x_init'] = x.copy()
init[0]['x_init'][index] = x[index] + 1

# # second graph, if present
# init[1] = {}
# init[1]['w_init'] = w_1
# init[1]['w0_init'] = w_1
# init[1]['sigma_init'] = sigma + 0.1
# init[1]['c_init'] = c + 1
# init[1]['t_init'] = t + 40


# remember that even if you have only one chain, you need to give G as a list: [G]
out = chain.mcmc_chains([G], iter, nburn, index,
                        # which variables to update?
                        sigma=False, c=False, t=False, tau=False,
                        w0=True,
                        n=False,
                        u=False,
                        x=True,
                        beta=False,
                        # set type of update for w: either 'HMC' or 'gibbs'
                        w_inference='HMC', epsilon=0.01, R=5,
                        # MH stepsize (here the sd of the proposals, which are all log normals
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        # save the values only once every save_every iterations
                        save_every=save_every,
                        # set plot True to see the traceplots. Indicate the folder in which the plots should go
                        # REMEMBER TO SET UP THE PATH FOLDER IN THE 'IMAGES' FOLDER
                        plot=True,  path='test_xw_1500nodes',
                        # save output and data now are set to false cause they'd be very big
                        save_out=False, save_data=False,
                        # set initialization values
                        init=init)
