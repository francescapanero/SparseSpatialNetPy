from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain
import scipy
import matplotlib.pyplot as plt

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 1  # exponent distance in the link probability
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

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)
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

# # number of iterations and burn in
iter = 500000
nburn = int(iter * 0.25)

# fix initaliazation values. Now they are all initialized to their true values.

init = {}

# # first graph
init[0] = {}
# init[0]['w_init'] = w
# init[0]['w0_init'] = w
# init[0]['beta_init'] = beta
# init[0]['n_init'] = n
# init[0]['u_init'] = u
# init[0]['sigma_init'] = sigma
# init[0]['c_init'] = c
# init[0]['t_init'] = t
# init[0]['tau_init'] = tau

ind = np.argsort(deg)
a = min(np.where(deg[ind] > 5)[0])
index = ind[a:-1]
init[0]['x_init'] = x.copy()
init[0]['x_init'][index] = x[index] + 2

# # second graph, if present
# init[1] = {}
# init[1]['w_init'] = w_1
# init[1]['w0_init'] = w_1
# init[1]['sigma_init'] = sigma + 0.1
# init[1]['c_init'] = c + 1
# init[1]['t_init'] = t + 40

# remember that even if you have only one chain, you need to give G as a list: [G]
out = chain.mcmc_chains([G], iter, nburn,
                        # which variables to update?
                        sigma=False, c=False, t=False, tau=False,
                        w0=False,
                        n=False,
                        u=False,
                        x=True,
                        beta=False,
                        # set type of update for w: either 'HMC' or 'gibbs'
                        w_inference='HMC', epsilon=0.01, R=5,
                        # MH stepsize (here the sd of the proposals, which are all log normals
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        # save the values only once every save_every iterations
                        save_every=1000,
                        # set plot True to see the traceplots. Indicate the folder in which the plots should go
                        # REMEMBER TO SET UP THE PATH FOLDER IN THE 'IMAGES' FOLDER
                        plot=False, path='test one third x',
                        # save output and data now are set to false cause they'd be very big
                        save_out=False, save_data=False,
                        # set initialization values
                        init=init)


x_est = out[0][12]
for j in index[0:10]:
    plt.figure()
    plt.plot([x_est[i][j] for i in range(int(iter/1000))])
    plt.axhline(y=x[j])
    plt.title('Traceplot location of node with deg %i' % deg[j])
    plt.savefig('images/testspace/deg greater 5 rand init longer/low/trace_%i' % j)
    plt.close()
for j in index[-10:-1]:
    plt.figure()
    plt.plot([x_est[i][j] for i in range(int(iter/1000))])
    plt.axhline(y=x[j])
    plt.title('Traceplot location of node with deg %i' % deg[j])
    plt.savefig('images/testspace/deg greater 5 rand init longer/high/trace_%i' % j)
    plt.close()

i = 0
save_every = 1000
size = G.number_of_nodes()

ind_big1 = index[0:10]
p_ij_est = out[i][11]
p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn+save_every)/save_every),
                                     int((iter+save_every)/save_every))] for j in ind_big1]
emp_ci_95_big = []
num = len(ind_big1)
for j in range(num):
    emp_ci_95_big.append(
        [scipy.stats.mstats.mquantiles(
            [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                  int((nburn + save_every) / save_every))],
            prob=[0.025, 0.975]) for l in range(size)])
if 'distances' in G.graph:
    p_ij = G.graph['distances']
    true_in_ci = [[emp_ci_95_big[j][k][0] <= p_ij[ind_big1[j], k] <= emp_ci_95_big[j][k][1]
                  for k in range(size)] for j in range(num)]
    print('posterior coverage of true p_ij for highest deg nodes (chain %i' % i, ') = ',
          [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
for j in range(num):
    plt.figure()
    for k in range(num):
        plt.plot((k + 1, k + 1), (emp_ci_95_big[j][ind_big1[k]][0], emp_ci_95_big[j][ind_big1[k]][1]),
             color='cornflowerblue', linestyle='-', linewidth=2)
        if 'distances' in G.graph:
            plt.plot(k + 1, p_ij[ind_big1[j], ind_big1[k]], color='navy', marker='o', markersize=5)
    plt.savefig('images/testspace/deg greater 5 rand init longer/low/pij_%i' % j)
    plt.close()

ind_big1 = index[-10:-1]
p_ij_est = out[i][11]
p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn+save_every)/save_every),
                                     int((iter+save_every)/save_every))] for j in ind_big1]
emp_ci_95_big = []
num = len(ind_big1)
for j in range(num):
    emp_ci_95_big.append(
        [scipy.stats.mstats.mquantiles(
            [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                  int((nburn + save_every) / save_every))],
            prob=[0.025, 0.975]) for l in range(size)])
if 'distances' in G.graph:
    p_ij = G.graph['distances']
    true_in_ci = [[emp_ci_95_big[j][k][0] <= p_ij[ind_big1[j], k] <= emp_ci_95_big[j][k][1]
                  for k in range(size)] for j in range(num)]
    print('posterior coverage of true p_ij for highest deg nodes (chain %i' % i, ') = ',
          [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
for j in range(num):
    plt.figure()
    for k in range(num):
        plt.plot((k + 1, k + 1), (emp_ci_95_big[j][ind_big1[k]][0], emp_ci_95_big[j][ind_big1[k]][1]),
             color='cornflowerblue', linestyle='-', linewidth=2)
        if 'distances' in G.graph:
            plt.plot(k + 1, p_ij[ind_big1[j], ind_big1[k]], color='navy', marker='o', markersize=5)
    plt.savefig('images/testspace/deg greater 5 rand init longer/high/pij_%i' % j)
    plt.close()


log_est = out[0][10]
plt.figure()
plt.plot(log_est)
plt.axhline(y=log_post)
plt.savefig('images/testspace/deg greater 5 rand init longer/logpost')
plt.close()