from utils.GraphSampler import *
import numpy as np
import mcmc_chains as chain
import scipy.stats
import matplotlib.pyplot as plt
import os

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

sigma = 0.2
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
gamma = 1
# ----------

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x, a_t, b_t,
                 T=T, K=K, L=1000)
deg = np.array(list(dict(G.degree()).values()))
x = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
w0 = np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])
n = G.graph['counts']

l = G.number_of_nodes()
dist = np.zeros((l, l))
dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x, 'euclidean'))

ind = np.argsort(deg)
index = ind[0:len(ind)-1]
# index = ind[-sum(deg>1):-1]
# index = ind[-10:-1]
p_ij = G.graph['distances']

init = {}
init[0] = {}
# init[0]['sigma'] = .6
# init[0]['t'] = 100
# init[0]['c'] = c
init[0]['x'] = x.copy()
init[0]['x'][index] = np.random.rand(len(index), dim_x)
# init[0]['counts'] = n.copy()
# init[0]['w0'] = w0
# init[0]['x'][index] = size_x * np.random.rand(len(index))
# init[1] = {}
# init[1]['sigma'] = 0.8
# init[1]['t'] = 300
# init[1]['c'] = 2
# init[1]['x'] = x.copy()
# init[1]['x'][index] = np.random.uniform(0, 1, len(index))
# init[1]['x'][index] = np.random.uniform(0, 1, (len(index), dim_x))
# # init[2] = {}
# # init[2]['sigma'] = 0.2
# # init[2]['t'] = 100
# # init[2]['c'] = 1
# # init[2]['x'] = x.copy()
# # init[2]['x'][index] = size_x * np.random.rand(len(index))

iter = 400000
save_every = 1000
nburn = int(iter * 0.25)
path = 'X_simul_fixwhyper'
for i in G.nodes():
    G.nodes[i]['w0'] = 1
    G.nodes[i]['w'] = 1
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=False, c=False, t=False, tau=False, w0=False, n=True, u=False, x=True, beta=False,
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

x_mean0 = np.zeros(G.number_of_nodes())
x_mean1 = np.zeros(G.number_of_nodes())
for m in range(G.number_of_nodes()):
    if dim_x == 1:
        x_mean0[m] = np.mean([out[0][12][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 2:
        x_mean0[m] = np.mean([out[0][12][j][m][0] for j in range(int(nburn/save_every), int(iter/save_every))])
        x_mean1[m] = np.mean([out[0][12][j][m][1] for j in range(int(nburn/save_every), int(iter/save_every))])

plt.figure()
plt.scatter(deg, [x[i][0] for i in G.nodes()])
plt.xlabel('Degree')
plt.ylabel('x0')
plt.title('Degree vs x0')
plt.savefig(os.path.join('images', path, 'deg_vs_x0_powerlaw'))
plt.close()

plt.figure()
plt.scatter(deg, x_mean0)
plt.xlabel('Degree')
plt.ylabel('Posterior mean x0')
plt.title('Degree vs posterior x0')
plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter(deg, [x[i][1] for i in G.nodes()])
plt.xlabel('Degree')
plt.ylabel('x1')
plt.title('Degree vs x1')
plt.savefig(os.path.join('images', path, 'deg_vs_x1_powerlaw'))
plt.close()

plt.figure()
plt.scatter(deg, x_mean1)
plt.xlabel('Degree')
plt.ylabel('Posterior mean x1')
plt.title('Degree vs posterior x1')
plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x1'))
plt.close()


l = G.number_of_nodes()
dist_est = np.zeros((l, l, len(out[0][11])))
i = 0
for m in range(l):
    for n in range(m + 1, l):
        for j in range(len(out[i][12])):
            if dim_x == 1:
                dist_est[m, n, j] = np.abs(out[i][12][j][m] - out[i][12][j][n])
                dist_est[n, m, j] = dist_est[m, n, j]
            if dim_x == 2:
                dist_est[m, n, j] = np.sqrt((out[i][12][j][m][0]-out[i][12][j][n][0])**2 +
                                            (out[i][12][j][m][1]-out[i][12][j][n][1])**2)
                dist_est[n, m, j] = dist_est[m, n, j]
plt.figure()
for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.scatter(np.mean([dist_est[set_nodes[m]][set_nodes[n]][j] for j in range(dist_est.shape[2])]),
                 dist[set_nodes[m]][set_nodes[n]], color='blue')
plt.savefig(os.path.join('images', path, 'posterior_distance_vs_true'))
plt.close()
for m in range(len(set_nodes)):
    plt.figure()
    plt.plot([out[i][12][j][set_nodes[m]] for j in range(len(out[i][12]))])
    # plt.axhline(x[set_nodes[m]][0], color='red')
    plt.savefig(os.path.join('images', path, 'x0_%i' % set_nodes[m]))
    plt.close()
if dim_x == 2:
    for m in range(len(set_nodes)):
        plt.figure()
        plt.plot([out[i][12][j][set_nodes[m]][0] for j in range(len(out[i][12]))])
        plt.axhline(x[set_nodes[m]][1], color='red')
        plt.savefig(os.path.join('images', path, 'x0_%i' % set_nodes[m]))
        plt.close()
    for m in range(len(set_nodes)):
        plt.figure()
        plt.plot([out[i][12][j][set_nodes[m]][1] for j in range(len(out[i][12]))])
        plt.axhline(x[set_nodes[m]][1], color='red')
        plt.savefig(os.path.join('images', path, 'x1_%i' % set_nodes[m]))
        plt.close()

for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.figure()
        plt.plot(dist_est[set_nodes[m], set_nodes[n], :])
        plt.axhline(dist[set_nodes[m], set_nodes[n]], color='red')
        plt.title('distance between nodes %i, %i' % (set_nodes[m], set_nodes[n]))
        plt.savefig(os.path.join('images', path, 'distance_%s_%s' % (set_nodes[m], set_nodes[n])))
        plt.close()

# 3. Plots of x vs longitude and latitude
plt.figure()
plt.scatter([x[i][0] for i in G.nodes()], x_mean0)
plt.legend()
plt.xlabel('True x0')
plt.ylabel('Posterior mean x0')
plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter([x[i][1] for i in G.nodes()], x_mean1)
plt.legend()
plt.xlabel('True x1')
plt.ylabel('Posterior mean x1')
plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x1'))
plt.close()


