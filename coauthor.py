import numpy as np
import matplotlib.pyplot as plt
import mcmc_chains as chain
import pandas as pd
import networkx as nx
import re
import json
import math
import scipy
import os
from utils.GraphSampler import *
from mpl_toolkits.mplot3d import Axes3D


def plt_deg_distr(deg, color='blue', label='', plot=True):
    deg = deg[deg > 0]
    num_nodes = len(deg)
    bins = np.array([2**i for i in range(11)])
    sizebins = (bins[1:] - bins[:-1])
    counts = np.histogram(deg, bins=bins)[0]
    freq = counts / num_nodes / sizebins
    if plot is True:
        plt.plot(bins[:-1], freq, 'o', color=color, label=label)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('deg')
        plt.ylabel('frequency')
    return freq


# # -----------------------------
# # COLLABORATIONS https://github.com/dsaffo/GeoSocialVis/tree/master/data
# # -----------------------------

with open('data/GeoSocialVis-master/authorGraph.json') as f:
    json_data = json.loads(f.read())

G = nx.Graph()
G.add_nodes_from(elem['id'] for elem in json_data['nodes'])
G.add_edges_from((elem['source'], elem['target']) for elem in json_data['links'])

# import spatial location of universities
with open('data/GeoSocialVis-master/affiliationList.js') as dataFile:
    data = dataFile.read()
    obj = data[data.find('['): data.rfind(']')+1]
    jsonObj = json.loads(obj)
c = [elem['Name'] for elem in jsonObj]
c1 = [elem['Position'] for elem in jsonObj]
long = {}
lat = {}
for i in range(len(c1)):
    if len(c1[i]) != 0:
        lat[c[i]] = float(c1[i].split(sep=',')[0])
        long[c[i]] = float(c1[i].split(sep=',')[1])

# create dictionary of attributes for each node
a = [elem['id'] for elem in json_data['nodes']]
b = [elem['affiliation'] for elem in json_data['nodes']]
d = {}
count = 0
for i in range(len(a)):
    if b[i] in long.keys():
        d[a[i]] = {'affiliation': b[i], 'long': long[b[i]], 'lat': lat[b[i]]}
    else:
        count += 1
        G.remove_node(a[i])
nx.set_node_attributes(G, d)

for i in [node for node, degree in G.degree() if degree == 0]:
    G.remove_node(i)

deg = deg = np.array(list(dict(G.degree()).values()))
plt_deg_distr(deg)

G = nx.relabel.convert_node_labels_to_integers(G)
l = G.number_of_nodes()
dist = np.zeros((l, l))
p_ij = np.zeros((l, l))
lat = np.zeros(l)
long = np.zeros(l)
for i in range(l):
    lat[i] = G.nodes[i]['lat'] * math.pi / 180
    long[i] = G.nodes[i]['long'] * math.pi / 180
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j]))+np.cos(lat[i])*np.cos(lat[j])
#                                                        * np.cos(long[j] - long[i]))
# dist = dist[dist != 0]
# plt.figure()
# plt.hist(dist, bins=50)

biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
set_nodes = biggest_deg[-7:]

attributes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
attributes = attributes.reset_index()

# # histogram active distances
# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             if lat[i] != lat[j] and long[i] != long[j]:
#                 dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                                                            * np.cos(long[j] - long[i]))
# dist = dist / np.max(dist)
# dist = dist + np.transpose(dist)
# plt.figure()
# plt.hist(dist, bins=20)

# size_x = 1
# prior = 'singlepl'
# c = 0.65
# sigma = 0.25
# t = 65
# tau = 5
# K = 100  # number of layers, for layers sampler
# T = 0.000001
# a_t = 200
# b_t = 1
# approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
# sampler = 'naive'  # can be 'layers' or 'naive'
# type_prop_x = 'tNormal'  # or 'tNormal'
# type_prior_x = 'tNormal'
# dim_x = 1
# gamma = 20
# Gsim = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x,
#                     a_t, b_t, T=T, K=K, L=G.number_of_nodes()+150)
# x = np.array([Gsim.nodes[i]['x'] for i in range(Gsim.number_of_nodes())])
# dist_sim = np.zeros((Gsim.number_of_nodes(), Gsim.number_of_nodes()))
# for i in range(Gsim.number_of_nodes()):
#     for j in [n for n in Gsim.neighbors(i)]:
#         if j > i:
#             if dim_x == 1:
#                 dist_sim[i, j] = np.abs(x[i] - x[j])
#             if dim_x == 2:
#                 dist_sim[i, j] = np.sqrt((x[i][0]-x[j][0])**2 +
#                                          (x[i][1]-x[j][1])**2)
# plt.figure()
# plt.hist(dist_sim[dist_sim != 0], bins=20, range=(0, 1))

# distances full matrix
l = G.number_of_nodes()
dist = np.zeros((l, l))
for i in range(l):
    for j in range(i+1, l):
        if lat[i] != lat[j] and long[i] != long[j]:
            dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                       * np.cos(long[j] - long[i]))
dist = dist / np.max(dist)
dist = dist + np.transpose(dist)


# -------------
# SET UP MCMC
# -------------

L0 = G.number_of_nodes()
nodes_added = 50
L = G.number_of_nodes() + nodes_added
G.add_nodes_from(range(L0, L))

deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[1:len(ind)]

gamma = 0
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1

init = {}
init[0] = {}
init[0]['sigma'] = 0.2  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = .5
init[0]['t'] = 50  # np.sqrt(G.number_of_edges())
size_x = 1
init[0]['size_x'] = size_x
dim_x = 1
lower = 0
upper = size_x
mu = 0.3
sigma = 0.1
if dim_x == 1:
    init[0]['x'] = size_x * scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(L)
    init[0]['x'][ind[-1]] = 0.3
if dim_x == 2:
    init[0]['x'] = size_x * scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs((L, dim_x))
    init[0]['x'][ind[-1]] = [0.3, 0.5]


# -------------
# MCMC
# -------------

iter = 100000
save_every = 100
nburn = int(iter * 0.25)
path = 'univ_rail_gamma0'
type_prop_x = 'normal'
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=False, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200, type_prop_x=type_prop_x)


# Save DF with posterior mean of w, x, sigma, c, t and attributes of nodes

w_mean = np.zeros(G.number_of_nodes())
sigma_mean = np.zeros(G.number_of_nodes())
c_mean = np.zeros(G.number_of_nodes())
t_mean = np.zeros(G.number_of_nodes())
x_mean0 = np.zeros(G.number_of_nodes())
x_mean1 = np.zeros(G.number_of_nodes())
for m in range(G.number_of_nodes()):
    w_mean[m] = np.mean([out[0][0][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    sigma_mean[m] = np.mean([out[0][3][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    c_mean[m] = np.mean([out[0][4][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    t_mean[m] = np.mean([out[0][5][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 1:
        x_mean0[m] = np.mean([out[0][12][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 2:
        x_mean0[m] = np.mean([out[0][12][j][m][0] for j in range(int(nburn/save_every), int(iter/save_every))])
        x_mean1[m] = np.mean([out[0][12][j][m][1] for j in range(int(nburn/save_every), int(iter/save_every))])

if dim_x == 1:
    posterior = pd.DataFrame({'x0': x_mean0, 'w': w_mean, 'sigma': sigma_mean, 'c': c_mean, 't': t_mean})
if dim_x == 2:
    posterior = pd.DataFrame({'x0': x_mean0, 'x1': x_mean1, 'w': w_mean, 'sigma': sigma_mean, 'c': c_mean, 't': t_mean})
posterior = posterior.reset_index()
posterior = posterior.merge(attributes, how='left', on='index')

posterior.to_csv(os.path.join('images', path, 'posterior.csv'))


# -------------
# PLOTS
# -------------

# 1. Estimated distance vs true distance plots (for few nodes)

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
for m in range(len(set_nodes)):
    plt.figure()
    plt.plot([out[i][12][j][set_nodes[m]] for j in range(len(out[i][12]))])
    plt.title('location %s' % int(posterior.iloc[set_nodes[m]]['index']))
    plt.savefig(os.path.join('images', path, 'x_%s' % int(posterior.iloc[set_nodes[m]]['index'])))
    plt.close()
for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.figure()
        plt.plot(dist_est[set_nodes[m], set_nodes[n], :])
        plt.axhline(dist[set_nodes[m], set_nodes[n]], color='red')
        plt.title('km distance between nodes %s, %s' % (int(posterior.iloc[set_nodes[m]]['index']),
                                                        int(posterior.iloc[set_nodes[n]]['index'])))
        plt.savefig(os.path.join('images', path, 'distance_%s_%s' % (int(posterior.iloc[set_nodes[m]]['index']),
                                                                     int(posterior.iloc[set_nodes[n]]['index']))))
        plt.close()

# 3. Plots of x vs east and north
plt.figure()
plt.scatter(posterior.iloc[range(L0)].east, posterior.iloc[range(L0)].x0)
plt.scatter(posterior.east[ind[-1]], posterior.x0[ind[-1]], color='red', label='fixed')
plt.xlabel('east')
plt.ylabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'east_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter(posterior.iloc[range(L0)].north, posterior.iloc[range(L0)].x0)
plt.scatter(posterior.north[ind[-1]], posterior.x0[ind[-1]], color='red', label='fixed')
plt.legend()
plt.xlabel('north')
plt.ylabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'north_vs_posterior_x0'))
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(posterior.iloc[range(L0)].east, posterior.iloc[range(L0)].north, np.array(posterior.iloc[range(L0)].x0))
ax.scatter(posterior.east[ind[-1]], posterior.north[ind[-1]], np.array(posterior.x0)[ind[-1]], color='red')
ax.set_xlabel('east')
ax.set_ylabel('north')
ax.set_zlabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'north_vs_east_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter(deg[range(L0)], posterior.iloc[range(L0)].x0)
plt.scatter(deg[ind[-1]], posterior.iloc[ind[-1]].x0, color='red')
plt.title('Degree vs posterior x first coordinate')
plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x0'))
plt.close()

if dim_x == 2:
    plt.figure()
    plt.scatter(posterior.iloc[range(L0)].east, posterior.iloc[range(L0)].x1)
    plt.scatter(posterior.east[ind[-1]], posterior.x1[ind[-1]], color='red')
    plt.savefig(os.path.join('images', path, 'east_vs_posterior_x1'))
    plt.close()
    plt.figure()
    plt.scatter(posterior.iloc[range(L0)].north, posterior.iloc[range(L0)].x1)
    plt.scatter(posterior.north[ind[-1]], posterior.x1[ind[-1]], color='red')
    plt.savefig(os.path.join('images', path, 'north_vs_posterior_x1'))
    plt.close()
    plt.figure()
    plt.scatter(deg[range(L0)], posterior.iloc[range(L0)].x1)
    plt.scatter(deg[ind[-1]], posterior.iloc[ind[-1]].x1, color='red')
    plt.title('Degree vs posterior x second coordinate')
    plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x1'))
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(posterior.iloc[range(L0)].east, posterior.iloc[range(L0)].north, np.array(posterior.iloc[range(L0)].x1))
    ax.scatter(posterior.east[ind[-1]], posterior.north[ind[-1]], np.array(posterior.x1)[ind[-1]], color='red')
    ax.set_xlabel('east')
    ax.set_ylabel('north')
    ax.set_zlabel('Posterior mean x')
    plt.savefig(os.path.join('images', path, 'north_vs_east_vs_posterior_x1'))
    plt.close()


# -------------
# POSTERIOR PREDICTIVE
# -------------

#f = open(os.path.join('images', path, 'posterior.csv'), "r")
#posterior = pd.read_csv(os.path.join('images', path, 'posterior.csv'))
w_p = posterior.w
x_p = posterior.x0
sigma_p = posterior.sigma[0]
c_p = posterior.c[0]
t_p = posterior.t[0]

prior = 'singlepl'
tau = 5
a_t = 200
b_t = 1
T = 0.000001
approximation = 'finite'
sampler = 'naive'
type_prop_x = 'tNormal'
type_prior_x = 'tNormal'


# compare degree distributions
def plt_deg_distr(deg, color='blue', label='', plot=True):
    deg = deg[deg > 0]
    num_nodes = len(deg)
    bins = np.array([2**i for i in range(11)])
    sizebins = (bins[1:] - bins[:-1])
    counts = np.histogram(deg, bins=bins)[0]
    freq = counts / num_nodes / sizebins
    if plot is True:
        plt.plot(bins[:-1], freq, 'o', color=color, label=label)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('deg')
        plt.ylabel('frequency')
    return freq


freq = {}
tot = 100
for i in range(tot):
    Gsim = GraphSampler(prior, approximation, sampler, sigma_p, c_p, t_p, tau, gamma, size_x, type_prior_x, dim_x,
                        a_t, b_t, print_=False, T=T, K=100, L=len(posterior))
    deg_Gsim = np.array(list(dict(Gsim.degree()).values()))
    freq[i] = plt_deg_distr(deg_Gsim, plot=False)

freq_ci = [scipy.stats.mstats.mquantiles([freq[i][j] for i in range(tot)],  prob=[0.025, 0.975])
            for j in range(len(freq[0]))]

bins = np.array([2**i for i in range(11)])
plt.figure()
plt.fill_between(bins[:-1], [freq_ci[i][0] for i in range(len(freq_ci))], [freq_ci[i][1] for i in range(len(freq_ci))],
                 color='powderblue')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('deg')
plt.ylabel('frequency')
deg_G = np.array(list(dict(G.degree()).values()))
plt_deg_distr(deg_G, color='blue', label='true')
plt.savefig(os.path.join('images', path, 'posterior_degrees_onlyhyperparams'))
plt.close()

# 4. Posterior coverage of distance
dist_est_fin = dist_est[:, :, range(int((iter + save_every) / save_every) -
                                    int((nburn + save_every) / save_every))]
dist_ci = [[scipy.stats.mstats.mquantiles(dist_est_fin[m, n, :], prob=[0.025, 0.975]) for m in range(L0)]
           for n in range(L0)]
true_in_ci = [[dist_ci[m][n][0] <= dist[m, n] <= dist_ci[m][n][1] for m in range(L0)] for n in range(L0)]
print('posterior coverage of distance = ', round(np.sum(true_in_ci) / (L0 ** 2) * 100, 1), '%')

print('End of experiment')