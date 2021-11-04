import numpy as np
import mcmc_chains as chain
import networkx as nx
import re
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy
import os
from utils.GraphSampler import *

G = nx.read_edgelist('data/airports/airports10.txt', nodetype=int, data=(('weight', float),))

f = open("data/airports/USairport_2010_codes.txt", "r")
id_dict = {}
names = []
for line in f:
    line = re.sub('"', '', line)
    key, value = line.split()
    id_dict[int(key)] = value
    names.append(value)
id_df = pd.DataFrame.from_dict(id_dict, orient='index')
id_df = id_df.reset_index()
id_df = id_df.rename(columns={0: 'iata', 'index': 'num_id'})

# g = open("data/airports/attributes10.txt", "r")
# longitude = {}
# latitude = {}
# lonlat_dict = {}
# for line in g.readlines()[1:]:
#     iata, icao, lat, long, altitude = line.split(" ")[-7:-2]
#     iata = iata[1:-1]
#     lonlat_dict[iata] = [float(lat), float(long), float(altitude)]
# lonlat_df = pd.DataFrame.from_dict(lonlat_dict, orient='index')
# lonlat_df = lonlat_df.reset_index()
# lonlat_df = lonlat_df.rename(columns={'index': 'iata', 0: 'longitude', 1: 'latitude', 2: 'altitude'})

g = pd.read_csv("data/airports/253021595_T_MASTER_CORD_All_All.csv")
g = g[['AIRPORT', 'LATITUDE', 'LONGITUDE', 'AIRPORT_COUNTRY_CODE_ISO', 'AIRPORT_STATE_CODE']]
lonlat_df = g.rename(columns={'AIRPORT': 'iata', 'LONGITUDE': 'longitude', 'LATITUDE': 'latitude',
                              'AIRPORT_STATE_CODE': 'state', 'AIRPORT_COUNTRY_CODE_ISO': 'country'})


id_df = id_df.merge(lonlat_df, on='iata', how='left')
nodes = list(G.nodes)
for i in id_df.num_id:
    if i not in nodes:
        id_df = id_df.drop(id_df.loc[id_df.num_id == i].index)
id_df = id_df.drop_duplicates(subset='num_id')
id_df = id_df[(id_df.state != 'AK') & (id_df.state != 'HI') & (id_df.country == 'US')]
id_df = id_df.set_index('num_id')
id_dict = id_df.to_dict(orient='index')

nx.set_node_attributes(G, id_dict)


# remove nodes not in US (exclude Alaska as well)
for i in nodes:
    if G.nodes[i] == {}:
        G.remove_node(i)
# remove isolated nodes
for i in [node for node, degree in G.degree() if degree == 0]:
    G.remove_node(i)
G = nx.relabel.convert_node_labels_to_integers(G)

deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
for i in range(10):
    print(np.sort(deg)[len(deg)-i-1], G.nodes[biggest_deg[i]]['iata'])
# full dataset:
# Miami, Huston, Minneapolis, Newark, Denver, JFK, LA, Chicago, Washington, Atlanta
# dataset constrained to only US (not alaska) airports
# Nashville, Cleveland, Detroit, Dallas, Burbank, Washington, Chicago, Atlanta, Mississipi, Denver
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')

gamma = 0.2
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1

# find highest and lowest deg nodes
deg = np.array(list(dict(G.degree()).values()))
set_nodes = np.concatenate((np.argsort(deg)[len(deg)-5: len(deg)], np.argsort(deg)[0: 2]))
l = len(set_nodes)
lat = np.zeros(l)
long = np.zeros(l)
dist = np.zeros((l, l))
p_ij = np.zeros((l, l))
for i in range(l):
    lat[i] = G.nodes[set_nodes[i]]['latitude'] * math.pi / 180
    long[i] = G.nodes[set_nodes[i]]['longitude'] * math.pi / 180
    print(G.nodes[set_nodes[i]]['iata'], lat[i], G.nodes[set_nodes[i]]['latitude'], long[i], G.nodes[set_nodes[i]]['longitude'])
for i in range(len(set_nodes)):
    for j in range(i+1, len(set_nodes)):
        dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                   * np.cos(long[j] - long[i]))
        p_ij[i, j] = 1 / ((1 + dist[i, j]) ** gamma)
p_ij = p_ij + np.transpose(p_ij)
np.fill_diagonal(p_ij, 1)

# Check distance distribution

l = G.number_of_nodes()
dist = np.zeros((l, l))
p_ij = np.zeros((l, l))
lat = np.zeros(l)
long = np.zeros(l)
for i in range(l):
    lat[i] = G.nodes[i]['latitude'] * math.pi / 180
    long[i] = G.nodes[i]['longitude'] * math.pi / 180
for i in range(l):
    for j in [n for n in G.neighbors(i)]:
        if j > i:
            dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                       * np.cos(long[j] - long[i]))
# # if you want to remove flights with distance < threshold (eg in the same city)
# # I wasn't able to make the distance distribution in such a way that looks more similar to the unif one
# ind_closeness = np.where((dist > 1) & (dist < 50))
# while len(ind_closeness[0]) != 0:
#     G = nx.contracted_nodes(G, ind_closeness[0][0], ind_closeness[1][0])
#     G = nx.relabel.convert_node_labels_to_integers(G)
#     dist = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
#     for i in range(G.number_of_nodes()):
#         for j in [n for n in G.neighbors(i)]:
#             if j > i:
#                 dist[i, j] = 1.609344 * 3963.0 * np.arccos(
#                     (np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                     * np.cos(long[j] - long[i]))
#     ind_closeness = np.where((dist > 1) & (dist < 50))
dist = dist[dist != 0] / np.max(dist)
plt.figure()
plt.hist(dist, bins=50, density=True)
# plt.figure()
# plt.hist(dist[dist > 800], bins=50)

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
# dim_x = 2
# gamma = 0.2
# Gsim = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x,
#                     a_t, b_t, T=T, K=K, L=G.number_of_nodes()+300)
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
# plt.hist(dist_sim[dist_sim != 0], bins=50, density=True)

# prepare dataset for MCMC
L0 = G.number_of_nodes()
nodes_added = 150
L = G.number_of_nodes() + nodes_added
G.add_nodes_from(range(L0, L))

deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[1:len(ind)-1]

init = {}
init[0] = {}
init[0]['sigma'] = 0.4  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = 1
init[0]['t'] = np.sqrt(G.number_of_edges())
size_x = 1
init[0]['size_x'] = size_x
dim_x = 2
lower = 0
upper = size_x
mu = 0.3
sigma = 0.3
init[0]['x'] = size_x * scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs((L, dim_x))
init[0]['x'][ind[-1]] = [0.5, 0.5]

iter = 250000
save_every = 100
nburn = int(iter * 0.25)
path = 'bivx_airports_continentalUS_gammapoint2_xfixedpoint5'
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200)

l = len(set_nodes)
dist_est = np.zeros((l, l, len(out[0][11])))
i = 0
for m in range(l):
    for n in range(m + 1, l):
        for j in range(len(out[i][12])):
            if dim_x == 1:
                np.abs(out[i][12][j][m] - out[i][12][j][n])
            if dim_x == 2:
                dist_est[m, n, j] = np.sqrt((out[i][12][j][m][0]-out[i][12][j][n][0])**2 +
                                            (out[i][12][j][m][1]-out[i][12][j][n][1])**2)
lat = np.zeros(l)
long = np.zeros(l)
dist = np.zeros((l, l))
for i in range(l):
    lat[i] = G.nodes[set_nodes[i]]['latitude'] * math.pi / 180
    long[i] = G.nodes[set_nodes[i]]['longitude'] * math.pi / 180
for i in range(l):
    for j in range(i+1, l):
        dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                   * np.cos(long[j] - long[i]))
dist = dist / np.max(dist)

for m in range(l):
    for n in range(m + 1, l):
        plt.figure()
        plt.plot(dist_est[m, n, :])
        plt.axhline(dist[m, n], color='red')
        plt.title('km distance b/w nodes %i, %i' % (set_nodes[m], set_nodes[n]))
        plt.savefig(os.path.join('images', path, 'distance_nodes_%i_%i' % (set_nodes[m], set_nodes[n])))
        plt.close()

x_mean0 = np.zeros(G.number_of_nodes()-nodes_added)
x_mean1 = np.zeros(G.number_of_nodes()-nodes_added)
longit = np.zeros(G.number_of_nodes()-nodes_added)
latit = np.zeros(G.number_of_nodes()-nodes_added)
for m in range(G.number_of_nodes()-nodes_added):
    if dim_x == 1:
        x_mean0[m] = np.mean([out[0][12][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 2:
        x_mean0[m] = np.mean([out[0][12][j][m][0] for j in range(int(nburn/save_every), int(iter/save_every))])
        x_mean1[m] = np.mean([out[0][12][j][m][1] for j in range(int(nburn/save_every), int(iter/save_every))])
    longit[m] = G.nodes[m]['longitude']
    latit[m] = G.nodes[m]['latitude']

plt.figure()
plt.scatter(longit, x_mean0)
plt.scatter(longit[ind[-1]], x_mean0[ind[-1]], color='red')
plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x0'))
plt.figure()
plt.scatter(latit, x_mean0)
plt.scatter(latit[ind[-1]], x_mean0[ind[-1]], color='red')
plt.savefig(os.path.join('images', path, 'latitude_vs_posterior_x0'))

if dim_x == 2:
    plt.figure()
    plt.scatter(longit, x_mean1)
    plt.scatter(longit[ind[-1]], x_mean1[ind[-1]], color='red')
    plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x1'))
    plt.figure()
    plt.scatter(latit, x_mean1)
    plt.scatter(latit[ind[-1]], x_mean1[ind[-1]], color='red')
    plt.savefig(os.path.join('images', path, 'latitude_vs_posterior_x1'))

l = L0
dist = np.zeros((l, l))
lat = np.zeros(l)
long = np.zeros(l)
for i in range(l):
    lat[i] = G.nodes[i]['latitude'] * math.pi / 180
    long[i] = G.nodes[i]['longitude'] * math.pi / 180
for i in range(l):
    for j in range(i+1, l):
        dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                   * np.cos(long[j] - long[i]))
dist = dist + np.transpose(dist)
dist = dist / np.max(dist)
p_ij = 1 / ((1 + dist)**gamma)

new_index = ind[nodes_added:len(ind)]
size = len(new_index)
p_ij_est = out[0][11]
p_ij_est_fin = [[p_ij_est[k][h, :] for k in range(int((nburn + save_every) / save_every),
                                                  int((iter + save_every) / save_every))] for h in new_index]

emp_ci = []
true_in_ci = []
for j in range(len(new_index)):
    # compute posterior coverage of these nodes
    emp_ci.append(
        [scipy.stats.mstats.mquantiles(
            [p_ij_est_fin[j][k][m] for k in range(int((iter + save_every) / save_every) -
                                                  int((nburn + save_every) / save_every))],
            prob=[0.025, 0.975]) for m in new_index])
    true_in_ci.append([emp_ci[j][k][0] <= p_ij[new_index[j], k] <= emp_ci[j][k][1]
                       for k in new_index])
total = sum([sum(true_in_ci[m]) for m in new_index])
print('posterior coverage of p_ij in chain %i' % l, ' = ', round(total / (size ** 2) * 100, 1), '%')

# plot p_ij and print posterior coverage for 5 lowest and 5 highest deg nodes
index = np.concatenate((new_index[0:5], new_index[len(new_index) - 5: len(new_index)]))
size = l
p_ij_est = out[0][11]
p_ij_est_fin = [[p_ij_est[k][h, :] for k in range(int((nburn + save_every) / save_every),
                                                  int((iter + save_every) / save_every))] for h in index]
emp_ci = []
true_in_ci = []
for j in range(len(index)):
    # compute posterior coverage of these nodes
    emp_ci.append(
        [scipy.stats.mstats.mquantiles(
            [p_ij_est_fin[j][k][m] for k in range(int((iter + save_every) / save_every) -
                                                  int((nburn + save_every) / save_every))],
            prob=[0.025, 0.975]) for m in new_index])
    true_in_ci.append([emp_ci[j][k][0] <= p_ij[index[j], k] <= emp_ci[j][k][1]
                       for k in new_index])
    # plot p_ij across these nodes
    plt.figure()
    for k in range(len(index)):
        plt.plot((k + 1, k + 1), (emp_ci[j][index[k]][0], emp_ci[j][index[k]][1]),
                 color='cornflowerblue', linestyle='-', linewidth=2)
        plt.plot(k + 1, p_ij[index[j], index[k]], color='navy', marker='o', markersize=5)
    plt.savefig(os.path.join('images', path, 'pij_deg%i_index%i_chain%i' % (deg[deg>0][index[j]], j, l)))
    plt.close()
    if 'distances' in G.graph:
        print('posterior coverage in chain %i' % l, ' of true p_ij for node with deg %i' % deg[deg>0][index[j]],
              ' = ', round(sum(true_in_ci[j]) / size * 100, 1), '%')
index_ = index[-5:]
for n in range(len(index_)):
    for m in range(n + 1, len(index_)):
        plt.figure()
        plt.plot([p_ij_est[k][index_[n], index_[m]] for k in range(len(p_ij_est))])
        plt.axhline(p_ij[index_[n], index_[m]], color='red')
        plt.savefig(os.path.join('images', path, 'trace_pij_nodes%i_%i_chain%i' % (index_[n], index_[m], l)))
        plt.close()

