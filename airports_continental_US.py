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
deg_freq_G = nx.degree_histogram(G)
plt.figure()
plt.loglog(deg_freq_G, 'go-')

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

# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# p_ij = np.zeros((l, l))
# lat = np.zeros(l)
# long = np.zeros(l)
# for i in range(l):
#     lat[i] = G.nodes[i]['latitude'] * math.pi / 180
#     long[i] = G.nodes[i]['longitude'] * math.pi / 180
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                                                        * np.cos(long[j] - long[i]))
# dist = dist[dist != 0]
# plt.figure()
# plt.hist(dist, bins=50)
# plt.figure()
# plt.hist(dist[dist > 800], bins=50)

# size_x = 4300
# prior = 'singlepl'
# gamma = 0.2
# c = 1.2
# sigma = 0.1
# t = 100
# tau = 5
# K = 100  # number of layers, for layers sampler
# T = 0.000001
# a_t = 200
# b_t = 1
# approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
# sampler = 'layers'  # can be 'layers' or 'naive'
# Gsim = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=G.number_of_nodes()+300)
# deg = np.array(list(dict(Gsim.degree()).values()))
# x = np.array([Gsim.nodes[i]['x'] for i in range(Gsim.number_of_nodes())])
# dist_sim = np.zeros((Gsim.number_of_nodes(), Gsim.number_of_nodes()))
# for i in range(Gsim.number_of_nodes()):
#     for j in [n for n in Gsim.neighbors(i)]:
#         if j > i:
#             dist_sim[i, j] = np.abs(x[i] - x[j])
# plt.figure()
# plt.hist(dist_sim[dist_sim!=0], bins=50)

# prepare dataset for MCMC
L0 = G.number_of_nodes()
nodes_added = 300
L = G.number_of_nodes() + nodes_added
G.add_nodes_from(range(L0, L))

deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[1:len(ind)-1]

init = {}
init[0] = {}
init[0]['sigma'] = 0.2  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = 1
init[0]['t'] = np.sqrt(G.number_of_edges())
size_x = 20000
init[0]['size_x'] = size_x
init[0]['x'] = size_x * np.random.uniform(0, 1, L)

iter = 1000000
save_every = 1000
nburn = int(iter * 0.25)
path = 'airport_gamma_point2'
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200)

dist_est = np.zeros((len(set_nodes), len(set_nodes), len(out[0][11])))
i = 0
for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        for j in range(len(out[i][12])):
            dist_est[m, n, j] = np.abs(out[i][12][j][m]-out[i][12][j][n])

for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.figure()
        plt.plot(dist_est[m, n, :])
        plt.axhline(dist[m, n])
        plt.title('km distance b/w nodes %i, %i' % (set_nodes[m], set_nodes[n]))
        plt.savefig(os.path.join('images', path, 'distance_nodes_%i_%i' % (set_nodes[m], set_nodes[n])))
        plt.close()

x_mean = np.zeros(G.number_of_nodes()-nodes_added)
longit = np.zeros(G.number_of_nodes()-nodes_added)
latit = np.zeros(G.number_of_nodes()-nodes_added)
for m in range(G.number_of_nodes()-nodes_added):
    x_mean[m] = np.mean([out[0][12][j][m] for j in range(int(nburn/save_every), int(iter/save_every))])
    longit[m] = G.nodes[m]['longitude']
    latit[m] = G.nodes[m]['latitude']
plt.figure()
plt.scatter(longit, x_mean)
plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x'))
plt.figure()
plt.scatter(latit, x_mean)
plt.savefig(os.path.join('images', path, 'latitude_vs_posterior_x'))



