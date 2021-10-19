import numpy as np
import mcmc_chains as chain
import networkx as nx
import re
import pandas as pd
import math
import matplotlib.pyplot as plt

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
#id_df = id_df.set_index('num_id')
# id_df = id_df.rename(columns={0: 'iata'})

g = open("data/airports/attributes10.txt", "r")
longitude = {}
latitude = {}
lonlat_dict = {}
for line in g.readlines()[1:]:
    iata, icao, lat, long, altitude = line.split(" ")[-7:-2]
    iata = iata[1:-1]
    lonlat_dict[iata] = [float(lat), float(long), float(altitude)]
lonlat_df = pd.DataFrame.from_dict(lonlat_dict, orient='index')
lonlat_df = lonlat_df.reset_index()
lonlat_df = lonlat_df.rename(columns={'index': 'iata', 0: 'longitude', 1: 'latitude', 2: 'altitude'})

nodes = list(G.nodes)
for i in id_df.num_id:
    if i not in nodes:
        id_df = id_df.drop(id_df.loc[id_df.num_id == i].index)
id_df = id_df.merge(lonlat_df, on='iata', how='left')
id_df = id_df.set_index('num_id')
id_dict = id_df.to_dict(orient='index')

nx.set_node_attributes(G, id_dict)

Gred = G.copy()

for i in id_dict.keys():
    if math.isnan(id_dict[i]['longitude']) or math.isnan(id_dict[i]['latitude']):
        Gred.remove_node(i)

Gred = nx.relabel.convert_node_labels_to_integers(Gred)
G = nx.relabel.convert_node_labels_to_integers(G)

# deg = np.array(list(dict(Gred.degree()).values()))
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for i in range(10):
#     print(biggest_deg[i], Gred.nodes[biggest_deg[i]]['iata'])
# deg_freq_Gred = nx.degree_histogram(Gred)
# plt.figure()
# plt.loglog(deg_freq_Gred, 'go-')
#
# deg = np.array(list(dict(G.degree()).values()))
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for i in range(10):
#     print(biggest_deg[i], G.nodes[biggest_deg[i]]['iata'])
# # Miami, Huston, Minneapolis, Newark, Denver, JFK, LA, Chicago, Washington, Atlanta
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')

# use reduced dataset
G = Gred.copy()

gamma = 0
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1

# # find highest and lowest deg nodes
# deg = np.array(list(dict(G.degree()).values()))
# set_nodes = np.concatenate((np.argsort(deg)[len(deg)-6: len(deg)], np.argsort(deg)[0: 2]))
# l = len(set_nodes)
# lat = np.zeros(l)
# long = np.zeros(l)
# dist = np.zeros((l, l))
# p_ij = np.zeros((l, l))
# for i in range(l):
#     lat[i] = G.nodes[set_nodes[i]]['latitude'] * math.pi / 180
#     long[i] = G.nodes[set_nodes[i]]['longitude'] * math.pi / 180
#     print(G.nodes[set_nodes[i]]['iata'], lat[i], G.nodes[set_nodes[i]]['latitude'], long[i], G.nodes[set_nodes[i]]['longitude'])
# for i in range(len(set_nodes)):
#     for j in range(i+1, len(set_nodes)):
#         dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                                                    * np.cos(long[j] - long[i]))
#         p_ij[i, j] = 1 / ((1 + dist[i, j]) ** gamma)
# p_ij = p_ij + np.transpose(p_ij)
# np.fill_diagonal(p_ij, 1)


# prepare dataset for MCMC
L0 = G.number_of_nodes()
L = G.number_of_nodes() + 300
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

iter = 3000
nburn = int(iter * 0.25)

out = chain.mcmc_chains([G], iter, nburn, range(len(G)),
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=False, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=100, plot=True,  path='airport_nospace',
                        save_out=False, save_data=False, init=init, a_t=200)

# p = np.zeros((len(set_nodes), len(set_nodes), len(out[0][11])))
# i = 0
# for m in range(len(set_nodes)):
#
#     for n in range(m + 1, len(set_nodes)):
#         for j in range(len(out[i][11])):
#             print(out[i][12][j][m])
#             p[m, n, j] = out[i][11][j][m, n]
#
# for m in range(len(set_nodes)):
#     for n in range(m + 1, len(set_nodes)):
#         plt.figure()
#         plt.plot(p[m, n, :])
#         plt.axhline(p_ij[m, n])


# import scipy
# from matplotlib import pyplot
# size_x = 20000
# x = size_x * np.random.rand(1275)
# x = x[:, None]
# p_unif = scipy.spatial.distance.pdist(x, 'euclidean')
# plt.figure()
# pyplot.hist(p_unif, bins=50)
#
# lower = 0
# upper = 1
# mu = 0.5
# sigma = 0.1
# x = scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1000)
# x = x[:, None]
# p_unif = scipy.spatial.distance.pdist(x, 'euclidean')
# plt.figure()
# pyplot.hist(p_unif, bins=50)
#
# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# p_ij = np.zeros((l, l))
# lat = np.zeros(l)
# long = np.zeros(l)
# for i in range(l):
#     lat[i] = G.nodes[i]['latitude'] * math.pi / 180
#     long[i] = G.nodes[i]['longitude'] * math.pi / 180
# for i in range(l):
#     for j in range(i+1, l):
#         dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                                                    * np.cos(long[j] - long[i]))
# dist = dist[dist != 0]
# plt.figure()
# pyplot.hist(dist.flatten(), bins=50)




