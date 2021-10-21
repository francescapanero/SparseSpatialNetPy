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

G = nx.relabel.convert_node_labels_to_integers(G)


gamma = 0
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1


# prepare dataset for MCMC
L0 = G.number_of_nodes()
L = G.number_of_nodes() + 300
G.add_nodes_from(range(L0, L))

deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[1:len(ind)-1]

init = {}
init[0] = {}
init[0]['sigma'] = 0.3  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = 1
init[0]['t'] = np.sqrt(G.number_of_edges())
size_x = 20000
init[0]['size_x'] = size_x
init[0]['x'] = size_x * np.random.uniform(0, 1, L)

iter = 30000
nburn = int(iter * 0.25)

out = chain.mcmc_chains([G], iter, nburn, range(len(G)),
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=False, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=100, plot=True,  path='airport_nospace_noreduction',
                        save_out=False, save_data=False, init=init, a_t=200)






