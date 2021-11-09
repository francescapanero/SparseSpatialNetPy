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
from mpl_toolkits.mplot3d import Axes3D


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
id_df['region'] = 'centre'
id_df['hub'] = 'no'
id_df.loc[(id_df.state == 'MI') | (id_df.state == 'IL') | (id_df.state == 'GA') | (id_df.state == 'OH')
          | (id_df.state == 'PA') | (id_df.state == 'TN') | (id_df.state == 'IN') | (id_df.state == 'ME')
          | (id_df.state == 'NH') | (id_df.state == 'MA') | (id_df.state == 'VT') | (id_df.state == 'NY')
          | (id_df.state == 'NJ') | (id_df.state == 'DE') | (id_df.state == 'MD') | (id_df.state == 'VA')
          | (id_df.state == 'NC') | (id_df.state == 'FL') | (id_df.state == 'SC') | (id_df.state == 'CT')
          | (id_df.state == 'RI') | (id_df.state == 'WV'), 'region'] = 'east'
id_df.loc[(id_df.state == 'CO') | (id_df.state == 'NV') | (id_df.state == 'CA') | (id_df.state == 'WY')
          | (id_df.state == 'MT') | (id_df.state == 'CA') | (id_df.state == 'NM')
          | (id_df.state == 'UT') | (id_df.state == 'AZ') | (id_df.state == 'WA') | (id_df.state == 'OR')
          | (id_df.state == 'ID'), 'region'] = 'west'
id_df.loc[(id_df.iata == 'JFK') | (id_df.iata == 'EWR') | (id_df.iata == 'MIA') | (id_df.iata == 'BOS') |
          (id_df.iata == 'DCA') | (id_df.iata == 'LAX') | (id_df.iata == 'ATL') | (id_df.iata == 'FLL') |
          (id_df.iata == 'ORD') | (id_df.iata == 'MSP') | (id_df.iata == 'BNA') | (id_df.iata == 'CLE') |
          (id_df.iata == 'DTW') | (id_df.iata == 'DFW') | (id_df.iata == 'BUR') | (id_df.iata == 'IAD'), 'hub'] = 'yes'
id_df = id_df.set_index('num_id')
id_dict = id_df.to_dict(orient='index')

nx.set_node_attributes(G, id_dict)

# remove nodes not in US (exclude Alaska and Hawaii as well)
for i in nodes:
    if G.nodes[i] == {}:
        G.remove_node(i)
# remove isolated nodes
for i in [node for node, degree in G.degree() if degree == 0]:
    G.remove_node(i)
G = nx.relabel.convert_node_labels_to_integers(G)

attributes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
attributes = attributes.reset_index()

deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
for i in range(10):
    print(np.sort(deg)[len(deg)-10+i], G.nodes[biggest_deg[i]]['iata'])

# find highest and lowest deg nodes
set_nodes = biggest_deg[-7:]
            # np.concatenate((np.argsort(deg)[len(deg)-7: len(deg)], np.argsort(deg)[0: 2]))

# Check distance distribution
l = G.number_of_nodes()
dist = np.zeros((l, l))
p_ij = np.zeros((l, l))
lat = np.zeros(l)
long = np.zeros(l)
for i in range(l):
    lat[i] = G.nodes[i]['latitude'] * math.pi / 180
    long[i] = G.nodes[i]['longitude'] * math.pi / 180
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) *
#                                                         np.cos(lat[j]) * np.cos(long[j] - long[i]))
for i in range(l):
    for j in range(i+1, l):
        dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
                                                   * np.cos(long[j] - long[i]))
        dist[j, i] = dist[i, j]
dist = dist / np.max(dist)
# dist_ = dist[dist > 0]
# plt.figure()
# plt.hist(dist_, bins=50, density=True)

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


# -------------
# SET UP MCMC
# -------------

L0 = G.number_of_nodes()
nodes_added = 150
L = G.number_of_nodes() + nodes_added
G.add_nodes_from(range(L0, L))

deg = np.array(list(dict(G.degree()).values()))
ind = np.argsort(deg)
index = ind[1:len(ind)-1]
gamma = .2

path = 'univ_airports'
f = open(os.path.join('images', path, 'posterior.csv'), "r")
posterior = pd.read_csv(os.path.join('images', path, 'posterior.csv'))
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
sampler = 'layers'
type_prop_x = 'tNormal'
type_prior_x = 'tNormal'

init = {}
init[0] = {}
init[0]['sigma'] = 0.4
init[0]['c'] = 1
init[0]['t'] = 100
size_x = 1
init[0]['size_x'] = size_x
dim_x = 1
lower = 0
upper = size_x
mu = 0.3
sigma = 0.1
if dim_x == 1:
    init[0]['x'] = x_p.copy()
    init[0]['x'][index] = size_x * scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(len(index))
if dim_x == 2:
    init[0]['x'] = x_p.copy()
    init[0]['x'][index] = size_x * scipy.stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu,
                                                         scale=sigma).rvs((len(index), dim_x))

Gsim = GraphSampler(prior, approximation, sampler, sigma_p, c_p, t_p, tau, gamma, size_x, type_prior_x, dim_x,
                    a_t, b_t, print_=False, T=T, K=100, L=len(posterior), x=x_p, w=w_p)

iter = 300000
save_every = 100
nburn = int(iter * 0.25)
path = 'univ_airports_simulated'
out = chain.mcmc_chains([Gsim], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200)


def plt_deg_distr(deg, color='blue', label='', plot=True):
    deg = deg[deg > 0]
    num_nodes = len(deg)
    bins = np.array([2**i for i in range(11)])
    sizebins = (bins[1:] - bins[:-1])
    counts = np.histogram(deg, bins=bins)[0]
    freq = counts / num_nodes / sizebins
    freq = counts / sizebins
    if plot is True:
        plt.scatter(bins[:-1], freq, color=color, label=label)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('deg')
        plt.ylabel('frequency')
    return freq


deg_Gsim = np.array(list(dict(Gsim.degree()).values()))
pd.DataFrame(deg_Gsim).to_csv(os.path.join('images', path, 'deg_Gsim.csv'))

plt_deg_distr(deg_Gsim, color='red', label='simulated')
plt_deg_distr(deg, color='blue', label='true')
plt.savefig(os.path.join('images', path, 'posterior_degrees'))

biggest_deg = np.argsort(deg_Gsim)[len(deg_Gsim)-10: len(deg_Gsim)]
set_nodes = biggest_deg[-7:]
for m in range(len(set_nodes)):
    plt.figure()
    plt.plot([out[0][12][j][set_nodes[m]] for j in range(len(out[0][12]))])
    plt.axhline(x_p[set_nodes[m]], color='red')
    plt.title('location %i' % set_nodes[m])
    plt.savefig(os.path.join('images', path, 'x_%i' % set_nodes[m]))
    plt.close()

print('End of experiment')