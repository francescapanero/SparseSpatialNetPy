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

# -------------
# CREATE DATASET
# -------------

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

gamma = 2
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1

init = {}
init[0] = {}
init[0]['sigma'] = 0.4  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = 1
init[0]['t'] = np.sqrt(G.number_of_edges())
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

iter = 400000
save_every = 100
nburn = int(iter * 0.25)
path = 'univ_airports_gamma2_updateonlyxn'
type_prop_x = 'tNormal'
sigma = 0.25
c = 0.65
t = 65
for i in G.nodes():
    G.nodes[i]['w0'] = 1
    G.nodes[i]['w'] = 1
    G.nodes[i]['u'] = tp.tpoissrnd((G.number_of_nodes() * sigma / t) ** (1 / sigma))
G.graph['sigma'] = sigma
G.graph['c'] = c
G.graph['t'] = t
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=False, c=False, t=False, tau=False, w0=False, n=True, u=False, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200, type_prop_x=type_prop_x)


# Save DF with posterior mean of w, x, sigma, c, t and attributes of nodes

# w_mean = np.zeros(G.number_of_nodes())
sigma_mean = np.zeros(G.number_of_nodes())
c_mean = np.zeros(G.number_of_nodes())
t_mean = np.zeros(G.number_of_nodes())
x_mean0 = np.zeros(G.number_of_nodes())
x_mean1 = np.zeros(G.number_of_nodes())
for m in range(G.number_of_nodes()):
    # w_mean[m] = np.mean([out[0][0][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    # sigma_mean[m] = np.mean([out[0][3][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    # c_mean[m] = np.mean([out[0][4][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    # t_mean[m] = np.mean([out[0][5][j] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 1:
        x_mean0[m] = np.mean([out[0][12][j][m] for j in range(int(nburn / save_every), int(iter / save_every))])
    if dim_x == 2:
        x_mean0[m] = np.mean([out[0][12][j][m][0] for j in range(int(nburn/save_every), int(iter/save_every))])
        x_mean1[m] = np.mean([out[0][12][j][m][1] for j in range(int(nburn/save_every), int(iter/save_every))])

if dim_x == 1:
    # posterior = pd.DataFrame({'x0': x_mean0, 'w': w_mean, 'sigma': sigma_mean, 'c': c_mean, 't': t_mean})
    posterior = pd.DataFrame({'x0': x_mean0})
if dim_x == 2:
    # posterior = pd.DataFrame({'x0': x_mean0, 'x1': x_mean1, 'w': w_mean, 'sigma': sigma_mean, 'c': c_mean, 't': t_mean})
    posterior = pd.DataFrame({'x0': x_mean0, 'x1': x_mean1})
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
    plt.title('location %s' % posterior.iloc[set_nodes[m]].iata)
    plt.savefig(os.path.join('images', path, 'x_%s' % posterior.iloc[set_nodes[m]].iata))
    plt.close()

for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.figure()
        plt.plot(dist_est[set_nodes[m], set_nodes[n], :])
        plt.axhline(dist[set_nodes[m], set_nodes[n]], color='red')
        plt.title('km distance between nodes %s, %s' % (posterior.iloc[set_nodes[m]].iata,
                                                    posterior.iloc[set_nodes[n]].iata))
        plt.savefig(os.path.join('images', path, 'distance_%s_%s' % (posterior.iloc[set_nodes[m]].iata,
                                                                     posterior.iloc[set_nodes[n]].iata)))
        plt.close()

# 3. Plots of x vs longitude and latitude
plt.figure()
plt.scatter(posterior.iloc[range(L0)].longitude, posterior.iloc[range(L0)].x0)
plt.scatter(posterior.longitude[ind[-1]], posterior.x0[ind[-1]], color='red', label='DEN (fixed)')
plt.scatter(posterior.longitude[posterior.hub=='yes'], posterior.x0[posterior.hub=='yes'], color='black', label='Hub')
for i in posterior.index[posterior.hub=='yes'].tolist():
    plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].longitude, posterior.iloc[i].x0))
plt.legend()
plt.xlabel('Longitude (degrees)')
plt.ylabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter(posterior.iloc[range(L0)].latitude, posterior.iloc[range(L0)].x0)
plt.scatter(posterior.latitude[ind[-1]], posterior.x0[ind[-1]], color='red', label='DEN (fixed)')
plt.scatter(posterior.latitude[posterior.hub=='yes'], posterior.x0[posterior.hub=='yes'], color='black', label='Hub')
for i in posterior.index[posterior.hub=='yes'].tolist():
    plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].latitude, posterior.iloc[i].x0))
plt.legend()
plt.xlabel('Latitude (degrees)')
plt.ylabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'latitude_vs_posterior_x0'))
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(posterior.iloc[range(L0)].longitude, posterior.iloc[range(L0)].latitude, np.array(posterior.iloc[range(L0)].x0))
ax.scatter(posterior.longitude[ind[-1]], posterior.latitude[ind[-1]], np.array(posterior.x0)[ind[-1]], color='red')
ax.scatter(posterior.longitude[posterior.hub=='yes'], posterior.latitude[posterior.hub=='yes'],
           posterior.x0[posterior.hub=='yes'], color='black', label='hub')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Posterior mean x')
plt.savefig(os.path.join('images', path, 'latitude_vs_longitude_vs_posterior_x0'))
plt.close()

plt.figure()
plt.scatter(deg[range(L0)], posterior.iloc[range(L0)].x0)
plt.scatter(deg[ind[-1]], posterior.iloc[ind[-1]].x0, color='red')
plt.scatter(deg[posterior.hub=='yes'], posterior.x0[posterior.hub=='yes'], color='black')
for i in posterior.index[posterior.hub=='yes'].tolist():
    plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].latitude, posterior.iloc[i].x0))
plt.xlabel('Degree')
plt.ylabel('Posterior mean x')
plt.title('Degree vs posterior x first coordinate')
plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x0'))
plt.close()

if dim_x == 2:
    plt.figure()
    plt.scatter(posterior.iloc[range(L0)].longitude, posterior.iloc[range(L0)].x1)
    plt.scatter(posterior.longitude[ind[-1]], posterior.x1[ind[-1]], color='red')
    plt.scatter(posterior.longitude[posterior.hub == 'yes'], posterior.x1[posterior.hub == 'yes'],
               color='black', label='hub')
    for i in posterior.index[posterior.hub == 'yes'].tolist():
        plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].longitude, posterior.iloc[i].x0))
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Posterior mean x second coordinate')
    plt.savefig(os.path.join('images', path, 'longitude_vs_posterior_x1'))
    plt.close()
    plt.figure()
    plt.scatter(posterior.iloc[range(L0)].latitude, posterior.iloc[range(L0)].x1)
    plt.scatter(posterior.latitude[ind[-1]], posterior.x1[ind[-1]], color='red')
    plt.scatter(posterior.latitude[posterior.hub == 'yes'], posterior.x1[posterior.hub == 'yes'],
               color='black', label='hub')
    for i in posterior.index[posterior.hub == 'yes'].tolist():
        plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].latitude, posterior.iloc[i].x0))
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Posterior mean x second coordinate')
    plt.savefig(os.path.join('images', path, 'latitude_vs_posterior_x1'))
    plt.close()
    plt.figure()
    plt.scatter(deg[range(L0)], posterior.iloc[range(L0)].x1)
    plt.scatter(deg[ind[-1]], posterior.iloc[ind[-1]].x1, color='red')
    plt.scatter(deg[posterior.hub == 'yes'], posterior.x1[posterior.hub == 'yes'], color='black')
    for i in posterior.index[posterior.hub == 'yes'].tolist():
        plt.annotate(posterior.iloc[i].iata, (posterior.iloc[i].latitude, posterior.iloc[i].x0))
    plt.xlabel('Degree')
    plt.ylabel('Posterior mean x second coordinate')
    plt.title('Degree vs posterior x second coordinate')
    plt.savefig(os.path.join('images', path, 'deg_vs_posterior_x1'))
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(posterior.iloc[range(L0)].longitude, posterior.iloc[range(L0)].latitude, np.array(posterior.iloc[range(L0)].x1))
    ax.scatter(posterior.longitude[ind[-1]], posterior.latitude[ind[-1]], np.array(posterior.x1)[ind[-1]], color='red')
    ax.scatter(posterior.longitude[posterior.hub == 'yes'], posterior.latitude[posterior.hub == 'yes'],
               posterior.x1[posterior.hub == 'yes'], color='black', label='hub')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Posterior mean x')
    plt.savefig(os.path.join('images', path, 'latitude_vs_longitude_vs_posterior_x1'))
    plt.close()


# -------------
# POSTERIOR PREDICTIVE
# -------------

# f = open(os.path.join('images', path, 'posterior.csv'), "r")
# posterior = pd.read_csv(os.path.join('images', path, 'posterior.csv'))
# w_p = posterior.w
x_p = posterior.x0
# sigma_p = posterior.sigma[0]
# c_p = posterior.c[0]
# t_p = posterior.t[0]

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
tot = 50
for i in range(tot):
    Gsim = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x,
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



