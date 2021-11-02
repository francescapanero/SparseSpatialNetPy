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


# # -----------------------------
# # RAIL UK https://bitbucket.org/deregtr/gb_ptn/src/master/
# # -----------------------------

G = nx.Graph()

with open('data/railUK/coord.txt', 'r') as file:
    text = file.readlines()

d = {}
for line in text[0:-1]:
    l = line.split(' ')
    G.add_node(l[0])
    l1 = []
    for i in range(1, len(l)):
        if len(l[i]) != 0:
            l1.append(l[i])
    d[l[0]] = {'east': float(l1[0]), 'north': float(l1[1])}

nx.set_node_attributes(G, d)

with open('data/railUK/edges.txt', 'r') as file:
    text_e = file.readlines()

for line in text_e[0:-1]:
    l = line.split(' ')
    l1 = []
    for i in range(0, len(l)):
        if len(l[i]) != 0:
            l1.append(l[i])
    G.add_edge(l1[0], l1[1])

# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')

G = nx.relabel.convert_node_labels_to_integers(G)

# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = np.sqrt((G.nodes[i]['east'] - G.nodes[j]['east'])**2 +
#                                  (G.nodes[i]['north'] - G.nodes[j]['north'])**2) / 1000
# dist = dist / np.max(dist)
# dist = dist[dist != 0]
# plt.figure()
# plt.hist(dist, bins=50)

# size_x = 1
# prior = 'singlepl'
# gamma = 1
# c = 1
# sigma = 0.05
# t = 100
# tau = 10
# K = 100  # number of layers, for layers sampler
# T = 0.000001
# a_t = 200
# b_t = 1
# approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
# sampler = 'naive'  # can be 'layers' or 'naive'
# type_prior_x = 'uniform'
# dim_x = 2
# Gsim = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x,
#                     a_t, b_t, T=T, K=K, L=2575+200)
# deg = np.array(list(dict(Gsim.degree()).values()))
# x = np.array([Gsim.nodes[i]['x'] for i in range(Gsim.number_of_nodes())])
# dist_sim = np.zeros((Gsim.number_of_nodes(), Gsim.number_of_nodes()))
# for i in range(Gsim.number_of_nodes()):
#     for j in [n for n in Gsim.neighbors(i)]:
#         if j >= i:
#             dist_sim[i, j] = np.sqrt((x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2)  # np.abs(x[i] - x[j])
# plt.figure()
# plt.hist(dist_sim[dist_sim != 0], bins=50)

deg = np.array(list(dict(G.degree()).values()))
set_nodes = np.concatenate((np.argsort(deg)[len(deg)-5: len(deg)], np.argsort(deg)[0: 2]))

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
init[0]['sigma'] = 0.8  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c'] = 1
init[0]['t'] = np.sqrt(G.number_of_edges())
size_x = 1
dim_x = 1
init[0]['size_x'] = size_x
init[0]['x'] = size_x * np.random.uniform(0, 1, L)
# init[0]['x'] = size_x * np.random.uniform(0, 1, (L, dim_x))
gamma = 0.2
G.graph['prior'] = 'singlepl'
G.graph['gamma'] = gamma
G.graph['size_x'] = 1

iter = 5000
save_every = 100
nburn = int(iter * 0.25)
path = 'rail'
out = chain.mcmc_chains([G], iter, nburn, index,
                        sigma=True, c=True, t=True, tau=False, w0=True, n=True, u=True, x=True, beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.1,
                        save_every=save_every, plot=True,  path=path,
                        save_out=False, save_data=False, init=init, a_t=200)

dist = np.zeros((len(set_nodes), len(set_nodes)))
for i in range(len(set_nodes)):
    for j in range(i+1, len(set_nodes)):
        dist[i, j] = np.sqrt((G.nodes[i]['east'] - G.nodes[j]['east'])**2 + (G.nodes[i]['north'] - G.nodes[j]['north'])**2) / 1000

dist_est = np.zeros((len(set_nodes), len(set_nodes), len(out[0][11])))
i = 0
for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        for j in range(len(out[i][12])):
            dist_est[m, n, j] = np.sqrt((out[i][12][j][m][0] - out[i][12][j][n][0])**2 +
                                        (out[i][12][j][m][1] - out[i][12][j][n][1])**2) / 1000

for m in range(len(set_nodes)):
    for n in range(m + 1, len(set_nodes)):
        plt.figure()
        plt.plot(dist_est[m, n, :])
        plt.axhline(dist[m, n])
        plt.title('km distance b/w nodes %i, %i' % (set_nodes[m], set_nodes[n]))
        plt.savefig(os.path.join('images', path, 'distance_nodes_%i_%i' % (set_nodes[m], set_nodes[n])))
        plt.close()

x_mean0 = np.zeros(G.number_of_nodes()-nodes_added)
x_mean1 = np.zeros(G.number_of_nodes()-nodes_added)
east = np.zeros(G.number_of_nodes()-nodes_added)
north = np.zeros(G.number_of_nodes()-nodes_added)
for m in range(G.number_of_nodes()-nodes_added):
    x_mean0[m] = np.mean([out[0][12][j][m][0] for j in range(int(nburn/save_every), int(iter/save_every))])
    x_mean1[m] = np.mean([out[0][12][j][m][1] for j in range(int(nburn / save_every), int(iter / save_every))])
    east[m] = G.nodes[m]['east']
    north[m] = G.nodes[m]['north']
plt.figure()
plt.scatter(east, x_mean0)
plt.savefig(os.path.join('images', path, 'east_vs_x0'))
plt.figure()
plt.scatter(east, x_mean1)
plt.savefig(os.path.join('images', path, 'east_vs_x1'))
plt.figure()
plt.scatter(north, x_mean0)
plt.savefig(os.path.join('images', path, 'north_vx_x0'))
plt.figure()
plt.scatter(north, x_mean1)
plt.savefig(os.path.join('images', path, 'north_vx_x1'))



# # # -----------------------------
# # # COLLABORATIONS - no power law https://github.com/dsaffo/GeoSocialVis/tree/master/data
# # # -----------------------------
#
# with open('data/GeoSocialVis-master/authorGraph.json') as f:
#     json_data = json.loads(f.read())
#
# G = nx.Graph()
# G.add_nodes_from(elem['id'] for elem in json_data['nodes'])
# G.add_edges_from((elem['source'], elem['target']) for elem in json_data['links'])
#
# # import spatial location of universities
# with open('data/GeoSocialVis-master/affiliationList.js') as dataFile:
#     data = dataFile.read()
#     obj = data[data.find('['): data.rfind(']')+1]
#     jsonObj = json.loads(obj)
# c = [elem['Name'] for elem in jsonObj]
# c1 = [elem['Position'] for elem in jsonObj]
# long = {}
# lat = {}
# for i in range(len(c1)):
#     if len(c1[i]) != 0:
#         lat[c[i]] = float(c1[i].split(sep=',')[0])
#         long[c[i]] = float(c1[i].split(sep=',')[1])
#
# # create dictionary of attributes for each node
# a = [elem['id'] for elem in json_data['nodes']]
# b = [elem['affiliation'] for elem in json_data['nodes']]
# d = {}
# count = 0
# for i in range(len(a)):
#     if b[i] in long.keys():
#         d[a[i]] = {'affiliation': b[i], 'long': long[b[i]], 'lat': lat[b[i]]}
#     else:
#         count += 1
#         G.remove_node(a[i])
# nx.set_node_attributes(G, d)
#
# for i in [node for node, degree in G.degree() if degree == 0]:
#     G.remove_node(i)
#
# # sadly not a powel law
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')
#
# G = nx.relabel.convert_node_labels_to_integers(G)
# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# p_ij = np.zeros((l, l))
# lat = np.zeros(l)
# long = np.zeros(l)
# for i in range(l):
#     lat[i] = G.nodes[i]['lat'] * math.pi / 180
#     long[i] = G.nodes[i]['long'] * math.pi / 180
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = 1.609344 * 3963.0 * np.arccos((np.sin(lat[i]) * np.sin(lat[j])) + np.cos(lat[i]) * np.cos(lat[j])
#                                                        * np.cos(long[j] - long[i]))
# dist = dist[dist != 0]
# plt.figure()
# plt.hist(dist, bins=50)
#
# # # -----------------------------
# # # SAN FRANCISCO ROADS https://www.cs.utah.edu/~lifeifei/SpatialDataset.htm
# # # -----------------------------
#
# G = nx.Graph()
#
# with open('data/san_francisco/nodes_location.txt', 'r') as file:
#     text = file.read()
#
# d = {}
# for line in text:
#     l = line.split(sep=' ')
#     G.add_node[l[0]]
#     d[l[0]] = {'long': float(l[1]), 'lat': float(l[2])}
#
# nx.set_node_attributes(G, d)
#
# with open('data/san_francisco/edges.rtf', 'r') as file:
#     text_e = file.read()
#
# for line in text:
#     l = line.split(sep=' ')
#     G.add_edges(l[1], l[2])
#
#
# # # -----------------------------
# # # COACH UK https://bitbucket.org/deregtr/gb_ptn/src/master/
# # # -----------------------------
#
# G = nx.Graph()
#
# with open('data/coach/coord.txt', 'r') as file:
#     text = file.readlines()
#
# d = {}
# for line in text[0:-1]:
#     l = line.split(' ')
#     G.add_node(l[0])
#     l1 = []
#     for i in range(1, len(l)):
#         if len(l[i]) != 0:
#             l1.append(l[i])
#     d[l[0]] = {'east': float(l1[0]), 'north': float(l1[1])}
#
# nx.set_node_attributes(G, d)
#
# with open('data/coach/edges.txt', 'r') as file:
#     text_e = file.readlines()
#
# for line in text_e[0:-1]:
#     l = line.split(' ')
#     l1 = []
#     for i in range(0, len(l)):
#         if len(l[i]) != 0:
#             l1.append(l[i].replace('\\\n', ''))
#     G.add_edge(l1[0], l1[1])
#
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')
#
# G = nx.relabel.convert_node_labels_to_integers(G)
#
# l = G.number_of_nodes()
# dist = np.zeros((l, l))
# for i in range(l):
#     for j in [n for n in G.neighbors(i)]:
#         if j > i:
#             dist[i, j] = np.sqrt((G.nodes[i]['east'] - G.nodes[j]['east'])**2 +
#                                  (G.nodes[i]['north'] - G.nodes[j]['north'])**2) / 1000
# dist = dist[dist != 0]
# plt.figure()
# plt.hist(dist, bins=50)
#
#
# # # -----------------------------
# # # Airports 2019
# # # -----------------------------
# #
# # df = pd.read_csv('data/airports/airports19.csv')
# # Graphtype = nx.DiGraph()
# # G = nx.from_pandas_edgelist(df, edge_attr=['PASSENGERS', 'DISTANCE', 'UNIQUE_CARRIER_NAME', 'YEAR', 'ORIGIN', 'DEST'],
# #                              create_using=Graphtype)
# # l = [(a, b) for a, b, attrs in G.edges(data=True) if attrs["PASSENGERS"] == 0]
# # G.remove_edges_from(l)
# #
# # d = {k: [] for k in G.nodes}
# # for node1, node2, data in G.edges(data=True):
# #         d[node1] = {'id': data['ORIGIN']}
# #         d[node2] = {'id': data['DEST']}
# # for node1, node2, data in G.edges(data=True):
# #     for att in ['YEAR', 'ORIGIN', 'DEST']:
# #         data.pop(att, None)
# # nx.set_node_attributes(G, d)
# # G.to_undirected()
# # G = nx.relabel.convert_node_labels_to_integers(G)
# # G = nx.Graph(G)
# # deg_freq_G = nx.degree_histogram(G)
# # plt.figure()
# # plt.loglog(deg_freq_G, 'go-')
# #
# # deg = np.array(list(dict(G.degree()).values()))
# # biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# # for node in biggest_deg:
# #     print(G.nodes[node]['id'])
# #
# # dist = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
# # for i in range(G.number_of_nodes()):
# #     for j in range(i+1, G.number_of_nodes()):
# #         if j in G.adj[i]:
# #             dist[i][j] = G[i][j]['DISTANCE']
# # dist = dist + np.transpose(dist)
# #
# # # # normalize in [0,1]
# # # dist_pos = dist[dist != 0]
# # # dist_pos = (dist_pos - np.min(dist_pos)) / (np.max(dist_pos) - np.min(dist_pos))
# # # ind_pos = np.where(dist > 0)
# # # dist[ind_pos] = dist_pos
# #
# # # standardize
# # dist_pos = dist[dist > 0]
# # dist_pos = (dist_pos - np.mean(dist_pos)) / np.std(dist_pos)
# # ind_pos = np.where(dist > 0)
# # dist[ind_pos] = dist_pos
# # dist = dist + np.abs(np.min(dist))
# #
# # num = 10
# # deg = np.array(list(dict(G.degree()).values()))
# # size = len(deg)
# # sort_ind = np.argsort(deg)
# # ind_big = sort_ind[range(size - num, size)]
# # p_ij = np.zeros((num, num))
# # p_ij_std = np.zeros((num, num))
# # gamma = 1
# # for i in range(num):
# #     for j in range(i+1, num):
# #         if ind_big[j] in G.adj[ind_big[i]]:
# #             p_ij[i, j] = 1 / ((1 + G[ind_big[i]][ind_big[j]]['DISTANCE']) ** gamma)
# #             p_ij_std[i, j] = 1 / ((1 + dist[ind_big[i]][ind_big[j]]) ** gamma)
# # p_ij = p_ij + np.transpose(p_ij)
# # p_ij_std = p_ij_std + np.transpose(p_ij_std)
# # for i in ind_big:
# #     print(G.nodes[i]['id'])
# #     print(deg[i])
# #
# # pass_CI = [scipy.stats.mstats.mquantiles([G[j][i]['PASSENGERS'] for i in G.adj[j]], prob=[0.025, 0.975]) for j in ind_big]
# # # passenger_CI = pd.DataFrame(pass_CI).transpose()
# # passenger_CI = pd.DataFrame([[G[j][i]['PASSENGERS'] for i in G.adj[j]] for j in ind_big]).transpose()
# # passenger_CI.columns = [G.nodes[j]['id'] for j in ind_big]
# # plt.figure()
# # passenger_CI.boxplot()
# # plt.title('Number of passengers')
# #
# # log_passenger_CI = pd.DataFrame([[np.log(G[j][i]['PASSENGERS']) for i in G.adj[j]] for j in ind_big]).transpose()
# # log_passenger_CI.columns = [G.nodes[j]['id'] for j in ind_big]
# # plt.figure()
# # log_passenger_CI.boxplot()
# # plt.title('Log number of passengers')
# #
# # CI_w = reversed([[2.5, 3.1], [2.5, 3.2], [2.25, 2.8], [2.15, 2.75], [2.8, 3.1], [2.2, 2.8], [2.4, 3.15], [1.8, 2.4],
# #         [1.65, 2.25], [1.4, 1.85]])
# # CI_w = pd.DataFrame(CI_w).transpose()
# # CI_w.columns = [G.nodes[j]['id'] for j in ind_big]
# # plt.figure()
# # CI_w.boxplot()
# # plt.title('CI w')
# #
# # L0 = G.number_of_nodes()
# # L = G.number_of_nodes() + 300
# # G.add_nodes_from(range(L0, L))
# #
# # # n = lil_matrix((L, L))
# # # for i in range(L):
# # #     for j in range(L):
# # #         if j >= i:
# # #             if G.get_edge_data(i, j, default=0) != 0:
# # #                 n[i, j] = G.get_edge_data(i, j, default=0)['weight']
# # # n = csr_matrix(n)
# # # G.graph['counts'] = n
# # # sum_n = np.array(csr_matrix.sum(n, axis=0) + np.transpose(csr_matrix.sum(n, axis=1)))[0]
# # # G.graph['sum_n'] = sum_n
# # # G.graph['sum_fact_n'] = 0
# #
# # G.graph['prior'] = 'singlepl'
# # G.graph['gamma'] = 1
# # G.graph['size_x'] = 1
# #
# # init = {}
# # init[0] = {}
# #
# # init[0]['sigma_init'] = 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
# # init[0]['c_init'] = 1.5
# # init[0]['t_init'] = np.sqrt(G.number_of_edges())
# #
# # init[0]['beta_init'] = np.ones(L)
# #
# # iter = 300000
# # nburn = int(iter * 0.25)
# #
# # out = chain.mcmc_chains([G], iter, nburn,
# #                         sigma=True, c=True, t=True, tau=False,
# #                         w0=True,
# #                         n=True,
# #                         u=True,
# #                         x=True,
# #                         # beta=False,
# #                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
# #                         w_inference='HMC', epsilon=0.01, R=5,
# #                         save_every=1000,
# #                         init=init,
# #                         save_out=False, save_data=False, path='air1', plot=True)
#
#
# # -----------------------------
# # Airports 2010
# # -----------------------------
#
# df = pd.read_csv('data/airports/577903858_T_T100_MARKET_ALL_CARRIER/577903858_T_T100_MARKET_ALL_CARRIER_2010_All.csv')
# df.drop(columns=['FREIGHT', 'MAIL', 'DISTANCE', 'UNIQUE_CARRIER',
#                  'AIRLINE_ID', 'UNIQUE_CARRIER_ENTITY', 'REGION',
#                  'CARRIER', 'CARRIER_NAME', 'CARRIER_GROUP', 'CARRIER_GROUP_NEW',
#                  'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID',
#                  'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_FIPS',
#                  'ORIGIN_STATE_NM', 'ORIGIN_COUNTRY', 'ORIGIN_COUNTRY_NAME',
#                  'ORIGIN_WAC', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID',
#                  'DEST_CITY_MARKET_ID', 'DEST_CITY_NAME', 'DEST_STATE_ABR',
#                  'DEST_STATE_FIPS', 'DEST_STATE_NM', 'DEST_COUNTRY', 'DEST_COUNTRY_NAME',
#                  'DEST_WAC', 'YEAR', 'QUARTER', 'MONTH', 'DISTANCE_GROUP', 'CLASS',
#                  'DATA_SOURCE', 'Unnamed: 41'])
# columns_titles = ["ORIGIN", "DEST", "UNIQUE_CARRIER_NAME", "PASSENGERS"]
# df = df.reindex(columns=columns_titles)
# df = df.rename(columns={'ORIGIN': 'source', 'DEST': 'target'})
# Graphtype = nx.DiGraph()
# G = nx.from_pandas_edgelist(df, edge_attr=['PASSENGERS', 'UNIQUE_CARRIER_NAME', 'source', 'target'],
#                             create_using=Graphtype)
# l = [(a, b) for a, b, attrs in G.edges(data=True) if attrs["PASSENGERS"] == 0]
# G.remove_edges_from(l)
# d = {k: [] for k in G.nodes}
# for node1, node2, data in G.edges(data=True):
#         d[node1] = {'id': data['source']}
#         d[node2] = {'id': data['target']}
# nx.set_node_attributes(G, d)
# G.to_undirected()
# G.remove_edges_from(nx.selfloop_edges(G))
# G.remove_nodes_from(list(nx.isolates(G)))
# G = nx.relabel.convert_node_labels_to_integers(G)
#
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')
#
# deg = np.array(list(dict(G.degree()).values()))
# size = len(deg)
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for i in range(10):
#     print(biggest_deg[i], G.nodes[biggest_deg[i]])
# # Miami, Huston, Minneapolis, Denver, Newark, JFK, Los Angeles, Chicago, Washington DC, Atlanta
#
#
# # -----------------------------
# # Airports 2010 - part b
# # -----------------------------
#
# G = nx.read_edgelist('data/airports/airports10.txt', nodetype=int, data=(('weight', float),))
# f = open("data/airports/USairport_2010_codes.txt", "r")
# id_dict = {}
# for line in f:
#     line = re.sub('"', '', line)
#     key, value = line.split()
#     id_dict[int(key)] = value
# nx.set_node_attributes(G, id_dict, name='id')
# G = nx.relabel.convert_node_labels_to_integers(G)
#
# deg = np.array(list(dict(G.degree()).values()))
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for i in range(10):
#     print(biggest_deg[i], G.nodes[biggest_deg[i]]['id'])
# # Miami, Huston, Minneapolis, Newark, Denver, JFK, LA, Chicago, Washington, Atlanta
#
# deg_freq_G = nx.degree_histogram(G)
# # plt.figure()
# # plt.loglog(deg_freq_G, 'go-')
#
# # df = pd.read_csv('data/airports/253021595_T_MASTER_CORD/253021595_T_MASTER_CORD_All_All.csv')
# # df.drop(columns=['DISPLAY_AIRPORT_NAME',
# #        'DISPLAY_AIRPORT_CITY_NAME_FULL', 'AIRPORT_WAC_SEQ_ID2', 'AIRPORT_WAC',
# #        'AIRPORT_COUNTRY_NAME', 'AIRPORT_COUNTRY_CODE_ISO',
# #        'AIRPORT_STATE_NAME', 'AIRPORT_STATE_CODE', 'AIRPORT_STATE_FIPS',
# #        'CITY_MARKET_SEQ_ID', 'CITY_MARKET_ID', 'DISPLAY_CITY_MARKET_NAME_FULL',
# #        'CITY_MARKET_WAC_SEQ_ID2', 'CITY_MARKET_WAC', 'LAT_DEGREES',
# #        'LAT_HEMISPHERE', 'LAT_MINUTES', 'LAT_SECONDS', 'LATITUDE',
# #        'LON_DEGREES', 'LON_HEMISPHERE', 'LON_MINUTES', 'LON_SECONDS',
# #        'LONGITUDE', 'UTC_LOCAL_TIME_VARIATION', 'AIRPORT_START_DATE',
# #        'AIRPORT_THRU_DATE', 'AIRPORT_IS_CLOSED', 'AIRPORT_IS_LATEST',
# #        'Unnamed: 32'])
# # a = []
# # for node in G.nodes():
# #     a.append(df.index[df['AIRPORT'] == G.nodes[node]['id']].tolist())
# #     G.nodes[node]['seq_id'] = df['AIRPORT_SEQ_ID'][a]
# # for node in G.nodes():
# #     print(int(G.nodes[node]['seq_id']))
#
# size = len(deg)
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for i in range(10):
#     print(G.nodes[biggest_deg[i]])
# # Miami, Huston, Minneapolis, Denver, Newark, JFK, Los Angeles, Chicago, Washington DC, Atlanta
#
# L0 = G.number_of_nodes()
# L = G.number_of_nodes() + 300
# G.add_nodes_from(range(L0, L))
#
# G.graph['prior'] = 'singlepl'
# G.graph['gamma'] = 0
# G.graph['size_x'] = 1
#
# init = {}
# init[0] = {}
# init[0]['sigma_init'] = 0.1  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
# init[0]['c_init'] = 2
# init[0]['t_init'] = np.sqrt(G.number_of_edges())
# init[0]['beta_init'] = np.ones(L)
# init[1] = {}
# init[1]['sigma_init'] = 0.2  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
# init[1]['c_init'] = 2
# init[1]['t_init'] = np.sqrt(G.number_of_edges())
# init[1]['beta_init'] = np.ones(L)
# init[2] = {}
# init[2]['sigma_init'] = 0.15  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
# init[2]['c_init'] = 1.2
# init[2]['t_init'] = np.sqrt(G.number_of_edges())
# init[2]['beta_init'] = np.ones(L)
#
# iter = 500000
# nburn = int(iter * 0.25)
#
# out = chain.mcmc_chains([G, G, G], iter, nburn,
#                         sigma=True, c=True, t=True, tau=False,
#                         w0=True,
#                         n=True,
#                         u=True,
#                         x=False,
#                         beta=False,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         w_inference='gibbs', epsilon=0.01, R=5,
#                         save_every=1000,
#                         init=init,
#                         save_out=False, save_data=False, path='air2010_no_space_3chains_gibbs', plot=True)
#
# # attrib = open("data/airports/attributes_nocolnames.txt")
# #
# # for line in attrib:
# #     words = line.split()
# #     node = int(words[0])
# #     attributes = words[1]
# #     splittedAttributes = attributes.split(' ')
# #     G.node[node]['Attributes'] = splittedAttributes
# #
# # for row in attrib:
# #     attributes = row[1].split(' ')
# #     G.add_node(row[0], attr=attributes)
