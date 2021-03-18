import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import mcmc_chains as chain
import time
from itertools import compress
import csv
import pandas as pd
import networkx as nx
import scipy
import re


# # -----------------------------
# # Airports 2019
# # -----------------------------
#
# df = pd.read_csv('data/airports/airports19.csv')
# Graphtype = nx.DiGraph()
# G = nx.from_pandas_edgelist(df, edge_attr=['PASSENGERS', 'DISTANCE', 'UNIQUE_CARRIER_NAME', 'YEAR', 'ORIGIN', 'DEST'],
#                              create_using=Graphtype)
# l = [(a, b) for a, b, attrs in G.edges(data=True) if attrs["PASSENGERS"] == 0]
# G.remove_edges_from(l)
#
# d = {k: [] for k in G.nodes}
# for node1, node2, data in G.edges(data=True):
#         d[node1] = {'id': data['ORIGIN']}
#         d[node2] = {'id': data['DEST']}
# for node1, node2, data in G.edges(data=True):
#     for att in ['YEAR', 'ORIGIN', 'DEST']:
#         data.pop(att, None)
# nx.set_node_attributes(G, d)
# G.to_undirected()
# G = nx.relabel.convert_node_labels_to_integers(G)
# G = nx.Graph(G)
# deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')
#
# deg = np.array(list(dict(G.degree()).values()))
# biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
# for node in biggest_deg:
#     print(G.nodes[node]['id'])
#
# dist = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
# for i in range(G.number_of_nodes()):
#     for j in range(i+1, G.number_of_nodes()):
#         if j in G.adj[i]:
#             dist[i][j] = G[i][j]['DISTANCE']
# dist = dist + np.transpose(dist)
#
# # # normalize in [0,1]
# # dist_pos = dist[dist != 0]
# # dist_pos = (dist_pos - np.min(dist_pos)) / (np.max(dist_pos) - np.min(dist_pos))
# # ind_pos = np.where(dist > 0)
# # dist[ind_pos] = dist_pos
#
# # standardize
# dist_pos = dist[dist > 0]
# dist_pos = (dist_pos - np.mean(dist_pos)) / np.std(dist_pos)
# ind_pos = np.where(dist > 0)
# dist[ind_pos] = dist_pos
# dist = dist + np.abs(np.min(dist))
#
# num = 10
# deg = np.array(list(dict(G.degree()).values()))
# size = len(deg)
# sort_ind = np.argsort(deg)
# ind_big = sort_ind[range(size - num, size)]
# p_ij = np.zeros((num, num))
# p_ij_std = np.zeros((num, num))
# gamma = 1
# for i in range(num):
#     for j in range(i+1, num):
#         if ind_big[j] in G.adj[ind_big[i]]:
#             p_ij[i, j] = 1 / ((1 + G[ind_big[i]][ind_big[j]]['DISTANCE']) ** gamma)
#             p_ij_std[i, j] = 1 / ((1 + dist[ind_big[i]][ind_big[j]]) ** gamma)
# p_ij = p_ij + np.transpose(p_ij)
# p_ij_std = p_ij_std + np.transpose(p_ij_std)
# for i in ind_big:
#     print(G.nodes[i]['id'])
#     print(deg[i])
#
# pass_CI = [scipy.stats.mstats.mquantiles([G[j][i]['PASSENGERS'] for i in G.adj[j]], prob=[0.025, 0.975]) for j in ind_big]
# # passenger_CI = pd.DataFrame(pass_CI).transpose()
# passenger_CI = pd.DataFrame([[G[j][i]['PASSENGERS'] for i in G.adj[j]] for j in ind_big]).transpose()
# passenger_CI.columns = [G.nodes[j]['id'] for j in ind_big]
# plt.figure()
# passenger_CI.boxplot()
# plt.title('Number of passengers')
#
# log_passenger_CI = pd.DataFrame([[np.log(G[j][i]['PASSENGERS']) for i in G.adj[j]] for j in ind_big]).transpose()
# log_passenger_CI.columns = [G.nodes[j]['id'] for j in ind_big]
# plt.figure()
# log_passenger_CI.boxplot()
# plt.title('Log number of passengers')
#
# CI_w = reversed([[2.5, 3.1], [2.5, 3.2], [2.25, 2.8], [2.15, 2.75], [2.8, 3.1], [2.2, 2.8], [2.4, 3.15], [1.8, 2.4],
#         [1.65, 2.25], [1.4, 1.85]])
# CI_w = pd.DataFrame(CI_w).transpose()
# CI_w.columns = [G.nodes[j]['id'] for j in ind_big]
# plt.figure()
# CI_w.boxplot()
# plt.title('CI w')
#
# L0 = G.number_of_nodes()
# L = G.number_of_nodes() + 300
# G.add_nodes_from(range(L0, L))
#
# # n = lil_matrix((L, L))
# # for i in range(L):
# #     for j in range(L):
# #         if j >= i:
# #             if G.get_edge_data(i, j, default=0) != 0:
# #                 n[i, j] = G.get_edge_data(i, j, default=0)['weight']
# # n = csr_matrix(n)
# # G.graph['counts'] = n
# # sum_n = np.array(csr_matrix.sum(n, axis=0) + np.transpose(csr_matrix.sum(n, axis=1)))[0]
# # G.graph['sum_n'] = sum_n
# # G.graph['sum_fact_n'] = 0
#
# G.graph['prior'] = 'singlepl'
# G.graph['gamma'] = 1
# G.graph['size_x'] = 1
#
# init = {}
# init[0] = {}
#
# init[0]['sigma_init'] = 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
# init[0]['c_init'] = 1.5
# init[0]['t_init'] = np.sqrt(G.number_of_edges())
#
# init[0]['beta_init'] = np.ones(L)
#
# iter = 300000
# nburn = int(iter * 0.25)
#
# out = chain.mcmc_chains([G], iter, nburn,
#                         sigma=True, c=True, t=True, tau=False,
#                         w0=True,
#                         n=True,
#                         u=True,
#                         x=True,
#                         # beta=False,
#                         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
#                         w_inference='HMC', epsilon=0.01, R=5,
#                         save_every=1000,
#                         init=init,
#                         save_out=False, save_data=False, path='air1', plot=True)


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
#     print(G.nodes[biggest_deg[i]])
# # Miami, Huston, Minneapolis, Denver, Newark, JFK, Los Angeles, Chicago, Washington DC, Atlanta


# -----------------------------
# Airports 2010 - part b
# -----------------------------

G = nx.read_edgelist('data/airports/airports10.txt', nodetype=int, data=(('weight', float),))
f = open("data/airports/USairport_2010_codes.txt", "r")
id_dict = {}
for line in f:
    line = re.sub('"', '', line)
    key, value = line.split()
    id_dict[int(key)] = value
nx.set_node_attributes(G, id_dict, name='id')
G = nx.relabel.convert_node_labels_to_integers(G)

deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
for i in range(10):
    print(G.nodes[biggest_deg[i]]['id'])
# Miami, Huston, Minneapolis, Newark, Denver, JFK, LA, Chicago, Washington, Atlanta

deg_freq_G = nx.degree_histogram(G)
# plt.figure()
# plt.loglog(deg_freq_G, 'go-')

# df = pd.read_csv('data/airports/253021595_T_MASTER_CORD/253021595_T_MASTER_CORD_All_All.csv')
# df.drop(columns=['DISPLAY_AIRPORT_NAME',
#        'DISPLAY_AIRPORT_CITY_NAME_FULL', 'AIRPORT_WAC_SEQ_ID2', 'AIRPORT_WAC',
#        'AIRPORT_COUNTRY_NAME', 'AIRPORT_COUNTRY_CODE_ISO',
#        'AIRPORT_STATE_NAME', 'AIRPORT_STATE_CODE', 'AIRPORT_STATE_FIPS',
#        'CITY_MARKET_SEQ_ID', 'CITY_MARKET_ID', 'DISPLAY_CITY_MARKET_NAME_FULL',
#        'CITY_MARKET_WAC_SEQ_ID2', 'CITY_MARKET_WAC', 'LAT_DEGREES',
#        'LAT_HEMISPHERE', 'LAT_MINUTES', 'LAT_SECONDS', 'LATITUDE',
#        'LON_DEGREES', 'LON_HEMISPHERE', 'LON_MINUTES', 'LON_SECONDS',
#        'LONGITUDE', 'UTC_LOCAL_TIME_VARIATION', 'AIRPORT_START_DATE',
#        'AIRPORT_THRU_DATE', 'AIRPORT_IS_CLOSED', 'AIRPORT_IS_LATEST',
#        'Unnamed: 32'])
# a = []
# for node in G.nodes():
#     a.append(df.index[df['AIRPORT'] == G.nodes[node]['id']].tolist())
#     G.nodes[node]['seq_id'] = df['AIRPORT_SEQ_ID'][a]
# for node in G.nodes():
#     print(int(G.nodes[node]['seq_id']))

size = len(deg)
biggest_deg = np.argsort(deg)[len(deg)-10: len(deg)]
for i in range(10):
    print(G.nodes[biggest_deg[i]])
# Miami, Huston, Minneapolis, Denver, Newark, JFK, Los Angeles, Chicago, Washington DC, Atlanta

L0 = G.number_of_nodes()
L = G.number_of_nodes() + 300
G.add_nodes_from(range(L0, L))

G.graph['prior'] = 'singlepl'
G.graph['gamma'] = 0
G.graph['size_x'] = 1

init = {}
init[0] = {}
init[0]['sigma_init'] = 0.1  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[0]['c_init'] = 2
init[0]['t_init'] = np.sqrt(G.number_of_edges())
init[0]['beta_init'] = np.ones(L)
init[1] = {}
init[1]['sigma_init'] = 0.2  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[1]['c_init'] = 2
init[1]['t_init'] = np.sqrt(G.number_of_edges())
init[1]['beta_init'] = np.ones(L)
init[2] = {}
init[2]['sigma_init'] = 0.15  # 2 * np.log(G.number_of_nodes()) / np.log(G.number_of_edges()) - 1
init[2]['c_init'] = 1.2
init[2]['t_init'] = np.sqrt(G.number_of_edges())
init[2]['beta_init'] = np.ones(L)

iter = 500000
nburn = int(iter * 0.25)

out = chain.mcmc_chains([G, G, G], iter, nburn,
                        sigma=True, c=True, t=True, tau=False,
                        w0=True,
                        n=True,
                        u=True,
                        x=False,
                        beta=False,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        w_inference='gibbs', epsilon=0.01, R=5,
                        save_every=1000,
                        init=init,
                        save_out=False, save_data=False, path='air2010_no_space_3chains_gibbs', plot=True)

# attrib = open("data/airports/attributes_nocolnames.txt")
#
# for line in attrib:
#     words = line.split()
#     node = int(words[0])
#     attributes = words[1]
#     splittedAttributes = attributes.split(' ')
#     G.node[node]['Attributes'] = splittedAttributes
#
# for row in attrib:
#     attributes = row[1].split(' ')
#     G.add_node(row[0], attr=attributes)
