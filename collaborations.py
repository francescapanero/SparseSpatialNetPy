import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import mcmc_chains as chain
import time
import utils.MCMCNew_fast as mcmc
from itertools import compress
import csv


# net = nx.read_gml('data/netscience/netscience.gml', label='id')
# print(net.number_of_nodes())  # 1500
# print(net.number_of_edges())  # 2700
# deg_net = np.array(list(dict(net.degree()).values()))
# print(sum(deg_net == 0))
# deg_freq_net = nx.degree_histogram(net)
# plt.figure()
# plt.loglog(range(1, len(deg_freq_net)), deg_freq_net[1:], 'go-')
#
# phys = nx.read_gml('data/cond-mat-1999/cond-mat.gml', label='label')
# print(phys.number_of_nodes())  # 16000
# print(phys.number_of_edges())  # 47000
# deg_phys = np.array(list(dict(phys.degree()).values()))
# print(sum(deg_phys == 0))
# deg_freq_phys = nx.degree_histogram(phys)
# plt.figure()
# plt.loglog(range(1, len(deg_freq_phys)), deg_freq_phys[1:], 'go-')
#
# hep = nx.read_gml('data/hep-th/hep-th.gml', label='label')
# print(hep.number_of_nodes())  # 8000
# print(hep.number_of_edges())  # 15700
# deg_hep = np.array(list(dict(hep.degree()).values()))
# print(sum(deg_hep == 0))
# deg_freq_hep = nx.degree_histogram(hep)
# plt.figure()
# plt.loglog(range(1, len(deg_freq_hep)), deg_freq_hep[1:], 'go-')


# air = nx.read_edgelist('data/airports/airports_us.txt', nodetype=int, data=(('weight', float),))
# deg_freq_air = nx.degree_histogram(air)
# plt.figure()
# plt.loglog(range(1, len(deg_freq_air)), deg_freq_air[1:], 'go-')

# attrib = open('data/airports/attributes.txt', 'r')
# for row in attrib:
#     attributes = row[1].split(' ')
#
# attr = csv.reader(open("data/airports/attributes.txt"), delimiter=' ')
# for row in attr:
#     attributes = row[1].split(' ')
#     air.add_node(row[0], attr=attributes)
#
# for n in air.nodes()[0:10]:
#     print('Node: ', str(n))
#     print('Atrributes', str(air.node[n]['attr']))

air = nx.read_edgelist('data/airports/airports_us.txt', nodetype=int, data=(('weight', float),))
G = nx.relabel.convert_node_labels_to_integers(air)

L0 = air.number_of_nodes()
L = air.number_of_nodes() + 300
G.add_nodes_from(range(L0, L))

G.graph['prior'] = 'singlepl'
G.graph['gamma'] = 1
G.graph['size_x'] = 1

# n = lil_matrix((L, L))
# for i in range(L):
#     for j in range(L):
#         if j >= i:
#             if G.get_edge_data(i, j, default=0) != 0:
#                 n[i, j] = G.get_edge_data(i, j, default=0)['weight']
# n = csr_matrix(n)
# G.graph['counts'] = n
# sum_n = np.array(csr_matrix.sum(n, axis=0) + np.transpose(csr_matrix.sum(n, axis=1)))[0]
# G.graph['sum_n'] = sum_n
# G.graph['sum_fact_n'] = 0

init = {}
init[0] = {}

init[0]['sigma_init'] = 0.15  # 2 * np.log(L) / np.log(G.number_of_edges()) - 1
init[0]['c_init'] = 1.5
init[0]['t_init'] = np.sqrt(G.number_of_edges())

init[0]['beta_init'] = np.ones(L)

iter = 1000
nburn = int(iter * 0.25)

out = chain.mcmc_chains([G], iter, nburn,
                        sigma=True, c=True, t=True, tau=False,
                        w0=True,
                        n=True,
                        u=True,
                        x=True,
                        # beta=False,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        w_inference='gibbs', epsilon=0.01, R=5,
                        save_every=250,
                        init=init,
                        save_out=False, save_data=False, path='all_rand16', plot=True,)

