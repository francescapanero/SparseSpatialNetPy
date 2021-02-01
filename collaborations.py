import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import utils.MCMCNew_fast as mcmc
from itertools import compress
import csv


net = nx.read_gml('data/netscience/netscience.gml', label='id')
print(net.number_of_nodes())  # 1500
print(net.number_of_edges())  # 2700
deg_net = np.array(list(dict(net.degree()).values()))
print(sum(deg_net == 0))
deg_freq_net = nx.degree_histogram(net)
plt.figure()
plt.loglog(range(1, len(deg_freq_net)), deg_freq_net[1:], 'go-')

phys = nx.read_gml('data/cond-mat-1999/cond-mat.gml', label='label')
print(phys.number_of_nodes())  # 16000
print(phys.number_of_edges())  # 47000
deg_phys = np.array(list(dict(phys.degree()).values()))
print(sum(deg_phys == 0))
deg_freq_phys = nx.degree_histogram(phys)
plt.figure()
plt.loglog(range(1, len(deg_freq_phys)), deg_freq_phys[1:], 'go-')

hep = nx.read_gml('data/hep-th/hep-th.gml', label='label')
print(hep.number_of_nodes())  # 8000
print(hep.number_of_edges())  # 15700
deg_hep = np.array(list(dict(hep.degree()).values()))
print(sum(deg_hep == 0))
deg_freq_hep = nx.degree_histogram(hep)
plt.figure()
plt.loglog(range(1, len(deg_freq_hep)), deg_freq_hep[1:], 'go-')


iter = 10000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'
sigma_x = 0.01
a_t = 200
b_t = 1

gamma = 1
size_x = 1
size = net.number_of_nodes()
prior = 'singlepl'

ind = {k: [] for k in net.nodes}
for i in net.nodes:
    for j in net.adj[i]:
        if j > i:
            ind[i].append(j)
selfedge = [i in ind[i] for i in net.nodes]
selfedge = list(compress(net.nodes, selfedge))

start = time.time()
output = mcmc.MCMC(prior, net, gamma, size, iter, nburn, size_x,
                   w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                   plot=False,
                   sigma=True, c=True, t=True, w0=True, n=True, u=True, x=True,
                   ind=ind, selfedge=selfedge,
                   save_every=1)
end = time.time()
print('minutes to produce the sample (true): ', round((end - start) / 60, 2))


plt.figure()
plt.plot(output[10], color='cornflowerblue')
plt.xlabel('iter')
plt.ylabel('log_post')
plt.savefig('images/net/log_post')
plt.close()

plt.figure()
plt.plot(output[3], color='cornflowerblue')
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/net/sigma')
plt.close()
#
plt.figure()
plt.plot(output[4], color='cornflowerblue')
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/net/c')
plt.close()
#
plt.figure()
plt.plot(output[5], color='cornflowerblue')
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/net/t')
plt.close()


air = nx.read_edgelist('data/airports/airports_us.txt', nodetype=int, data=(('weight', float),))
deg_freq_air = nx.degree_histogram(air)
plt.figure()
plt.loglog(range(1, len(deg_freq_air)), deg_freq_air[1:], 'go-')
attrib = open('data/airports/attributes.txt', 'r')
for row in attrib:
    attributes = row[1].split(' ')

attr = csv.reader(open("data/airports/attributes.txt"), delimiter=' ')
for row in attr:
    attributes = row[1].split(' ')
    air.add_node(row[0], attr=attributes)

for n in air.nodes()[0:10]:
    print('Node: ', str(n))
    print('Atrributes', str(air.node[n]['attr']))