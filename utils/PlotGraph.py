import matplotlib.pyplot as plt
import pandas
import scipy
from utils.GraphSampler import *
import numpy as np
import networkx as nx


# --------------------------
# different plots methods:
# - plt_space_adj: adjacency matrix
# - plt_ccdf: ccdf of degrees
# - plt_rank: ranked frequencies plot (for w or deg)
# - plt_large_deg_nodes: plot large degree nodes vs their asymptotic value for doublepl prior
# - plt_deg_dist: degree distribution, can be binned and with asymptotic value
# - plt_compare_sparsity: compares number of nodes and edges in the asymptotic regime for multiple samples for naive
#                         vs layers methods
# - plt_compare_clustering: compares clustering coefficient in the asymptotic regime for multiple samples for naive
#                           vs layers methods, plotting the theoretical limit
# --------------------------


# G networkx graph
# vector of locations of nodes of G
def plt_space_adj(G, x):
    Z = nx.to_numpy_matrix(G)
    ind1, ind2 = np.nonzero(np.triu(Z, 1))
    plt.figure()
    plt.plot(x[ind1], x[ind2], 'b.', x[ind2], x[ind1], 'b.')


# plot of ccdf of degrees (it could be extended to general vectors I guess
# for prior=='doublepl' it plots the asymptotics  in tau and sigma
def plt_ccdf(deg, sigma='NA', tau='NA', prior='NA'):
    deg = deg[deg>0]
    deg_ = pandas.Series(deg)
    freq = deg_.value_counts()
    freq = dict(freq)
    ind = np.argsort(list(freq.keys()))
    cum_deg = list(reversed(np.cumsum(list(reversed([list(freq.values())[i] for i in ind])))))
    num_nodes = len(deg)
    plt.figure()
    plt.plot([x / num_nodes for x in cum_deg], 'bo')
    if prior == 'doublepl':
        plt.plot(range(1, 1000), [np.exp((-tau) * np.log(i)) for i in range(1, 1000)], 'g-', label='-tau')
        plt.plot(range(1, 1000), [np.exp((-sigma) * np.log(i)) for i in range(1, 1000)], 'r-', label='-sigma')
        plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('j')
    plt.ylabel('P(deg>d)')
    # plt.title('ccdf deg sigma=%f, tau=%i' % (sigma, tau))


# plot of ranked frequencies of a vector x. It is thought for w or degrees.
# the argument deg is helpful if you have zero deg nodes and you want to remove them
def plt_rank(x, **kwargs):
    if 'deg' in kwargs:
        deg = kwargs['deg']
        if len(deg) == len(x):
            x = x[list(deg > 0)]
        else:
            print('dimensions of x and deg did not match')
    sorted_x = np.flip(np.sort(x))
    plt.figure()
    plt.plot(range(len(x)), sorted_x, 'bo')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('rank')
    plt.ylabel('frequency')


def plt_large_deg_nodes(j, nj, exp_nj):
    plt.figure()
    plt.plot(j, nj, 'bo', label='observed')
    plt.plot(j, exp_nj, 'ro', label='expected')
    plt.xscale('log')
    plt.yscale('log')
    # plt.title('Doublepl sigma=%f, tau=%i, size_x=%i, alpha=%i, T=%f, beta=%i' % (sigma, tau, size_x, alpha, T, beta))
    plt.legend()


# plot log log degree distribution
# if prior = 'GGP' then you have to specify sigma as well and you'll get the asymptotics
# can be binned or not
def plt_deg_distr(deg, sigma='NA', prior='NA', binned=True):
    if sum(deg == 0) > 1:
        deg = deg[list(deg > 0)]
    num_nodes = len(deg)
    deg_ = pandas.Series(deg)
    freq = deg_.value_counts()
    freq = dict(freq)
    if binned == True:
        freq = [x / num_nodes for x in list(freq.values())]
        bins = np.exp(np.linspace(np.log(min(freq)), np.log(max(freq)), int((np.log(max(freq))-np.log(min(freq)))*5)))
        sizebins = (bins[1:] - bins[:-1])
        # sizebins = np.append(sizebins, 1)
        counts = np.histogram(freq, bins=bins)
        counts = counts[0]
        freq = counts/sizebins
        plt.figure()
        plt.plot(bins[:-1], freq, 'bo', label='empirical')
        plt.legend()
    else:
        plt.figure()
        plt.plot(list(freq.keys()), [np.exp(np.log(x) - np.log(num_nodes)) for x in list(freq.values())], 'bo')
    if prior == 'GGP':
        if binned == True:
            plt.plot(bins[:-1], [sigma * x ** (- 1 - sigma) / scipy.special.gamma(1 - sigma) for x in bins[:-1]],
                    label='1+sigma')
        else:
            plt.plot(list(freq.keys()),
                     [np.exp(np.log(sigma) + scipy.special.gammaln(j - sigma) - scipy.special.gammaln(j + 1)
                             - scipy.special.gammaln(1 - sigma)) for j in list(freq.keys())], 'r-', label='1+sigma')

        plt.legend()
        # plt.title('Degree distribution in log log scale with asymptotics')
    # else:
        # plt.title('Degree distribution in log log scale')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('deg')
    plt.ylabel('frequency')


# compare number of nodes and edges as alpha grows for samples drawn from layers and naive methods
# nodes and edges must be n x len(alpha) arrays
def plt_compare_sparsity(nodes_n, edges_n, nodes_l, edges_l, alpha):
    plt.figure()
    n = nodes_n.shape[0]
    for i in range(n - 1):
        plt.plot(alpha, nodes_n[i, ], 'bo-', alpha, nodes_l[i, ], 'ko-')
    plt.plot(alpha, nodes_n[n - 1, ], 'bo-', label='naive')
    plt.plot(alpha, nodes_l[n - 1, ], 'ko-', label='layers')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('number of nodes')
    plt.legend()
    plt.figure()
    for i in range(n - 1):
        plt.plot(alpha, edges_n[i, ], 'bo-', alpha, edges_l[i, ], 'ko-')
    plt.plot(alpha, edges_n[n - 1, ], 'bo-', label='naive')
    plt.plot(alpha, edges_l[n - 1, ], 'ko-', label='layers')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('number of edges')
    plt.legend()


# similar as previous function, but with clustering coefficients. Plots also asymptotic limits.
def plt_compare_clustering(global_n, local_n, global_l, local_l, limit_glob, limit_loc, alpha):
    plt.figure()
    n = global_n.shape[0]
    for i in range(n - 1):
        plt.plot(alpha, global_n[i, ], 'r-', alpha, global_l[i, ], 'r-.')
        plt.plot(alpha, local_n[i, ], 'b-', alpha, local_l[i, ], 'b-.')
    plt.plot(alpha, global_n[n - 1, ], 'r-', label='glob N')
    plt.plot(alpha, global_l[n - 1, ], 'r-.', label='glob L')
    plt.plot(alpha, local_n[n - 1, ], 'b-', label='loc N')
    plt.plot(alpha, local_l[n - 1, ], 'b-.', label='loc L')
    plt.hlines(limit_glob, min(alpha), max(alpha), color='r', linestyles='dashed', label='limit glob')
    plt.hlines(limit_loc, min(alpha), max(alpha), color='b', linestyles='dashed', label='limit avg loc')
    plt.xlabel('t')
    plt.ylabel('clustering coefficient')
    plt.legend()


# def plt_loglogweights(prior, alpha, sigma, tau, T):
#     w = WeightsSampler(prior, alpha, sigma, tau, T=T)
#     bins = np.exp(np.linspace(np.log(min(w)), np.log(max(w)), int(np.log(max(w))-np.log(min(w)))))
#     sizebins = (bins[1:] - bins[:-1])
#     # sizebins = np.append(sizebins, 1)
#     counts = np.histogram(w, bins=bins)
#     counts = counts[0]
#     freq = counts/sizebins
#     plt.plot(bins[:-1], freq, 'bo')
#     plt.plot(bins[:-1], [x ** (-1 - sigma) for x in bins[:-1]])
#     plt.xscale('log')
#     plt.yscale('log')


