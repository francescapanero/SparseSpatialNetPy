from scipy.sparse import lil_matrix
import networkx as nx
import utils.Weights as weight
import utils.LocationsSampler as loc
import time
import numpy as np
from itertools import compress
import utils.Updates as up
import utils.Auxiliary as aux
import utils.TruncPois as tp
from scipy.sparse import csr_matrix
import scipy

# --------------------------
# 3 functions:
# GraphSampler: general function to sample a graph calling the various methods
# NaiveSampler: sample in naive way
# SamplerLayers: sample graph according to "layers" method: divide space in cells and weights in layers
# --------------------------

# Function to sample the graph. You need to specify the type of prior, of approximation and of sampler:
# prior:
# - "singlepl"
# - "doublepl":
# approximation:
# - "finite": **kwargs: L sample size
# - "truncated": **kwargs: T truncation level
# typesampler:
# - "layers": **kwargs: K grid size
# - "naive"


def GraphSampler(prior, approximation, typesampler, sigma, c, t, tau, gamma, size_x, type_prior_x, dim_x,
                 a_t=200, b_t=1, **kwargs):

    start = time.time()
    # sample weights w, w0, beta
    output = weight.WeightsSampler(prior, approximation, t, sigma, c, tau, **kwargs)
    w = kwargs['w'] if 'w' in kwargs else output[0]
    w0 = kwargs['w0'] if 'w0' in kwargs else output[1]
    beta = kwargs['beta'] if 'beta' in kwargs else output[2]
    size = len(w)
    # sample locations
    x = kwargs['x'] if 'x' in kwargs else loc.LocationsSampler(size_x, size, type_prior_x, dim_x)
    # sample graph
    if typesampler == "naive":
        [G, w, x, size] = NaiveSampler(w, x, gamma, dim_x)
    if typesampler == "layers":
        K = kwargs['K'] if 'K' in kwargs else 100
        [G, w, x, size] = SamplerLayers_optim(w, x, gamma, size_x, K)
    end = time.time()

    print('time to produce sample: ', round((end - start) / 60, 2), ' min')

    deg = np.array(list(dict(G.degree()).values()))
    print('number of active nodes: ', sum(deg > 0))
    print('total number of nodes L: ', len(deg))

    G.graph['prior'] = prior
    G.graph['sigma'] = sigma
    G.graph['c'] = c
    G.graph['t'] = t
    G.graph['tau'] = tau
    G.graph['gamma'] = gamma
    G.graph['size_x'] = size_x
    G.graph['a_t'] = a_t
    G.graph['b_t'] = b_t

    # set nodes attributes: w, w0, beta, x, u
    z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
        (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
    G.graph['z'] = z
    u = tp.tpoissrnd(z * w0)
    d = {k: [] for k in G.nodes}
    for i in G.nodes():
        d[i] = {'w': w[i], 'w0': w0[i], 'beta': beta[i], 'x': x[i], 'u': u[i]}
    nx.set_node_attributes(G, d)

    # set graph attributes: ind (upper triangular matrix of neighbors of nodes) and selfedge (list of nodes w/ selfedge)
    ind = {k: [] for k in G.nodes}
    for i in G.nodes:
        for j in G.adj[i]:
            if j >= i:
                ind[i].append(j)
    selfedge = [i in ind[i] for i in G.nodes]
    selfedge = list(compress(G.nodes, selfedge))
    G.graph['ind'] = ind
    G.graph['selfedge'] = selfedge

    # computing "distance" matrix p_ij = 1 / ((1 + |x_i-x_j|) ** gamma)
    p_ij = aux.space_distance(x, gamma) if gamma != 0 else np.ones((size, size))
    G.graph['distances'] = p_ij

    # computing counts upper triangular matrix n
    n_out = up.update_n(w, G, size, p_ij, ind, selfedge)
    n = n_out[0]
    G.graph['counts'] = n  # for the counts, it would be nice to set up a nx.MultiGraph, but some algorithms don't work
    #  on these graphs, so for the moment I'll assign n as attribute to the whole graph rather then the single nodes
    sum_n = np.array(csr_matrix.sum(n, axis=0) + np.transpose(csr_matrix.sum(n, axis=1)))[0]
    G.graph['sum_n'] = sum_n
    sum_fact_n = n_out[1]
    G.graph['sum_fact_n'] = sum_fact_n

    #  attach log posterior of the graph as attribute
    adj = n > 0

    # ### SPEED UP - when updating x alone
    # ind = np.argsort(deg)
    # index = ind[0:len(ind) - 1]
    # log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n,
    #                                         adj, x, index=index)
    # ### SPEED UP - when updating x alone
    log_post_param = aux.log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t)
    log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n,
                                                                                 adj, x)
    G.graph['log_post'] = log_post
    G.graph['log_post_param'] = log_post_param

    return G


def NaiveSampler(w, x, gamma, dim_x):

    n = len(w)  # Number of potential nodes

    # Construct matrix with pairwise distances XY_utr and pairwise products of weights XWw_utr
    X,Y = np.meshgrid(x,x)
    XY = np.absolute(X-Y)
    ind_XY = np.triu_indices(n)
    # if dim_x == 1:
    #     XY_utr = XY[ind_XY] # upper triangular matrix of pairwise distances
    # # Same but with locations in R^2
    # if dim_x == 2:
    #     dist = np.zeros((n, n))
    #     for i in range(n):
    #        for j in range(i+1, n):
    #            dist[i,j] = np.sqrt((X[2*i, 2*j]-Y[2*i, 2*j]) ** 2 + (X[2*i+1, 2*j+1]-Y[2*i+1, 2*j+1]) ** 2)
    #     XY_utr = dist[ind_XY]
    x_ = x[:, None] if dim_x == 1 else x
    temp = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x_, 'euclidean'))
    np.fill_diagonal(temp, 0)
    XY_utr = temp[ind_XY]

    Xw, Yw = np.meshgrid(w, w)
    XYw = Xw*Yw
    XYw_utr = XYw[ind_XY]  # upper triangular matrix of pairwise product of weights

    # Sample adjacency matrix
    prob = 1 - np.exp(-2*XYw_utr/((1+XY_utr)**gamma))  # probability vector of dimension n(n-1)/2+n
    Z = np.random.rand(len(prob)) < prob  # binary adjacency vector
    Z1 = np.zeros((n, n))
    Z1[ind_XY] = Z
    Z1 = Z1 + np.transpose(Z1)  # adjacency matrix, including non connected nodes
    a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
    Z1[np.diag_indices(n)] = a

    Z1 = lil_matrix(Z1)
    G = nx.from_scipy_sparse_matrix(Z1)
    num_nodes = G.number_of_nodes()

    return [G, w, x, num_nodes]


# implementation with sparse matrix (lil matrix)

def SamplerLayers_optim(w, x, gamma, size_x, K):

    if sum(x) == 0:
        print('You cannot use the Layers method without locations!')

    n = len(w)
    G = lil_matrix((n,n))  # set adjacency matrix to 0. lil is a sparse matrix

    # define K cells in [0,size_x]
    _, bin = np.histogram(x, K, range=(0, size_x))
    loc = np.array(np.digitize(x, bin))  # index of the cell to which nodes belong
    delta = size_x/K

    # define weight layers
    lay = weight.WeightLayers(w)
    layer = lay[1]
    w_lay = lay[0]

    if K > 2:
        # construct pmf p_K (each of the K-2 rows corresponds to a k) that represents the pobabilities of connection
        # with cells k+2...K
        # Z[k] is the normalizing constant of p_K[k,:]
        Z = np.zeros(K - 2)
        p_K = np.zeros((K-2, K))
        for k in range(K-2):  # O(K)
            p_K[k, (k+2):K] = np.array(1 / ((1 - delta + delta * np.array(range(2, K - k))) ** gamma))
            Z[k] = p_K[k, ].sum()
            p_K[k, ] = p_K[k, ] / Z[k]

        # construct auxiliary quantities:
        # V array dim J with cardinality of each layer
        # w_jk list of lists of dim J x K to store in position j,k the list of w belonging to cell k and layer j
        # ind_ij is the same as w_jk but with indices of weights
        # W array dim J x K: in pos j,k there's the sum of w_jk[j][k]
        J = max(layer)
        V = np.zeros(J)
        w_jk = [[[] for k in range(K)] for j in range(J)]
        ind_jk = [[[] for k in range(K)] for j in range(J)]
        W = np.zeros((J, K))
        for j in range(J):
            V[j] = sum((layer == (j+1)))  # number of elements in layer j
            for k in range(K):
                ind_jk[j][k] = np.where((loc == (k+1)) & (layer == (j+1)))[0]  # list of indexes nodes in cell k layer j
                w_jk[j][k] = w[ind_jk[j][k]]  # in every element there are weights in cell k and layer j
                W[j][k] = sum(w_jk[j][k])  # in every element there is sum of weights in cell k and layer j

        # sample the edges from mu^(2) (notation paper): Poisson measure from cell k=1..K-2 to cell l=k+2..K
        l_ = [[[] for j in range(J)] for i in range(J)]  # list of lists of dim J x J. List [j1][j2] will be a
        # K-2 long list with, in element k, the index of possible cells connected with k
        for iter in range(2):
            for j1 in range(J):
                for j2 in range(J):
                    # sample the cells connected to every k, for this configuration of i,j (m^{j1,j2} in paper)
                    m_j1j2 = np.random.poisson(W[j1, 0:(K - 2)] * w_lay[j2 + 1] * V[j2] * Z, K - 2)
                    for k in range(K-2):
                        l_[j1][j2].append(0)  # this helps if there is a m_j1j2[k] = 0, o/w the for would be stuck
                        if m_j1j2[k] != 0:
                            l_[j1][j2][k] = np.random.choice(np.array(range(k + 2, K)), m_j1j2[k], p=p_K[k, (k + 2):K])
                            # accept classes l[j1][j2][k] with a certain thinning probability
                            index_temp = np.random.rand(m_j1j2[k]) < (W[j2][l_[j1][j2][k]] / (w_lay[j2 + 1] * V[j2]))
                            l_[j1][j2][k] = l_[j1][j2][k][index_temp]
                            l_temp = l_[j1][j2][k]  # classes accepted for connections
                            if len(l_temp) > 0:
                                for m in range(len(l_temp)):
                                    l = int(l_temp[m])

                                    if len(ind_jk[j1][k]) > 0 and len(ind_jk[j2][l]) > 0: # you need at least one node in
                                        # the intersection of cell k and layer j1 (and l and j2)
                                        w_j1j2kl = np.outer(w_jk[j1][k], w_jk[j2][l]) / (W[j1][k] * W[j2][l])
                                        w_j1j2kl_flat = np.ndarray.flatten(np.array(w_j1j2kl))
                                        # possible pairs of nodes
                                        pair_ = [[(ind_jk[j1][k][n], ind_jk[j2][l][o]) for n in
                                                 range(len(ind_jk[j1][k]))] for o in range(len(ind_jk[j2][l]))]
                                        pair = [item for sublist in pair_ for item in sublist]
                                        a = int(np.random.choice(range(len(pair)), 1, p=w_j1j2kl_flat))
                                        ind_x = pair[a][0]  # chosen pair
                                        ind_y = pair[a][1]
                                        x_j1k = x[ind_x]
                                        x_j2l = x[ind_y]
                                        prob = ((1 - delta + delta*np.absolute(k-l)) ** gamma) / \
                                               ((1 + np.absolute(x_j1k - x_j2l)) ** gamma)
                                        accept_edge = np.random.rand(1) < prob
                                        if accept_edge == 1:
                                            #print(prob * p_K[k, l] * w_j1j2kl_flat[a] * (W[j2][l] / (w_lay[j2 + 1] * V[j2])) * W[j1, k] * w_lay[j2 + 1] * V[j2] * Z[k])
                                            #print(w[ind_x] * w[ind_y] / ((1 + np.absolute(x_j1k - x_j2l)) ** gamma))
                                            G[ind_x, ind_y] = 1
                                            G[ind_y, ind_x] = 1

    # sample connections from cell k=1..k to cell k or k+1
    for k in range(K):
        ind_k = np.where(loc == (k+1))[0]
        w_k_temp = w[ind_k]
        w_kk = np.outer(w_k_temp, w_k_temp)
        X_k, X_kk = np.meshgrid(x[ind_k], x[ind_k])
        diff_X_kk = np.absolute(X_k - X_kk)
        param_kk = w_kk / ((1 + diff_X_kk) ** gamma)
        n_kk = np.random.poisson(param_kk)
        accept_edge_k = (n_kk > 0)
        ind_ = np.where(accept_edge_k == 1)
        for i in range(len(ind_[0])):
            if ind_k[ind_[0][i]] != ind_k[ind_[1][i]]:
                G[ind_k[ind_[0][i]], ind_k[ind_[1][i]]] = 1
                G[ind_k[ind_[1][i]], ind_k[ind_[0][i]]] = 1
        if k != (K-1):
            ind_k1 = np.where(loc == (k + 2))[0]
            w_k1_temp = w[ind_k1]
            w_kk1 = np.outer(w_k_temp, w_k1_temp)
            X_k, X_k1 = np.meshgrid(x[ind_k1], x[ind_k])
            diff_X_kk1 = np.absolute(X_k-X_k1)
            param_kk1 = w_kk1 / ((1 + diff_X_kk1) ** gamma)
            n_kk1 = np.random.poisson(param_kk1)
            accept_edge_k1 = (n_kk1 > 0)
            ind1_ = np.where(accept_edge_k1 == 1)
            for i in range(len(ind1_[0])):
                G[ind_k[ind1_[0][i]], ind_k1[ind1_[1][i]]] = 1
                G[ind_k1[ind1_[1][i]], ind_k[ind1_[0][i]]] = 1
        if k != 0:
            ind_k1 = np.where(loc == k)[0]
            w_k1_temp = w[ind_k1]
            w_kk1 = np.outer(w_k_temp, w_k1_temp)
            X_k, X_k1 = np.meshgrid(x[ind_k1], x[ind_k])
            diff_X_kk1 = np.absolute(X_k-X_k1)
            param_kk1 = w_kk1 / ((1 + diff_X_kk1) ** gamma)
            n_kk1 = np.random.poisson(param_kk1)
            accept_edge_k1 = (n_kk1 > 0)
            ind1_ = np.where(accept_edge_k1 == 1)
            for i in range(len(ind1_[0])):
                G[ind_k[ind1_[0][i]], ind_k1[ind1_[1][i]]] = 1
                G[ind_k1[ind1_[1][i]], ind_k[ind1_[0][i]]] = 1

    # sample self edges
    a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
    a = np.array(a)
    if sum(a) > 0:
        ind_a = np.where(a > 0)[0]
        for i in range(len(ind_a)):
            G[ind_a[i],ind_a[i]] = 1

    G = nx.from_scipy_sparse_matrix(G)
    num_nodes = G.number_of_nodes()

    return [G, w, x, num_nodes]

