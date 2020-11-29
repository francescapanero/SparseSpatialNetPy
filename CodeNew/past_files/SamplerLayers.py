import numpy as np
from WeightLayers import WeightLayers
from scipy.sparse import lil_matrix
import networkx as nx


# implementation with sparse matrix (lil matrix)

def SamplerLayers_optim(w, x, beta, size_x, K):

    if sum(x) == 0:
        print('You cannot use the Layers method without locations!')

    n = len(w)
    G = lil_matrix((n,n))  # set adjacency matrix to 0. lil is a sparse matrix

    # define K cells in [0,size_x]

    _, bin = np.histogram(x, K, range=(0, size_x))
    loc = np.array(np.digitize(x, bin))  # index of the cell to which nodes belong
    delta = size_x/K

    lay = WeightLayers(w)
    layer = lay[1]
    w_lay = lay[0]

    if K > 2:

        # construct pmf p_K (each of the K-2 rows corresponds to a k) that represents the pobabilities of connection
        # with cells k+2...K
        # Z[k] is the normalizing constant of p_K[k,:]
        Z = np.zeros(K - 2)
        p_K = np.zeros((K-2, K))
        for k in range(K-2):  # O(K)
            p_K[k, (k+2):K] = np.array(1 / ((1 - delta + delta * np.array(range(2, K - k))) ** beta))
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
                                    prob = ((1 - delta + delta*np.absolute(k-l)) ** beta) / \
                                           ((1 + np.absolute(x_j1k - x_j2l)) ** beta)
                                    accept_edge = np.random.rand(1) < prob
                                    if accept_edge == 1:
                                        print(prob * p_K[k, l] * w_j1j2kl_flat[a] * (W[j2][l] / (w_lay[j2 + 1] * V[j2])) * W[j1, k] * w_lay[j2 + 1] * V[j2] * Z[k])
                                        print(w[ind_x] * w[ind_y] / ((1 + np.absolute(x_j1k - x_j2l)) ** beta))
                                        G[ind_x, ind_y] = 1
                                        G[ind_y, ind_x] = 1

    # sample connections from cell k=1..k to cell k or k+1
    for k in range(K):
        ind_k = np.where(loc == (k+1))[0]
        w_k_temp = w[ind_k]
        w_kk = np.outer(w_k_temp, w_k_temp)
        X_k, X_kk = np.meshgrid(x[ind_k], x[ind_k])
        diff_X_kk = np.absolute(X_k - X_kk)
        param_kk = w_kk / ((1 + diff_X_kk) ** beta)
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
            param_kk1 = w_kk1 / ((1 + diff_X_kk1) ** beta)
            n_kk1 = np.random.poisson(param_kk1)
            accept_edge_k1 = (n_kk1 > 0)
            ind1_ = np.where(accept_edge_k1 == 1)
            for i in range(len(ind1_[0])):
                G[ind_k[ind1_[0][i]], ind_k1[ind1_[1][i]]] = 1
                G[ind_k1[ind1_[1][i]], ind_k[ind1_[0][i]]] = 1

    a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
    a = np.array(a)
    if sum(a) > 0:
        ind_a = np.where(a > 0)[0]
        for i in range(len(ind_a)):
            G[ind_a[i],ind_a[i]] = 1

    G = nx.from_scipy_sparse_matrix(G)
    # G.remove_edges_from(nx.selfloop_edges(G)) # NO SELF LOOPS
    # isol = list(nx.isolates(G))
    # x = np.delete(x, isol)
    # w = np.delete(w, isol)
    # G.remove_nodes_from(isol)
    num_nodes = G.number_of_nodes()

    return [G, w, x, num_nodes]


# # older version of the sampler, but doesn't use sparse matrix
#
# def SamplerLayers(w, x, beta, size_x, K):
#
#     if sum(x) == 0:
#         print('You cannot use the Layers method without locations!')
#
#     n = len(w)
#     G = np.zeros((n,n)) # set adjacency matrix to 0
#
#     # define K cells in [0,size_x]
#
#     _, bin = np.histogram(x, K, range=(0,size_x))
#     loc = np.digitize(x, bin)
#     loc = np.array(loc) # index of the cell to which nodes belong
#     delta = size_x/K
#
#     lay = WeightLayers(w)
#     layer = lay[1]
#     w_lay = lay[0]
#
#     Z = np.zeros(K - 2)
#     p_K = np.zeros((K-2,K))
#     for k in range(K-2): # O(K)
#         p_K[k,(k+2):K] = np.array(1/((1 - delta + delta * np.array(range(2, K - k))) ** beta))
#         Z[k] = p_K[k,].sum()
#         p_K[k,] = p_K[k,]/Z[k]
#
#
#     V = np.zeros((max(layer))) # n elements in every layer
#     w_jk = [[[] for k in range(K)] for j in range(max(layer))] # list J x K
#     ind_jk = [[[] for k in range(K)] for j in range(max(layer))]
#     # in every element there are weights in cell k and layer j
#     W = np.zeros((max(layer), K)) # in position j,k there is the sum of w_jk[j][k]
#     loc_jk = [[[] for k in range(K)] for j in range(max(layer))]
#     # same but with locations
#     for j in range(max(layer)):
#         V[j] = sum((layer == (j+1)))
#         for k in range(K):
#             w_jk[j][k] = (w[(loc == (k+1)) & (layer == (j+1))])
#             W[j][k] = sum(w_jk[j][k])
#             loc_jk[j][k] = ((loc == (k+1)) & (layer == (j+1)))
#
#     loc_k = []
#     w_k_temp = []
#     x_k = []
#     for k in range(K):
#         loc_k.append((loc == (k + 1)))
#         w_k_temp.append(w[loc_k[k]])
#         x_k.append(x[loc_k[k]])
#
#
#     l = [[[] for j in range(max(layer))] for i in range(max(layer))]
#     for i in range(max(layer)):
#         for j in range(i,max(layer)):
#             n_ij = np.random.poisson(2 * W[i,0:(K-2)] * w_lay[j + 1] * V[j] * Z, K - 2)
#             for k in range(K-2):
#                 l[i][j].append(0) # this helps if there is a n_ij[k] = 0, o/w the for would be stuck
#                 if n_ij[k] != 0:
#                     l[i][j][k] = np.random.choice(np.array(range(k + 2, K)), n_ij[k], p=p_K[k, (k + 2):K], replace=True)
#                     # l_temp = l[k].astype(int)
#                     index_temp = np.random.rand(n_ij[k]) < (W[j][l[i][j][k]]/(w_lay[j+1] * V[j]))
#                     l[i][j][k] = l[i][j][k][index_temp]
#                     b_temp = l[i][j][k]
#                     for m in range(len(b_temp)):
#                         b = int(b_temp[m])
#                         w_ijkb = np.outer(w_jk[i][k], w_jk[j][b])/(W[i][k]*W[j][b])
#                         w_ijkb = np.array(w_ijkb)
#                         w_ijkb_flat = np.ndarray.flatten(w_ijkb)
#                         a = np.random.choice(range(1,w_ijkb.shape[0]*w_ijkb.shape[1]+1), 1, p=w_ijkb_flat)
#                         row = int(np.ceil(a/w_ijkb.shape[1]))-1
#                         col = int(a % w_ijkb.shape[1])-1
#                         x_jb = x[loc_jk[j][b]]
#                         x_ik = x[loc_jk[i][k]]
#                         diff_x_ijkb = np.absolute(x_ik[row] - x_jb[col])
#                         prob = ((1-delta+delta*np.absolute(k-b))**beta)/((1+diff_x_ijkb)**beta)
#                         accept_edge = np.random.rand(1) < prob
#                         G_temp = np.zeros((w_ijkb.shape[0],w_ijkb.shape[1]))
#                         G_temp[row][col] = accept_edge
#                         G[np.ix_(loc_jk[i][k], loc_jk[j][b])] = G[np.ix_(loc_jk[i][k], loc_jk[j][b])] + G_temp
#                         G[np.ix_(loc_jk[j][b], loc_jk[i][k])] = G[np.ix_(loc_jk[j][b], loc_jk[i][k])] + G_temp.T
#
#     for k in range(K):
#         loc_k = (loc == (k+1))
#         w_k_temp = w[loc_k]
#         w_kk = np.outer(w_k_temp, w_k_temp)
#         X_k, X_kk = np.meshgrid(x[loc_k], x[loc_k])
#         diff_X_kk = np.absolute(X_k - X_kk)
#         param_kk = w_kk / ((1 + diff_X_kk) ** beta) # *2 ?
#         n_kk = np.random.poisson(param_kk)
#         accept_edge_k = (n_kk > 0)
#         accept_edge_k = accept_edge_k + accept_edge_k.T
#         G[np.ix_(loc_k, loc_k)] = accept_edge_k
#         if k != (K-1):
#             loc_k1 = (loc == (k + 2))
#             w_k1_temp = w[loc_k1]
#             w_kk1 = np.outer(w_k_temp, w_k1_temp)
#             X_k, X_k1 = np.meshgrid(x[loc_k1], x[loc_k])
#             diff_X_kk1 = np.absolute(X_k-X_k1)
#             param_kk1 = 2*w_kk1 / ((1 + diff_X_kk1) ** beta) # *2 ?
#             n_kk1 = np.random.poisson(param_kk1)
#             accept_edge_k1 = (n_kk1 > 0)
#             G[np.ix_(loc_k, loc_k1)] = accept_edge_k1
#             G[np.ix_(loc_k1, loc_k)] = accept_edge_k1.T
#
#     a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
#     a = np.array(a)
#     G[np.diag_indices(n)] = a
#
#     return G




