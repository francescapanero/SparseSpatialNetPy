import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
import networkx as nx

def NaiveSampler(w, x, beta):

    n = len(w) # Number of potential nodes

    # Construct matrix with pairwise distances XY_utr and pairwise products of weights XWw_utr

    X,Y = np.meshgrid(x,x)
    XY = np.absolute(X-Y)
    ind_XY = np.triu_indices(n)
    XY_utr = XY[ind_XY] # upper triangular matrix of pairwise distances

    Xw,Yw = np.meshgrid(w,w)
    XYw = Xw*Yw
    XYw_utr = XYw[ind_XY] # upper triangular matrix of pairwise product of weights

    # Same but with locations in R^2
    # dist = np.zeros((n,n))
    # for i in range(n):
    #    for j in range(i+1, n):
    #        dist[i,j] = np.sqrt((X[2*i,2*j]-Y[2*i,2*j])**2+(X[2*i+1,2*j+1]-Y[2*i+1,2*j+1])**2)
    # XY_utr = dist[ind_XY]

    # Sample adjacency matrix

    # prob = 1 - np.exp(-XYw / ((1 + XY) ** beta))
    prob = 1 - np.exp(-2*XYw_utr/((1+XY_utr)**beta))  # probability vector of dimension n(n-1)/2+n
    # Z = np.random.rand(n,n) < prob
    Z = np.random.rand(len(prob)) < prob  # binary adjacency vector
    Z1 = np.zeros((n,n))
    Z1[ind_XY] = Z
    Z1 = Z1 + np.transpose(Z1)  # adjacency matrix, including non connected nodes
    a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
    Z1[np.diag_indices(n)] = a

    Z1 = lil_matrix(Z1)
    G = nx.from_scipy_sparse_matrix(Z1)
    # isol = list(nx.isolates(G))
    # x = np.delete(x, isol)
    # w = np.delete(w, isol)
    # G.remove_nodes_from(isol)
    num_nodes = G.number_of_nodes()

    return [G, w, x, num_nodes]



# def NaiveSampler_optim(w, x, beta):
#
#     n = len(w) # Number of potential nodes
#
#     # Construct matrix with pairwise distances XY_utr and pairwise products of weights XWw_utr
#
#     X,Y = np.meshgrid(x,x)
#     XY = np.absolute(X-Y)
#     ind_XY = np.triu_indices(n)
#     XY_utr = XY[ind_XY] # upper triangular matrix of pairwise distances
#
#     Xw,Yw = np.meshgrid(w,w)
#     XYw = Xw*Yw
#     XYw_utr = XYw[ind_XY] # upper triangular matrix of pairwise product of weights
#
#     # Sample adjacency matrix
#
#     prob = 1-np.exp(-XYw_utr/((1+XY_utr)**beta)) # probability vector of dimension n(n-1)/2 + n
#     Z = np.random.rand(len(prob)) < prob # binary adjacency vector
#     a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
#     ind_Z = np.where(Z > 0)
#     ind_a = np.where(a > 0)
#     Z1 = lil_matrix((n,n))
#     for i in ind_Z[0]:
#         if ind_XY[0][i] != ind_XY[1][i]:
#             Z1[ind_XY[0][i], ind_XY[1][i]] = 1
#             Z1[ind_XY[1][i], ind_XY[0][i]] = 1
#     for i in ind_a[0]:
#         Z1[i,i] = 1
#
#     G = nx.from_scipy_sparse_matrix(Z1)
#     isol = list(nx.isolates(G))
#     x = np.delete(x, isol)
#     w = np.delete(w, isol)
#     G.remove_nodes_from(isol)
#
#     return [G, w, x, isol]
