import numpy as np
from scipy import sparse


def NaiveSampler_count(w, x, beta): # samples count matrix

    n = len(w) # Number of potential nodes

    # Construct matrix with pairwise distances XY_utr and pairwise products of weights XWw_utr

    X,Y = np.meshgrid(x,x)
    XY = np.absolute(X-Y)
    ind_XY = np.triu_indices(n)
    XY_utr = XY[ind_XY] # upper triangular matrix of pairwise distances

    Xw,Yw = np.meshgrid(w,w)
    XYw = Xw*Yw
    XYw_utr = XYw[ind_XY] # upper triangular matrix of pairwise product of weights

    # Sample adjacency matrix

    prob = 2*XYw_utr/(1+XY_utr**beta) # probability vector of dimension n(n-1)/2+n

    G = np.random.poisson(prob)
    G1 = np.zeros((n,n))
    G1[ind_XY] = G

    G1 = G1 + np.transpose(G1)

    a = [np.random.poisson(w[i] ** 2) for i in range(n)]
    G1[np.diag_indices(n)] = a
    #G1 = sparse.csr_matrix(G1)

    return G1 # outputs count matrix

