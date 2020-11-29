import numpy as np


def NaiveSampler1(w, x, beta):

    n = len(w) # Number of potential nodes

    # Construct matrix with pairwise distances XY_utr and pairwise products of weights XWw_utr

    X,Y = np.meshgrid(x,x)
    XY = np.absolute(X-Y)
    ind_XY = np.triu_indices(n)
    XY_utr = XY[ind_XY] # upper triangular matrix of pairwise distances

    Xw,Yw = np.meshgrid(w,w)
    XYw = Xw*Yw
    XYw_utr = XYw[ind_XY] # upper triangular matrix of pairwise product of weights
    XYw_ltr = XYw[np.tril_indices(n)]


    # Sample adjacency matrix

    prob = 1 - np.exp(- XYw_utr / (1 + XY_utr ** beta))
    proba = 1 - np.exp(- XYw_ltr / (1 + XY_utr ** beta))
    Z = np.random.rand(len(prob)) < prob # binary adjacency vector
    Za = np.random.rand(len(proba)) < proba
    Z1 = np.zeros((n,n))
    Z1[ind_XY] = Z
    Z1[np.tril_indices(n)] = Za
    Z1 = Z1+ Z1.transpose()
    Z1[Z1>1] = 1

    a = np.random.rand(n) < [1 - np.exp(-w[i] ** 2) for i in range(n)]
    Z1[np.diag_indices(n)] = a


    return Z1

