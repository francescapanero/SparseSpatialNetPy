from utils.PlotGraph import *
import numpy as np
from utils.SimpleGraph import *

# --------------------------
# compute and plot degree distribution, ranked degrees, ranked weights, ccdf
# --------------------------

alpha = 200
sigma = 0.3
size_x = 1
tau = 1
c = 2
beta = 0
T = 0.001
K = 10  # number layers for layers typesampler
prior = 'GGP'  # or 'doublepl' or 'exptiltBFRY
typesampler = 'layers'

output = GraphSampler(prior, typesampler, alpha, sigma, tau, beta, size_x, T=T, K=K)
w, x, G, size = output[0:4]

G, isol = SimpleGraph(G)
w_red = np.delete(w, isol)
deg = np.array(list(dict(G.degree()).values()))

# ranked w plot
plt_rank(w_red)
plt_rank(deg)

# degree distribution
plt_deg_distr(deg, prior='GGP', sigma=sigma, binned=False)

# ccdf plot
plt_ccdf(deg, sigma, tau)


