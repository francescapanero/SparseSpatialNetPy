from utils.RepeatedSamples import *
from utils.PlotGraph import *

# --------------------------
# compare number of nodes and edges as alpha grows
# --------------------------

alpha = [300, 350, 400]
sigma = 0.2
tau = 2.5
beta = 0
size_x = 1
T = 0.001
c = 2  # for doublepl
K = 10  # for layers sampler

n = 2  # number of repeated samples

[nodes_n, edges_n] = repeated_samples('sparsity', 'GGP', 'naive', alpha, sigma, tau, beta, size_x, T=T, n=n, plot=False)
[nodes_l, edges_l] = repeated_samples('sparsity', 'GGP', 'layers', alpha, sigma, tau, beta, size_x, T=T, n=n, K=K,
                                      plot=False)

# plot them together
plt_compare_sparsity(nodes_n, edges_n, nodes_l, edges_l, alpha)


