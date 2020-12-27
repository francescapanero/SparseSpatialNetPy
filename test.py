from utils.TestSample import *

# --------------------------
# run the tests to check the samples implemented in utils.TestSample
# --------------------------


# Set parameters for simulating data
t = 400
sigma = 0.3
c = 5
tau = 5  # if singlepl then set it to 0, otherwise > 1
gamma = 0  # exponent distance in the link probability
size_x = 1

K = 100  # number of layers, for layers sampler
T = 0.00001  # threshold simulations weights for GGP and doublepl (with w0 from GGP)
L = 500000  # tot number of nodes in exptiltBFRY

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'truncated'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

# check layers vs naive sampling, samples n sample to get statistics about mean and var of number of nodes / edges
unit_test('layers_vs_naive', prior, approximation, sigma, c, t, tau, gamma, size_x, n=20, T=T, K=K,
          L=L)

# check sampler outputs
unit_test('sampler', prior, approximation, sigma, c, t, tau, gamma, size_x, n=10, T=T, K=K, L=L, sampler=sampler)

# clustering asymptotics comparison (layers vs naive sampling)
t = np.concatenate((np.array((20, 35)), np.linspace(50, 350, num=7)))
unit_test('clustering_coefficient', prior, approximation, sigma, c, t, tau, gamma, size_x, n=10, T=T, K=K, L=L)

# number of nodes and edges comparison (layers vs naive sampling)
alpha = np.linspace(50, 500, num=10)
unit_test('sparsity', prior, approximation, sigma, c, t, tau, gamma, size_x, n=10, T=T, K=K, L=L)