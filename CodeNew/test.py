from utils.TestSample import *

# --------------------------
# run the tests to check the samples implemented in utils.TestSample
# --------------------------


# Set parameters for simulation data
alpha = 50
sigma = 0.2
tau = 1
beta = 1
size_x = 100
c = 2  # just for doublepl

T = 0.001  # lower bound for simulation weights from GGP

n = 10  # sometimes you need to specify number of samples

K = 50  # number of layers

# check layers vs naive sampling, samples n sample to get statistics about mean and var of number of nodes / edges
unit_test('layers_vs_naive', 'GGP', sigma, tau, alpha, beta, size_x, T, n=n, K=K)

# check sampler outputs
unit_test('sampler', 'doublepl', sigma, tau, alpha, beta, size_x, T, K=K, c=c, typesampler='layers')

# clustering asymptotics comparison (layers vs naive sampling)
alpha = np.concatenate((np.array((20, 35)), np.linspace(50, 350, num=7)))
unit_test('clustering_coefficient', 'GGP', sigma, tau, alpha, beta, size_x, T, K=K, n=n)

# number of nodes and edges comparison (layers vs naive sampling)
alpha = np.linspace(50, 500, num=10)
unit_test('sparsity', 'GGP', sigma, tau, alpha, beta, size_x, T, c=c, K=K, n=n)