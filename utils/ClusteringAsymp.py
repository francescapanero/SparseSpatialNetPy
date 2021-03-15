from utils.RepeatedSamples import *
from utils.Limits import *
import numpy as np

# --------------------------
# study the asymptotics of global and local clustering as t grows
# the limit for the GGP looks correct with the naive method, but the doublepl looks off.
# --------------------------

t = np.concatenate((np.array((20, 35)), np.linspace(50, 800, num=16)))
sigma = 0.2
c = 2
tau = 5  # just for doublepl
gamma = 1
size_x = 1
T = 0.001  # lower bound for simulation weights from GGP
n = 7  # number of samples
K = 10  # number of layers
prior = 'singlepl'
approximation = 'truncated'
mode = 'clustering'
sampler = 'layers'

[glob_l, loc_l, glob_lim, loc_lim] = repeated_samples(mode, prior, approximation, sampler, t, sigma, c, tau, gamma, 
                                                      size_x, T=T, K=K, n=n, plot=True)
# plt_compare_clustering(glob_n, loc_n, glob_l, loc_l, glob_lim, loc_lim, t)

# plot asymptotic limits for different betas
beta1 = 100
[limit_glob1, limit_loc1] = limit_clustering(prior, sigma, c, beta1, size_x, tau)
beta2 = 1
[limit_glob2, limit_loc2] = limit_clustering(prior, sigma, c, beta2, size_x, tau)
beta3 = 2
[limit_glob3, limit_loc3] = limit_clustering(prior, sigma, c, beta3, size_x, tau)
beta4 = 0.5
[limit_glob4, limit_loc4] = limit_clustering(prior, sigma, c, beta4, size_x, tau)
plt.hlines(limit_loc1, min(t), max(t), color='b', linestyles='solid')
plt.hlines(limit_glob1, min(t), max(t), color='r', linestyles='solid', label='%i' % int(beta1))
plt.hlines(limit_loc2, min(t), max(t), color='b', linestyles='dashed')
plt.hlines(limit_glob2, min(t), max(t), color='r', linestyles='dashed', label='%i' % int(beta2))
plt.hlines(limit_loc3, min(t), max(t), color='b', linestyles='dashdot')
plt.hlines(limit_glob3, min(t), max(t), color='r', linestyles='dashdot', label='%i' % int(beta3))
plt.hlines(limit_loc4, min(t), max(t), color='b', linestyles='dotted')
plt.hlines(limit_glob4, min(t), max(t), color='r', linestyles='dotted', label='%i' % int(beta4))
plt.legend()
plt.title('Clustering coefficients for varying betas')