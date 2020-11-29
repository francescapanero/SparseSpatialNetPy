import matplotlib.pyplot as plt
from utils.RepeatedSamples import *

alpha = [100, 200, 300]
size_x = 1
tau = 2
beta = 0
T = 0.0001

sigma_1 = 0.2
[n_alpha1, n_alpha_e1] = repeated_samples('sparsity', 'GGP', 'layers', alpha, sigma_1, tau, beta, size_x, T=T, c=2, K=10)
[n_alpha1_naive, n_alpha_e1_naive] = repeated_samples('sparsity', 'GGP', 'naive', alpha, sigma_1, tau, beta, size_x, T=T)

plt.plot(n_alpha1, n_alpha_e1, 'b--o', label='sigma=%f obs lay' %sigma_1)
plt.plot(n_alpha1, n_alpha1**(2/(1+sigma_1)), 'b-o', label='sigma=%f th lay' %sigma_1)
plt.plot(n_alpha1_naive, n_alpha_e1_naive, 'r--o', label='sigma=%f obs naive' %sigma_1)
plt.xlabel('number of nodes')
plt.ylabel('number of edges')
plt.xscale("log")
plt.yscale("log")
plt.title('GGP sigma=%f, tau=%i, size_x=%i, T=%f' % (sigma_1, tau, size_x, T))
plt.legend()