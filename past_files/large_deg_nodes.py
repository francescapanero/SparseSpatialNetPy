from RepeatedSamples import *
import time
import matplotlib.pyplot as plt
from PlotGraph import *

alpha = 200
sigma = 0.2
beta = 2
tau = 5
size_x = 5
T = 0.001
start = time.time()
[j, nj, exp_nj] = repeated_samples('large_deg', 'doublepl', 'naive', alpha, sigma, tau, beta, size_x, T=T, c=2)
end = time.time()
print(end-start)

plt_large_deg_nodes(j, nj, exp_nj, sigma, tau, size_x, alpha, T, beta)
plt.legend('naive')
# plt.savefig('largedeg')

