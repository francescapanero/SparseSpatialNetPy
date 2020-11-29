import matplotlib.pyplot as plt
import numpy as np
import TruncPois
import Updates
import time
import multiprocessing as mp
import MCMCmethods
from GraphSampler import GraphSampler
import scipy
import shelve
from WeightsSampler import WeightsSampler
import pickle
import seaborn as sns
from MCMC import *
from repeated_samples import *
import networkx as nx
import pandas

# Set parameters alpha, sigma, tau, beta, size_x, c (cutoff), T (the new epsilon, truncation level)

alpha = 300
sigma = 0.2
beta = 5
tau = 2
size_x = 5
c = 2
K = 10
sigma_tau = 0.08
sigma_alpha = 0.01
a_alpha = 200
b_alpha = 1
epsilon = 0.00001
R = 5

iter = 10000
nburn = int(0.75*iter)

################## SIMULATE DATA (with WeightLayers Algorithm) ####################

# [w, x, Z, size, betaw, w0] = GraphSampler("doublepl", "naive", alpha, sigma, tau, beta, size_x,
#                                               T=0.01, c=2)
start = time.time()
[w, x, G, size] = GraphSampler("GGP", "naive", alpha, sigma, tau, beta, size_x, T=0.0001)
end = time.time()
print(end-start)
# [s_true, x, G, size] = GraphSampler("exptiltBFRY", "naive", alpha_true, sigma_true, tau_true, beta, size_x, L=2000)

# np.fill_diagonal(Z, np.zeros(size))

# [s_true, x, G, size] = GraphSampler("exptiltBFRY", "count", alpha_true, sigma_true, tau_true, beta, size_x, L=2000)
#

# precompute distances

dist = np.zeros((size, size))
p_ij = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        dist[i, j] = np.absolute(x[i] - x[j])
        p_ij[i, j] = 1 / (1 + dist[i, j] ** beta)


# CHANGE THIIIIIIIIIIIIS FROM GGP TO DOUBLEPL

# alpha_doublepl = alpha*(c**(tau-sigma))/tau
# t = (sigma*size/alpha_doublepl)**(1/sigma) # size è quella finale o originale?
# u = TruncPois.tpoissrnd(t*w0)
t = (sigma*size/alpha)**(1/sigma)
u = TruncPois.tpoissrnd(t*w)
n = Updates.update_n(w, G, p_ij) # sui vecchi o nuovi w?


# output = MCMC("w_gibbs", "exptiltBFRY", iter, sigma=sigma_true, tau=tau_true, alpha=alpha_true, u=u_true,
#               p_ij=p_ij, n=n_true)
output = MCMC("w_gibbs", "GGP", iter, sigma_tau=0.08,
              tau=tau, sigma=sigma, alpha=alpha, u=u, n=n, p_ij=p_ij, c=c, w_init=w)
# output = MCMC("w_HMC", "exptiltBFRY", iter, epsilon=epsilon, R=R,
#               tau=tau_true, sigma=sigma_true, alpha=alpha_true, u=u_true, n=n_true, p_ij=p_ij, w_init=w)
output = MCMC("w_HMC", "GGP", iter, epsilon=epsilon, R=R,
              tau=tau, sigma=sigma, alpha=alpha, u=u, n=n, p_ij=p_ij, c=c, w_init=w)
# output = MCMC("tau", "doublepl", iter, w=s_true, sigma=sigma_true, alpha=alpha_true, u=u_true, w0=w0,
#               betaw=betaw, c=c)
#
# output = MCMC("alpha", "doublepl", iter, w=s_true, sigma=sigma_true, tau=tau_true, u=u_true, w0=w0,
#               betaw=betaw, c=c, a_alpha=a_alpha, b_alpha=b_alpha, sigma_alpha=sigma_alpha)

# output = MCMC("c", "doublepl", iter, w=s_true, sigma=sigma_true, alpha=alpha_true, tau=tau_true, u=u_true, w0=w0,
#               betaw=betaw, sigma_c=sigma_tau)


# G = nx.from_numpy_matrix(Z)
deg_ = G.degree()
deg = list(dict(deg_).values())
# ind_1 = [list(G.edges())[i][0] for i in range(G.number_of_edges())]
# ind_2 = [list(G.edges())[i][1] for i in range(G.number_of_edges())]
# plt.plot(x_[list(G.edges())])

# ind1, ind2 = np.nonzero(np.triu(Z1, 1))
# plt.plot(x1[ind1], x1[ind2], 'b.', x1[ind2], x1[ind1], 'b.')


################################ Posterior inference #################################


####################### Estimating ONLY SIGMA (and t) ######################


sigma0 = []
t0 = []
sigma0.append(np.random.rand(1))
t0.append(np.float((size * sigma0[0] / alpha_true) ** (1 / sigma0[0])))
temp0 = []
accept0 = []
accept0.append(0)
sigma1 = []
t1 = []
sigma1.append(np.random.rand(1))
t1.append(np.float((size * sigma1[0] / alpha_true) ** (1 / sigma1[0])))
temp1 = []
accept1 = []
accept1.append(0)
sigma2 = []
t2 = []
sigma2.append(np.random.rand(1))
t2.append(np.float((size * sigma2[0] / alpha_true) ** (1 / sigma2[0])))
temp2 = []
accept2 = []
accept2.append(0)

t_in = time.time()
MCMC_sigma(s_true, u_true, sigma0, tau_true, alpha_true, t0, temp0, sigma_tau, iter, accept0)
MCMC_sigma(s_true, u_true, sigma1, tau_true, alpha_true, t1, temp1, sigma_tau, iter, accept1)
MCMC_sigma(s_true, u_true, sigma2, tau_true, alpha_true, t2, temp2, sigma_tau, iter, accept2)
t_tot = time.time() - t_in

print((accept0[iter-1]-accept0[iter-1-10000])/10000) # acceptance rate around 84% for sigma_tau = 0.01, between 15-25% for 0.1
print((accept1[iter-1]-accept1[iter-1-10000])/10000)
print((accept2[iter-1]-accept2[iter-1-10000])/10000)


sigma0_fin = [sigma0[i] for i in range(iter-nburn,iter)]
# sigma1_fin = [sigma1[i] for i in range(iter-nburn,iter)]
# sigma2_fin = [sigma2[i] for i in range(iter-nburn,iter)]
# plt.subplot(3,1,1)
plt.plot(sigma0_fin, label='sigma0_fin')
plt.axhline(y=np.mean(sigma0_fin), label='mean', color='b')
plt.axhline(y=sigma_true, label='true', color='r')
plt.axhline(y=sigma_maxloglik, label='maxloglik', color='g')
plt.legend()
# plt.subplot(3,1,2)
# plt.plot(sigma1_fin, label='sigma1_fin')
# plt.axhline(y=np.mean(sigma1_fin), label='mean', color='r')
# plt.axhline(y=sigma_true, label='true', color='g')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(sigma2_fin, label='sigma2_fin')
# plt.axhline(y=np.mean(sigma2_fin), label='mean', color='r')
# plt.axhline(y=sigma_true, label='true', color='g')
# plt.legend()
#
# plt.savefig('sigma0_1_2_fin.png')

rate0_fin = [rate0[i] for i in range(iter-nburn,iter)]
rate1_fin = [rate1[i] for i in range(iter-nburn,iter)]
rate2_fin = [rate2[i] for i in range(iter-nburn,iter)]
print(np.mean(rate0))
print(np.mean(rate1))
print(np.mean(rate2))

####################### Estimating ONLY TAU ######################

# # del temp0, t0
# tau0 = []
# tau0.append(tau_true)
# temp0 = []
#
# t_in = time.time()
# MCMC_tau(s_true, sigma_true, tau0, t_true, temp0, sigma_tau, iter)
# t_tot = time.time() - t_in
#
#
# plt.plot(tau0, label='tau0')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
#
# tau0_fin = [tau0[i] for i in range(iter-nburn,iter)]
# print(np.mean(tau0_fin))
# plt.plot(tau0_fin, label='tau0_fin')
# plt.axhline(y=np.mean(tau0_fin), label='mean', color='r')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
#
#
tau0 = []
tau0.append(10*np.random.rand(1))
accept0 = []
accept0.append(0)
tau1 = []
tau1.append(10*np.random.rand(1))
accept1 = []
accept1.append(0)
tau2 = []
tau2.append(10*np.random.rand(1))
accept2 = []
accept2.append(0)

temp0 = []
temp1 = []
temp2 = []

t_in = time.time()
MCMC_tau(s_true, sigma_true, tau0, t_true, temp0, sigma_tau, iter, accept0)
MCMC_tau(s_true, sigma_true, tau1, t_true, temp1, sigma_tau, iter, accept1)
MCMC_tau(s_true, sigma_true, tau2, t_true, temp2, sigma_tau, iter, accept2)
t_tot = time.time() - t_in

print((accept0[iter-1]-accept0[iter-1-10000])/10000) # acceptance rate around 81% for sigma_tau = 0.05 and 65% for 0.1
print((accept1[iter-1]-accept1[iter-1-10000])/10000)
print((accept2[iter-1]-accept2[iter-1-10000])/10000)

# plt.subplot(3,1,1)
# plt.plot(tau0, label='tau0')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
# plt.subplot(3,1,2)
# plt.plot(tau1, label='tau1')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(tau2, label='tau2')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
#
#
# plt.plot(tau0, label='tau0')
# plt.plot(tau1, label='tau1', color='r')
# plt.plot(tau2, label='tau2', color='b')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()

tau0_fin = [tau0[i] for i in range(iter-nburn,iter)]
# tau1_fin = [tau1[i] for i in range(iter-nburn,iter)]
# tau2_fin = [tau2[i] for i in range(iter-nburn,iter)]
# plt.subplot(3,1,1)
plt.plot(tau0_fin, label='tau0_fin')
plt.axhline(y=np.mean(tau0_fin), label='mean', color='b')
plt.axhline(y=tau_true, label='true', color='g')
plt.axhline(y=tau_maxloglik, label='maxloglik', color='r')
plt.legend()
# plt.subplot(3,1,2)
# plt.plot(tau1_fin, label='tau1_fin')
# plt.axhline(y=np.mean(tau1_fin), label='mean', color='r')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
# plt.subplot(3,1,3)
# plt.plot(tau2_fin, label='tau2_fin')
# plt.axhline(y=np.mean(tau2_fin), label='mean', color='r')
# plt.axhline(y=tau_true, label='true', color='g')
# plt.legend()
#
# ######################## Estimating ONLY ALPHA ######################
#

alpha0 = []
alpha0.append(np.random.gamma(a_alpha, 1/b_alpha))
# alpha0.append(alpha_true)
t0 = []
t0.append(np.power(size * sigma_true / alpha0[0], 1 / sigma_true))
temp0 = []
accept0 = []
accept0.append(0)
alpha1 = []
alpha1.append(np.random.gamma(a_alpha, 1/b_alpha))
# alpha0.append(alpha_true)
t1 = []
t1.append(np.power(size * sigma_true / alpha1[0], 1 / sigma_true))
temp1 = []
accept1 = []
accept1.append(0)
alpha2 = []
alpha2.append(np.random.gamma(a_alpha, 1/b_alpha))
# alpha0.append(alpha_true)
t2 = []
t2.append(np.power(size * sigma_true / alpha2[0], 1 / sigma_true))
temp2 = []
accept2 = []
accept2.append(0)


# why is this slower than sigma and tau?

t_in = time.time()
MCMC_alpha(alpha0, t0, sigma_true, tau_true, s_true, u_true, temp0,  a_alpha, b_alpha, sigma_alpha, size, iter, accept0)
MCMC_alpha(alpha1, t1, sigma_true, tau_true, s_true, u_true, temp1,  a_alpha, b_alpha, sigma_alpha, size, iter, accept1)
MCMC_alpha(alpha2, t2, sigma_true, tau_true, s_true, u_true, temp2,  a_alpha, b_alpha, sigma_alpha, size, iter, accept2)
t_tot = time.time() - t_in

# plt.subplot(3,1,1)
# plt.plot(alpha0, label='alpha0')
# plt.axhline(y=np.mean(alpha_true), label='true', color='g')
# plt.legend()

# plt.subplot(3,1,1)
alpha0_fin = [alpha0[i] for i in range(iter-nburn,iter)]
print(np.mean(alpha0_fin))
plt.plot(alpha0_fin, label='alpha0_fin')
plt.axhline(y=np.mean(alpha0_fin), label='mean', color='r')
plt.axhline(y=np.mean(alpha_true), label='true', color='g')
plt.legend()
# plt.subplot(3,1,2)
# alpha1_fin = [alpha0[i] for i in range(iter-nburn,iter)]
# print(np.mean(alpha0_fin))
# plt.plot(alpha1_fin, label='alpha1_fin')
# plt.axhline(y=np.mean(alpha1_fin), label='mean', color='r')
# plt.axhline(y=np.mean(alpha_true), label='true', color='g')
# plt.legend()
# plt.subplot(3,1,3)
# alpha2_fin = [alpha2[i] for i in range(iter-nburn,iter)]
# print(np.mean(alpha2_fin))
# plt.plot(alpha2_fin, label='alpha2_fin')
# plt.axhline(y=np.mean(alpha2_fin), label='mean', color='r')
# plt.axhline(y=np.mean(alpha_true), label='true', color='g')
# plt.legend()

########## Estimating c for doublepl ##########
c0 = output[0]
c0_fin = [c0[i] for i in range(iter-nburn,iter)]
# sigma1_fin = [sigma1[i] for i in range(iter-nburn,iter)]
# sigma2_fin = [sigma2[i] for i in range(iter-nburn,iter)]
# plt.subplot(3,1,1)
plt.plot(c0_fin, label='c0_fin')
plt.axhline(y=np.mean(c0_fin), label='mean', color='b')
plt.axhline(y=c, label='true', color='r')
plt.legend()

######################## If you're estimating only sigma, tau and alpha ####################

####### one chain

# del sigma0, tau0, alpha0, t0, temp0
#
# sigma0 = []
# tau0 = []
# t0 = []
# alpha0 = []
# sigma0.append(0.1*np.float(np.random.rand(1)))
# # sigma0.append(sigma_true)
# tau0.append(np.random.gamma(1, 1))
# # tau0.append(tau_true)
# alpha0.append(500*np.float(np.random.rand(1)))
# # alpha0.append(alpha_true)
# t0.append(np.float((size * sigma0[0] / alpha0[0]) ** (1 / sigma0[0])))
# temp0 = []
#
# t_in = time.time()
# MCMC_sigma_tau_alpha(s_true, u_true, sigma0, tau0, alpha0, t0, temp0, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter)
# t_tot = time.time() - t_in
#
# plt.plot(sigma0, label='sigma0')
# plt.legend()
#
# nburn = 10000
#
# sigma0_fin = [sigma0[i] for i in range(iter-nburn,iter)]
# print(np.mean(sigma0_fin))
# plt.plot(sigma0_fin, label='sigma last 500')
# plt.axhline(y=np.mean(sigma0_fin), label='mean', color='r')
# plt.legend()
#
# plt.plot(tau0, label='tau0')
# plt.legend()
#
# tau0_fin = [tau0[i] for i in range(iter-nburn,iter)]
# print(np.mean(tau0_fin))
# plt.plot(tau0_fin, label='tau0_fin')
# plt.axhline(y=np.mean(tau0_fin), label='mean', color='r')
# plt.legend()
#
# plt.plot(alpha0, label='alpha0')
# plt.legend()
#
# alpha0_fin = [alpha0[i] for i in range(iter-nburn,iter)]
# print(np.mean(alpha0_fin))
# plt.plot(alpha0_fin, label='alpha0_fin')
# plt.axhline(y=np.mean(alpha0_fin), label='mean', color='r')
# plt.legend()



# # three chains

# sigma0 = []
# tau0 = []
# t0 = []
# alpha0 = []
# sigma1 = []
# tau1 = []
# t1 = []
# alpha1 = []
# sigma2 = []
# tau2 = []
# t2 = []
# alpha2 = []
# temp0 = []
# temp1 = []
# temp2 =[]

# sigma0.append(np.float(np.random.rand(1)))
# sigma1.append(np.float(np.random.rand(1)))
# sigma2.append(np.float(np.random.rand(1)))
# tau0.append(np.random.gamma(1, 1))
# tau1.append(np.random.gamma(1, 1))
# tau2.append(np.random.gamma(1, 1))
# alpha0.append(500*np.float(np.random.rand(1)))
# alpha1.append(500*np.float(np.random.rand(1)))
# alpha2.append(500*np.float(np.random.rand(1)))
# t0.append(np.float((size * sigma0[0] / alpha0[0]) ** (1 / sigma0[0])))
# t1.append(np.float((size * sigma1[0] / alpha1[0]) ** (1 / sigma1[0])))
# t2.append(np.float((size * sigma2[0] / alpha2[0]) ** (1 / sigma2[0])))

# inputs = [(s_true, u_true, sigma0, tau0, alpha0, t0, temp0, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter),
#           (s_true, u_true, sigma1, tau1, alpha1, t1, temp1, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter),
#           (s_true, u_true, sigma2, tau2, alpha2, t2, temp2, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter)]
#
# t_in = time.time()
# with mp.Pool(processes=3) as pool:
#     results = pool.starmap(MCMC_sigma_tau_alpha, inputs)
# [sigma0, tau0, alpha0, t0] = results[0]
# [sigma1, tau1, alpha1, t1] = results[1]
# [sigma2, tau2, alpha2, t2] = results[2]
# t_tot = time.time() - t_in


############## General case (estimate weights, latent counts, U, sigma, tau) #################

##### one chain

# sigma0 = []
# tau0 = []
# t0 = []
# alpha0 = []
# sigma0.append(np.float(np.random.rand(1)))
# tau0.append(np.random.gamma(1, 1))
# alpha0.append(500*np.float(np.random.rand(1)))
# t0.append(np.float((size * sigma0[0] / alpha0[0]) ** (1 / sigma0[0])))
#
# temp0 = []
#
# S0 = []
# U0 = []
# N0 = []
# g0 = np.random.gamma(1 - sigma0[0], 1, size)
# unif = [np.random.rand(size) for j in range(3)]
# S0.append(np.multiply(g0, np.power(((t0[0] + tau0[0]) ** sigma0[0])*(1 - unif[0]) + (tau0[0] ** sigma0[0])*unif[0], -1 / sigma0[0])))
# U0.append(TruncPois.tpoissrnd(np.multiply(t0[0], S0[0])))
# N0.append(Updates.update_n(S0[0], Z, p_ij))
#
#
# t_in = time.time()
# MCMC(Z, S0, N0, U0, sigma0, tau0, alpha0, t0, temp0, p_ij, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter)
# t_tot = time.time() - t_in


##### three chains
#
# sigma0 = []
# tau0 = []
# t0 = []
# alpha0 = []
# sigma1 = []
# tau1 = []
# t1 = []
# alpha1 = []
# sigma2 = []
# tau2 = []
# t2 = []
# alpha2 = []
#
# sigma0.append(np.float(np.random.rand(1)))
# sigma1.append(np.float(np.random.rand(1)))
# sigma2.append(np.float(np.random.rand(1)))
# tau0.append(np.random.gamma(1, 1))
# tau1.append(np.random.gamma(1, 1))
# tau2.append(np.random.gamma(1, 1))
# alpha0.append(500*np.float(np.random.rand(1)))
# alpha1.append(500*np.float(np.random.rand(1)))
# alpha2.append(500*np.float(np.random.rand(1)))
# t0.append(np.float((size * sigma0[0] / alpha0[0]) ** (1 / sigma0[0])))
# t1.append(np.float((size * sigma1[0] / alpha1[0]) ** (1 / sigma1[0])))
# t2.append(np.float((size * sigma2[0] / alpha2[0]) ** (1 / sigma2[0])))
#
# S0 = []
# U0 = []
# N0 = []
# # temp0 = []
# S1 = []
# U1 = []
# N1 = []
# temp1 = []
# S2 = []
# U2 = []
# N2 = []
# temp2 = []

# # initialization for S from the prior: finite approx of a GGP(sigma, tau): s = G((t+τ)^σ (1−U)+τ^σ U)^(1/sigma)
# # with U unif and G gamma(1-sigma,1)
#
# g0 = np.random.gamma(1 - sigma0[0], 1, size)
# g1 = np.random.gamma(1 - sigma1[0], 1, size)
# g2 = np.random.gamma(1 - sigma2[0], 1, size)
# unif = [np.random.rand(size) for j in range(3)]
# S0.append(np.multiply(g0, np.power(((t0[0] + tau0[0]) ** sigma0[0])*(1 - unif[0]) + (tau0[0] ** sigma0[0])*unif[0], -1 / sigma0[0])))
# U0.append(TruncPois.tpoissrnd(np.multiply(t0[0], S0[0])))
# N0.append(Updates.update_n(S0[0], Z, p_ij))
# S1.append(np.multiply(g1, np.power(((t1[0] + tau1[0]) ** sigma1[0])*(1 - unif[1]) + (tau1[0] ** sigma1[0])*unif[1], -1 / sigma1[0])))
# U1.append(TruncPois.tpoissrnd(np.multiply(t1[0], S1[0])))
# N1.append(Updates.update_n(S1[0], Z, p_ij))
# S2.append(np.multiply(g2, np.power(((t2[0] + tau2[0]) ** sigma2[0])*(1 - unif[2]) + (tau2[0] ** sigma2[0])*unif[2], -1 / sigma2[0])))
# U2.append(TruncPois.tpoissrnd(np.multiply(t2[0], S2[0])))
# N2.append(Updates.update_n(S2[0], Z, p_ij))

# inputs = [(Z, S0, N0, U0, sigma0, tau0, alpha0, t0, temp0, p_ij, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter),
#           (s_true, u_true, sigma1, tau1, alpha1, t1, temp1, p_ij, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter),
#           (s_true, u_true, sigma2, tau2, alpha2, t2, temp2, p_ij, sigma_tau, sigma_alpha, a_alpha, b_alpha, size, iter)]
#
# t_in = time.time()
# with mp.Pool(processes=3) as pool:
#     results = pool.starmap(MCMC, inputs)
# [S0, N0, U0, sigma0, tau0, alpha0, t0] = results[0]
# [S1, N1, U1, sigma1, tau1, alpha1, t1] = results[1]
# [S2, N2, U2, sigma2, tau2, alpha2, t2] = results[2]
# t_tot = time.time() - t_in




############### estimating only WEIGHTS ##############



# k = 3
# ind_big = np.argsort(s_true)[len(s_true)-k:len(s_true)]
# big_s = s_true[ind_big]
# for j in range(k):
#     plt.subplot(k, 1, j+1)
#     plt.plot([S0_fin[i][ind_big[j]] for i in range(nburn)])
#     plt.axhline(s_true[ind_big[j]], label='true', color='r')
#     plt.axhline(S0_avg[ind_big[j]], label='avg est')
#     plt.legend()



S0_fin_highestdeg = [S0_fin[i][ind_big1[len(ind_big1)-5]] for i in range(len(S0_fin))]
plt.plot(S0_fin_highestdeg, label='S0_fin highest deg')
plt.axhline(y=S0_avg[ind_big1[len(ind_big1)-1]], label='mean', color='r')
plt.axhline(y=w[ind_big1[len(ind_big1)-1]], label='true', color='g')
# plt.axhline(y=s_linspace[d_ind], label='max loglik', color='orange')
plt.legend()

sns.distplot(S0_fin_highestdeg, hist = False, kde = True,
                 kde_kws = {'linewidth': 3}, label="sample")
plt.axvline(x=S0_avg[ind_big1[len(ind_big1)-1]], label='mean', color='r')
plt.axvline(x=w[ind_big1[len(ind_big1)-1]], label='true', color='g')
# plt.axvline(x=s_linspace[d_ind], label='max loglik', color='orange')
plt.legend()

S0_highestdeg = [S0[i][ind_big1[len(ind_big1)-1]] for i in range(iter)]
plt.plot(S0_highestdeg, label='S0_fin highest deg')
plt.axhline(y=S0_avg[ind_big1[len(ind_big1)-5]], label='mean', color='r')
plt.axhline(y=w[ind_big1[len(ind_big1)-5]], label='true', color='g')
# plt.axhline(y=s_linspace[d_ind], label='max loglik', color='orange')
plt.legend()


x_big = x[ind_big]
x_small = x[ind_small]
plt.subplot(2,2,3)
plt.plot(x_big, 'bo')
plt.subplot(2,2,4)
plt.plot(x_small, 'bo')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('index')
ax1.set_ylabel('s_i', color=color)
for i in range(50):
    ax1.plot((i,i), (emp_CI_big[i][0],emp_CI_big[i][1]), 'ro-', color=color)
    ax1.plot(i, big_s_20[i], 'x', color="black")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('x_i', color=color)  # we already handled the x-label with ax1
ax2.plot(x_big, 'x', color=color)
ax2.tick_params(axis='y', labelcolor=color)



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('x_i', color=color)  # we already handled the x-label with ax1
ax2.plot((i+1,i+1), (emp_CI_small[i][0],emp_CI_small[i][1]), 'ro-', color=color)
ax2.plot(i+1, np.log(small_s_20[i]), 'bo', color=color)
ax2.tick_params(axis='y', labelcolor=color)


plt.savefig('big_small_10^4nodes_10^5iter.png')




#### three chains

#
# S0_fin = [S0[i] for i in range(iter-nburn,iter)]
# S1_fin = [S1[i] for i in range(iter-nburn,iter)]
# S2_fin = [S2[i] for i in range(iter-nburn,iter)]
# ## estimates of S with average over last iterations
# S0_avg = [np.mean([S0_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S1_avg = [np.mean([S1_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S2_avg = [np.mean([S2_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S0_var = np.divide(np.power([sum([S0_fin[i][j] for i in range(nburn)]) - size*S0_avg[j] for j in range(size)], 2), size-1)
# # S1_var = np.divide(np.power([sum([S1_fin[i][j] for i in range(nburn)]) - size*S1_avg[j] for j in range(size)], 2), size-1)
# # S2_var = np.divide(np.power([sum([S2_fin[i][j] for i in range(nburn)]) - size*S2_avg[j] for j in range(size)], 2), size-1)
#
# emp0_CI_95 = [scipy.stats.mstats.mquantiles([S0_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# # emp1_CI_95 = [scipy.stats.mstats.mquantiles([S1_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# # emp2_CI_95 = [scipy.stats.mstats.mquantiles([S2_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# true0_in_CI = [emp0_CI_95[i][0] <= s_true[i] <= emp0_CI_95[i][1] for i in range(size)]
# # true1_in_CI = [emp1_CI_95[i][0] <= s_true[i] <= emp1_CI_95[i][1] for i in range(size)]
# # true2_in_CI = [emp2_CI_95[i][0] <= s_true[i] <= emp2_CI_95[i][1] for i in range(size)]
# print(sum(true0_in_CI)/len(true0_in_CI))
# # perc1 = sum(true1_in_CI)/len(true1_in_CI)
# # perc2 = sum(true2_in_CI)/len(true2_in_CI)
#
# for j in range(10):
#     plt.plot([S0_fin[i][j] for i in range(nburn)], label='%s' % j)
#     plt.axhline(s_true[j])
# plt.legend()
# # for j in range(10):
# #     plt.plot([S1_fin[i][j] for i in range(nburn)], label='%s' % j)
# # plt.legend()
# # plt.savefig('traceplots/S1.png', bbox_inches='tight')
# # plt.close()
# # for j in range(10):
# #     plt.plot([S2_fin[i][j] for i in range(nburn)], label='%s' % j)
# # plt.legend()
# # plt.savefig('traceplots/S2.png', bbox_inches='tight')
# # plt.close()
#
# # plot the histogram of the estimate of s[0]
# est_s = []
# for j in range(10):
#     est_s.append([S0_fin[i][j] for i in range(nburn)])
# plt.hist(est_s[0], bins=50)
#
# # plot empirical 95% CI
# s_true_sort = np.sort(s_true)
# # of the biggest 20 values
# big_s_20 = s_true_sort[range(len(s_true)-20, len(s_true))]
# ind_big = np.argsort(s_true)[range(len(s_true)-20, len(s_true))]
# emp_CI_big = []
# for i in range(20):
#     emp_CI_big.append(emp0_CI_95[ind_big[i]])
# for i in range(20):
#     plt.plot((i+1,i+1), (emp_CI_big[i][0],emp_CI_big[i][1]), 'ro-')
#     plt.plot(i+1, big_s_20[i], 'bo')
# # of the smallest 20 values (log scale)
# log_emp0_CI_95 = [scipy.stats.mstats.mquantiles([np.log(S0_fin[i][j]) for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# small_s_20 = np.log(s_true_sort[range(20)])
# ind_small = np.argsort(s_true)[range(0,20)]
# emp_CI_small = []
# for i in range(20):
#     emp_CI_small.append(log_emp0_CI_95[ind_small[i]])
# for i in range(20):
#     plt.plot((i+1,i+1), (emp_CI_small[i][0],emp_CI_small[i][1]), 'ro-')
#     plt.plot(i+1, small_s_20[i], 'bo')


############### estimating weights, latent counts and U ##############

####### one chain


# S0 = []
# U0 = []
# N0 = []
#
# ## initialization for S from the prior: finite approx of a GGP(sigma, tau): s = G((t+τ)^σ (1−U)+τ^σ U)^(1/sigma)
# ## with U unif and G gamma(1-sigma,1)
#
# sigma_guess = np.random.rand(0,1)
# tau_guess = np.random.gamma(2, 1)
# alpha_guess = 200*np.random.rand(1)
# t_guess = (sigma_guess*size/alpha_guess)**(1/sigma_guess)
# g0 = np.random.gamma(1 - sigma_guess, 1, size)
# unif = [np.random.rand(size)]
# S0.append(np.multiply(g0, np.power(((t_guess + tau_guess) ** sigma_guess)*(1 - unif) + (tau_guess ** sigma_guess)*unif, -1 / sigma_guess)))
# U0.append(TruncPois.tpoissrnd(np.multiply(t0[0], S0[0])))
# N0.append(Updates.update_n(S0[0], Z, p_ij))
#
# t_in = time.time()
# [S0, N0, U0] = MCMC_w_n_u(Z, S0, N0, U0, sigma_true, tau_true, t_true, p_ij, iter)
# t_tot = time.time() - t_in

######## three chains

# S0 = []
# U0 = []
# N0 = []
# S1 = []
# U1 = []
# N1 = []
# S2 = []
# U2 = []
# N2 = []

# sigma0_guess = np.random.rand(0,1)
# tau0_guess = np.random.gamma(2, 1)
# alpha0_guess = 200*np.random.rand(1)
# t0_guess = (sigma0_guess*size/alpha0_guess)**(1/sigma0_guess)
# sigma1_guess = np.random.rand(0,1)
# tau1_guess = np.random.gamma(2, 1)
# alpha1_guess = 200*np.random.rand(1)
# t1_guess = (sigma1_guess*size/alpha1_guess)**(1/sigma1_guess)
# sigma2_guess = np.random.rand(0,1)
# tau2_guess = np.random.gamma(2, 1)
# alpha2_guess = 200*np.random.rand(1)
# t2_guess = (sigma2_guess*size/alpha2_guess)**(1/sigma2_guess)

# g0 = np.random.gamma(1 - sigma0_guess, 1, size)
# g1 = np.random.gamma(1 - sigma1_guess, 1, size)
# g2 = np.random.gamma(1 - sigma2_guess, 1, size)
# unif = [np.random.rand(size) for j in range(3)]
# S0.append(np.multiply(g0, np.power(((t0_guess + tau0_guess) ** sigma0_guess)*(1 - unif) + (tau_guess ** sigma0_guess)*unif, -1 / sigma0_guess)))
# U0.append(TruncPois.tpoissrnd(np.multiply(t0_guess, S0[0])))
# N0.append(Updates.update_n(S0[0], Z, p_ij))
# S1.append(np.multiply(g1, np.power(((t1_guess + tau1_guess) ** sigma1_guess)*(1 - unif[1]) + (tau1_guess ** sigma1_guess)*unif[1], -1 / sigma1_guess)))
# U1.append(TruncPois.tpoissrnd(np.multiply(t1_guess, S1[0])))
# N1.append(Updates.update_n(S1[0], Z, p_ij))
# S2.append(np.multiply(g2, np.power(((t2_guess + tau2_guess) ** sigma2_guess)*(1 - unif[2]) + (tau2_guess ** sigma2_guess)*unif[2], -1 / sigma2_guess)))
# U2.append(TruncPois.tpoissrnd(np.multiply(t2_guess, S2[0])))
# N2.append(Updates.update_n(S2[0], Z, p_ij))

# inputs = [(Z, S0, N0, U0, sigma_true, tau_true, t_true, p_ij, iter),
#           (Z, S1, N1, U1, sigma_true, tau_true, t_true, p_ij, iter),
#           (Z, S2, N2, U2, sigma_true, tau_true, t_true, p_ij, iter)]
#
# t_in = time.time()
# with mp.Pool(processes=3) as pool:
#     results = pool.starmap(MCMC_w_n_u, inputs)
# [S0, N0, U0] = results[0]
# [S1, N1, U1] = results[1]
# [S2, N2, U2] = results[2]
# t_tot = time.time() - t_in

#
# S0_fin = [S0[i] for i in range(iter-nburn,iter)]
#
# # simple estimates of S with average over last iterations
# S0_avg = [np.mean([S0_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S1_avg = [np.mean([S1_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S2_avg = [np.mean([S2_fin[i][j] for i in range(nburn)]) for j in range(size)]
# # S0_var = np.divide(np.power([sum([S0_fin[i][j] for i in range(nburn)]) - size*S0_avg[j] for j in range(size)], 2), size-1)
# # S1_var = np.divide(np.power([sum([S1_fin[i][j] for i in range(nburn)]) - size*S1_avg[j] for j in range(size)], 2), size-1)
# # S2_var = np.divide(np.power([sum([S2_fin[i][j] for i in range(nburn)]) - size*S2_avg[j] for j in range(size)], 2), size-1)
#
# emp0_CI_95 = [scipy.stats.mstats.mquantiles([S0_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# # emp1_CI_95 = [scipy.stats.mstats.mquantiles([S1_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# # emp2_CI_95 = [scipy.stats.mstats.mquantiles([S2_fin[i][j] for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# true0_in_CI = [emp0_CI_95[i][0] <= s_true[i] <= emp0_CI_95[i][1] for i in range(size)]
# # true1_in_CI = [emp1_CI_95[i][0] <= s_true[i] <= emp1_CI_95[i][1] for i in range(size)]
# # true2_in_CI = [emp2_CI_95[i][0] <= s_true[i] <= emp2_CI_95[i][1] for i in range(size)]
# perc0 = sum(true0_in_CI)/len(true0_in_CI)
# # perc1 = sum(true1_in_CI)/len(true1_in_CI)
# # perc2 = sum(true2_in_CI)/len(true2_in_CI)
# print(perc0)
# # print(perc1)
# # print(perc2)
#
# # post_int_95 = [scipy.stats.gamma.interval()]
#
# # [plt.boxplot([S_fin[i][j] for i in range(len(S_fin))]) for j in range(size)]
# # plt.plot(s_true[0])
#
# # import matplotlib
# #
# # matplotlib.axes.Axes.axvline(x=1, ymin=emp_CI_95[1][0], ymax=emp_CI_95[1][1])
#
#
# for j in range(10):
#     plt.plot([S0_fin[i][j] for i in range(nburn)], label='%s' % j)
#     plt.axhline(s_true[j])
# plt.legend()
# plt.savefig('traceplots/S0.png', bbox_inches='tight')
# plt.close()
# # for j in range(10):
# #     plt.plot([S1_fin[i][j] for i in range(nburn)], label='%s' % j)
# # plt.legend()
# # plt.savefig('traceplots/S1.png', bbox_inches='tight')
# # plt.close()
# # for j in range(10):
# #     plt.plot([S2_fin[i][j] for i in range(nburn)], label='%s' % j)
# # plt.legend()
# # plt.savefig('traceplots/S2.png', bbox_inches='tight')
# # plt.close()
#
# # plot the histogram of the estimate of s[0]
# est_s = []
# for j in range(10):
#     est_s.append([S0_fin[i][j] for i in range(nburn)])
# plt.hist(est_s[0], bins=50)
#
# # plot empirical 95% CI
# s_true_sort = np.sort(s_true)
# # of the biggest 20 values
# big_s_20 = s_true_sort[range(len(s_true)-20, len(s_true))]
# ind_big = np.argsort(s_true)[range(len(s_true)-20, len(s_true))]
# emp_CI_big = []
# for i in range(20):
#     emp_CI_big.append(emp0_CI_95[ind_big[i]])
# for i in range(20):
#     plt.plot((i+1,i+1), (emp_CI_big[i][0],emp_CI_big[i][1]), 'ro-')
#     plt.plot(i+1, big_s_20[i], 'bo')
# # of the smallest 20 values (log scale)
# log_emp0_CI_95 = [scipy.stats.mstats.mquantiles([np.log(S0_fin[i][j]) for i in range(nburn)], prob=[0.025, 0.975]) for j in range(size)]
# small_s_20 = np.log(s_true_sort[range(20)])
# ind_small = np.argsort(s_true)[range(0,20)]
# emp_CI_small = []
# for i in range(20):
#     emp_CI_small.append(log_emp0_CI_95[ind_small[i]])
# for i in range(20):
#     plt.plot((i+1,i+1), (emp_CI_small[i][0],emp_CI_small[i][1]), 'ro-')
#     plt.plot(i+1, small_s_20[i], 'bo')





# for j in range(20):
#     plt.plot([U[i][j] for i in range(iter-100,iter)])
#
# print([S[i][1] for i in range(4970, iter)])
# max([max([S[i][j] for i in range(iter-100,iter)]) for j in range(size)]) # max value
# # check which node has max avg value of s
# a = [np.mean([S[i][j] for i in range(iter-1000,iter)]) for j in range(size)]
# a.index(max(a))





# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('objs.pkl', 'wb') as f:
    pickle.dump([s_true, x, G, size, S0, S0_fin, S0_avg, emp0_CI_95,deg_sort, log_emp0_CI_95], f)

# Getting back the objects:
with open('objs.pkl', 'rb') as f:
    s_true, x, G, size, S0, S0_fin, S0_avg, emp0_CI_95,deg_sort, log_emp0_CI_95 = pickle.load(f)

# Save workspace: need to change it so that it only saves useful variables o/w explodes

# filename='/tmp/18nov_100000iter.out'
# my_shelf = shelve.open(filename,'n') # 'n' for new
#
# for key in dir():
#     try:
#         my_shelf[key] = globals()[key]
#     except TypeError:
#         #
#         # __builtins__, my_shelf, and imported modules can not be shelved.
#         #
#         print('ERROR shelving: {0}'.format(key))
# my_shelf.close()
