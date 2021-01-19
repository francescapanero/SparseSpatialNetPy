import utils.MCMCNew_fast as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew_fast as aux
import utils.UpdatesNew_fast as up
import numpy as np
import pandas as pd
# import pymc3 as pm3
import matplotlib.pyplot as plt
import scipy

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 2  # exponent distance in the link probability
size_x = 1  # space threshold: [0, size_x]

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs
L = 800  # tot number of nodes in finite approx of weights simulations (exptiltBFRY)

# prior parameters of t \sim gamma(a_t, b_t)
a_t = 200
b_t = 1

# prior for weights and type of sampler
prior = 'singlepl'  # can be 'singlepl' or 'doublepl'
approximation = 'finite'  # for w0: can be 'finite' (etBFRY) or 'truncated' (generalized gamma process w/ truncation)
sampler = 'layers'  # can be 'layers' or 'naive'

compute_distance = True  # you need distances if you are performing inference on w, n, u
reduce = False  # reduce graph G, locations x and weights w to active nodes. Usually not necessary.
check = False  # to check the log likelihood of the parameters sigma, c, t, tau given w and u in a grid around the
# original parameters

iter = 200000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'

# ----------------------------------
# L = 1000
# ----------------------------------

L = 1000
w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                            T=T, K=K, L=L)
ind1 = []
ind2 = []
for i in G.nodes:
    for j in G.adj[i]:
        if j >= i:
            ind1.append(i)
            ind2.append(j)
selfedge = [ind1[i] == ind2[i] for i in range(len(ind1))]

# compute distances
if compute_distance is True and gamma != 0:
    p_ij = aux.space_distance(x, gamma)
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)
if compute_distance is True and gamma == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)

# compute auxiliary variables and quantities
z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u = tp.tpoissrnd(z * w0)
sum_n = np.array(lil_matrix.sum(n, axis=0) + np.transpose(lil_matrix.sum(n, axis=1)))[0]
log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)

start2 = time.time()
output2 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                    plot=False,
                    sigma=True, c=True, t=True, w0=True, n=True, u=True,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u,
                    ind1=ind1, ind2=ind2, selfedge=selfedge)
end2 = time.time()
print('minutes to produce the sample (chain 1 rand init): ', round((end2 - start2) / 60, 2))

plt.figure()
w_est = output2[0]
deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(iter)]
plt.plot(biggest_w_est)
biggest_w = w[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand7/w0_trace_chain1000')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
    for j in range(size)]
print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)])/L)
true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
deg = np.array(list(dict(G.degree()).values()))
size = len(deg)
num = 50
sort_ind = np.argsort(deg)
ind_big1 = sort_ind[range(size - num, size)]
big_w = w[ind_big1]
emp_ci_big = []
for i in range(num):
    emp_ci_big.append(emp0_ci_95[ind_big1[i]])
plt.subplot(1, 3, 1)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_big[i][0], emp_ci_big[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, big_w[i], color='navy', marker='o', markersize=5)
plt.ylabel('w')
# smallest deg nodes
zero_deg = sum(deg == 0)
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 2)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
# zero deg nodes
zero_deg = 0
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 3)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.savefig('images/all_rand7/w0_CI_chain1000')
plt.close()

# ----------------
# L = 2000
# ----------------

L = 2000
w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                            T=T, K=K, L=L)
ind1 = []
ind2 = []
for i in G.nodes:
    for j in G.adj[i]:
        if j >= i:
            ind1.append(i)
            ind2.append(j)
selfedge = [ind1[i] == ind2[i] for i in range(len(ind1))]
# compute distances
if compute_distance is True and gamma != 0:
    p_ij = aux.space_distance(x, gamma)
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)
if compute_distance is True and gamma == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)
# compute auxiliary variables and quantities
z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u = tp.tpoissrnd(z * w0)
sum_n = np.array(lil_matrix.sum(n, axis=0) + np.transpose(lil_matrix.sum(n, axis=1)))[0]
log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)

start3 = time.time()
output3 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                    plot=False,
                    sigma=True, c=True, t=True, w0=True,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
end3 = time.time()
print('minutes to produce the sample (chain 2 rand init): ', round((end3 - start3) / 60, 2))

plt.figure()
w_est = output3[0]
deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(iter)]
plt.plot(biggest_w_est)
biggest_w = w[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand7/w0_trace_chain2000')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
    for j in range(size)]
print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)])/L)
true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
deg = np.array(list(dict(G.degree()).values()))
size = len(deg)
num = 50
sort_ind = np.argsort(deg)
ind_big1 = sort_ind[range(size - num, size)]
big_w = w[ind_big1]
emp_ci_big = []
for i in range(num):
    emp_ci_big.append(emp0_ci_95[ind_big1[i]])
plt.subplot(1, 3, 1)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_big[i][0], emp_ci_big[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, big_w[i], color='navy', marker='o', markersize=5)
plt.ylabel('w')
# smallest deg nodes
zero_deg = sum(deg == 0)
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 2)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
# zero deg nodes
zero_deg = 0
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 3)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.savefig('images/all_rand7/w0_CI_chain2000')
plt.close()


# ----------------
# L = 3000
# ----------------

L = 3000
w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                            T=T, K=K, L=L)
ind1 = []
ind2 = []
for i in G.nodes:
    for j in G.adj[i]:
        if j >= i:
            ind1.append(i)
            ind2.append(j)
selfedge = [ind1[i] == ind2[i] for i in range(len(ind1))]
# compute distances
if compute_distance is True and gamma != 0:
    p_ij = aux.space_distance(x, gamma)
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)
if compute_distance is True and gamma == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij, ind1, ind2, selfedge)
# compute auxiliary variables and quantities
z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u = tp.tpoissrnd(z * w0)
sum_n = np.array(lil_matrix.sum(n, axis=0) + np.transpose(lil_matrix.sum(n, axis=1)))[0]
log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)

start4 = time.time()
output4 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                    plot=False,
                    sigma=True, c=True, t=True, w0=True,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
end4 = time.time()
print('minutes to produce the sample (chain 3 rand init): ', round((end3 - start3) / 60, 2))

plt.figure()
w_est = output4[0]
deg = np.array(list(dict(G.degree()).values()))
biggest_deg = np.argsort(deg)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(iter)]
plt.plot(biggest_w_est)
biggest_w = w[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand7/w0_trace_chain3000')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
    for j in range(size)]
print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)])/L)
true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
deg = np.array(list(dict(G.degree()).values()))
size = len(deg)
num = 50
sort_ind = np.argsort(deg)
ind_big1 = sort_ind[range(size - num, size)]
big_w = w[ind_big1]
emp_ci_big = []
for i in range(num):
    emp_ci_big.append(emp0_ci_95[ind_big1[i]])
plt.subplot(1, 3, 1)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_big[i][0], emp_ci_big[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, big_w[i], color='navy', marker='o', markersize=5)
plt.ylabel('w')
# smallest deg nodes
zero_deg = sum(deg == 0)
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 2)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
# zero deg nodes
zero_deg = 0
ind_small = sort_ind[range(zero_deg, zero_deg + num)]
small_w = w[ind_small]
emp_ci_small = []
for i in range(num):
    emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
plt.subplot(1, 3, 3)
for i in range(num):
    plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
             linestyle='-', linewidth=2)
    plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
plt.ylabel('log w')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.savefig('images/all_rand7/w0_CI_chain3000')
plt.close()


# ----------------
# Traceplots
# ----------------

plt.figure()
plt.plot(output2[10], color='cornflowerblue', label='L=1k')
plt.plot(output3[10], color='navy', label='L=2k')
plt.plot(output4[10], color='blue', label='L=3k')
plt.axhline(y=log_post, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('log_post')
plt.legend()
plt.savefig('images/all_rand7/log_post')
plt.close()

plt.figure()
plt.plot(output2[3], color='cornflowerblue', label='L=1k')
plt.plot(output3[3], color='navy', label='L=2k')
plt.plot(output4[3], color='blue', label='L=3k')
plt.axhline(y=sigma, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/all_rand7/sigma')
plt.close()

plt.figure()
plt.plot(output2[4], color='cornflowerblue', label='L=1k')
plt.plot(output3[4], color='navy', label='L=2k')
plt.plot(output4[4], color='blue', label='L=3k')
plt.axhline(y=c, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/all_rand7/c')
plt.close()

plt.figure()
plt.plot(output2[5], color='cornflowerblue', label='L=1k')
plt.plot(output3[5], color='navy', label='L=2k')
plt.plot(output4[5], color='blue', label='L=3k')
plt.axhline(y=t, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/all_rand7/t')
plt.close()