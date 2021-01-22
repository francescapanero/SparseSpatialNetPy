import utils.MCMCNew_fast as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew_fast as aux
import utils.UpdatesNew_fast as up
import numpy as np
# import pymc3 as pm3
import matplotlib.pyplot as plt
import scipy
import pickle

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 2  # exponent distance in the link probability
L_x = 1  # space threshold: [0, L_x]

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs

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

save_every = 1000  # save output every save_every iterations. Must be multiple of 25

# # ----------------------------------
# # L = 1000
# # ----------------------------------

with open('data_outputs/output1_all_rand8.pickle', 'rb') as f:
    output1 = pickle.load(f)

# L1 = 1000
# w, w0, beta, x, G, L, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
#                                             T=T, K=K, L=L1)
# ind1 = []
# ind2 = []
# for i in G.nodes:
#     for j in G.adj[i]:
#         if j >= i:
#             ind1.append(i)
#             ind2.append(j)
# selfedge = [ind1[i] == ind2[i] for i in range(len(ind1))]
# 
# # compute distances
# if compute_distance is True and gamma != 0:
#     p_ij = aux.space_distance(x, gamma)
#     n = up.update_n(w, G, L, p_ij, ind1, ind2, selfedge)
# if compute_distance is True and gamma == 0:
#     p_ij = np.ones((L, L))
#     n = up.update_n(w, G, L, p_ij, ind1, ind2, selfedge)
# 
# # compute auxiliary variables and quantities
# z = (L * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
#             (L * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
# u = tp.tpoissrnd(z * w0)
# sum_n = np.array(lil_matrix.sum(n, axis=0) + np.transpose(lil_matrix.sum(n, axis=1)))[0]
# log_post2 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)
# 
# with open('w_all_rand8.pickle', 'wb') as f:
#     pickle.dump(w, f)
# 
# with open('x_all_rand8.pickle', 'wb') as f:
#     pickle.dump(x, f)
# 
# with open('n_all_rand8.pickle', 'wb') as f:
#     pickle.dump(n, f)
# 
# with open('u_all_rand8.pickle', 'wb') as f:
#     pickle.dump(u, f)
# 
# with open('G_all_rand8.pickle', 'wb') as f:
#     pickle.dump(G, f)
# 
# start2 = time.time()
# output2 = mcmc.MCMC(prior, G, gamma, L, iter, nburn, p_ij=p_ij,
#                     w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
#                     plot=False,
#                     sigma=True, c=True, t=True, w0=True, n=True, u=True,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u,
#                     save_every=save_every)
# end2 = time.time()
# print('minutes to produce the sample (chain 1 rand init): ', round((end2 - start2) / 60, 2))
# 
# with open('output2_all_rand8.pickle', 'wb') as f:
#     pickle.dump(output2, f)
# 
# plt.figure()
# w_est = output2[0]
# deg = np.array(list(dict(G.degree()).values()))
# biggest_deg = np.argsort(deg)[-1]
# biggest_w_est = [w_est[i][biggest_deg] for i in range(int((iter+save_every)/save_every))]
# plt.plot([i for i in range(0, iter+save_every, save_every)], biggest_w_est)
# biggest_w = w[biggest_deg]
# plt.axhline(y=biggest_w, label='true')
# plt.xlabel('iter')
# plt.ylabel('highest degree w')
# plt.legend()
# plt.savefig('images/all_rand8/w0_trace_chain1')
# plt.close()
# # plot empirical 95% ci for highest and lowest degrees nodes
# plt.figure()
# w_est_fin = [w_est[i] for i in range(int((nburn+save_every)/save_every), int((iter+save_every)/save_every))]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int((iter+save_every)/save_every) -
#                                                                   int((nburn+save_every)/save_every))],
#                                   prob=[0.025, 0.975]) for j in range(L)]
# print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(L)])/L1)
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(L)]
# print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
# deg = np.array(list(dict(G.degree()).values()))
# L = len(deg)
# num = 50
# sort_ind = np.argsort(deg)
# ind_big1 = sort_ind[range(L - num, L)]
# big_w = w[ind_big1]
# emp_ci_big = []
# for i in range(num):
#     emp_ci_big.append(emp0_ci_95[ind_big1[i]])
# plt.subplot(1, 3, 1)
# for i in range(num):
#     plt.plot((i + 1, i + 1), (emp_ci_big[i][0], emp_ci_big[i][1]), color='cornflowerblue',
#              linestyle='-', linewidth=2)
#     plt.plot(i + 1, big_w[i], color='navy', marker='o', markersize=5)
# plt.ylabel('w')
# # smallest deg nodes
# zero_deg = sum(deg == 0)
# ind_small = sort_ind[range(zero_deg, zero_deg + num)]
# small_w = w[ind_small]
# emp_ci_small = []
# for i in range(num):
#     emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
# plt.subplot(1, 3, 2)
# for i in range(num):
#     plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
#              linestyle='-', linewidth=2)
#     plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
# plt.ylabel('log w')
# # zero deg nodes
# zero_deg = 0
# ind_small = sort_ind[range(zero_deg, zero_deg + num)]
# small_w = w[ind_small]
# emp_ci_small = []
# for i in range(num):
#     emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
# plt.subplot(1, 3, 3)
# for i in range(num):
#     plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
#              linestyle='-', linewidth=2)
#     plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
# plt.ylabel('log w')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
# plt.savefig('images/all_rand8/w0_CI_chain1')
# plt.close()

# ----------------
# L = 3000
# ----------------

L2 = 3000

w2, w02, beta2, x2, G2, L2, deg2 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
                                                   T=T, K=K, L=L2)

# compute distances
if compute_distance is True and gamma != 0:
    p_ij2 = aux.space_distance(x2, gamma)
    n2 = up.update_n(w2, G2, L2, p_ij2)
if compute_distance is True and gamma == 0:
    p_ij2 = np.ones((L2, L2))
    n2 = up.update_n(w2, G2, L2, p_ij2)

# compute auxiliary variables and quantities
z2 = (L2 * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (L2 * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u2 = tp.tpoissrnd(z2 * w02)
sum_n2 = np.array(lil_matrix.sum(n2, axis=0) + np.transpose(lil_matrix.sum(n2, axis=1)))[0]
log_post2 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w2, w02, beta2, n2, u2, p_ij2, a_t, b_t, gamma, sum_n2)

with open('data_outputs/w2_all_rand9.pickle', 'wb') as f:
    pickle.dump(w2, f)

with open('data_outputs/x2_all_rand9.pickle', 'wb') as f:
    pickle.dump(x2, f)

with open('data_outputs/n2_all_rand9.pickle', 'wb') as f:
    pickle.dump(n2, f)

with open('data_outputs/u2_all_rand9.pickle', 'wb') as f:
    pickle.dump(u2, f)

with open('data_outputs/G2_all_rand9.pickle', 'wb') as f:
    pickle.dump(G2, f)

# G2 = G.__class__()
# G2.add_nodes_from(G)
# G2.add_nodes_from(range(L1, L2))
# G2.add_edges_from(G.edges)
# 
# x = np.concatenate((x, L_x * np.random.rand(L2-L1)))
# # compute distances
# if compute_distance is True and gamma != 0:
#     p_ij = aux.space_distance(x, gamma)
# if compute_distance is True and gamma == 0:
#     p_ij = np.ones((L, L))
# n2 = lil_matrix((L2, L2))
# n2[0:L1, 0:L1] = n
# sum_n = np.array(lil_matrix.sum(n2, axis=0) + np.transpose(lil_matrix.sum(n2, axis=1)))[0]

start2 = time.time()
output2 = mcmc.MCMC(prior, G2, gamma, L2, iter, nburn, p_ij=p_ij2,
                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                    plot=False,
                    sigma=True, c=True, t=True, w0=True, n=True, u=True,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w02, w_true=w2, beta_true=beta2, n_true=n2, u_true=u2,
                    save_every=save_every)
end2 = time.time()
print('minutes to produce the sample (chain 2 rand init): ', round((end2 - start2) / 60, 2))

with open('data_outputs/output2_all_rand9.pickle', 'wb') as f:
    pickle.dump(output2, f)

plt.figure()
w_est = output2[0]
biggest_deg = np.argsort(deg2)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(int(iter/save_every))]
plt.plot(biggest_w_est)
biggest_w = w2[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand9/w0_trace_chain2')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(int(nburn/save_every), int(iter/save_every))]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int(iter/save_every) - int(nburn/save_every))], prob=[0.025, 0.975])
    for j in range(L2)]
print(sum([emp0_ci_95[i][0] <= w2[i] <= emp0_ci_95[i][1] for i in range(L2)])/L2)
true0_in_ci = [emp0_ci_95[i][0] <= w2[i] <= emp0_ci_95[i][1] for i in range(L2)]
print('posterior coverage of true w in chain 2 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
deg = np.array(list(dict(G2.degree()).values()))
L = len(deg)
num = 50
sort_ind = np.argsort(deg)
ind_big1 = sort_ind[range(L2 - num, L2)]
big_w = w2[ind_big1]
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
small_w = w2[ind_small]
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
small_w = w2[ind_small]
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
plt.savefig('images/all_rand9/w0_CI_chain2')
plt.close()


# ----------------
# L = 5000
# ----------------

L3 = 5000

w3, w03, beta3, x3, G3, L3, deg3 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
                                                T=T, K=K, L=L3)

# compute distances
if compute_distance is True and gamma != 0:
    p_ij3 = aux.space_distance(x3, gamma)
    n3 = up.update_n(w3, G3, L3, p_ij3)
if compute_distance is True and gamma == 0:
    p_ij3 = np.ones((L3, L3))
    n3 = up.update_n(w3, G3, L3, p_ij3)

# compute auxiliary variables and quantities
z3 = (L3 * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (L3 * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u3 = tp.tpoissrnd(z3 * w03)
sum_n3 = np.array(lil_matrix.sum(n3, axis=0) + np.transpose(lil_matrix.sum(n3, axis=1)))[0]
log_post3 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w3, w03, beta3, n3, u3, p_ij3, a_t, b_t, gamma, sum_n3)

with open('data_outputs/w3_all_rand9.pickle', 'wb') as f:
    pickle.dump(w3, f)

with open('data_outputs/x3_all_rand9.pickle', 'wb') as f:
    pickle.dump(x3, f)

with open('data_outputs/n3_all_rand9.pickle', 'wb') as f:
    pickle.dump(n3, f)

with open('data_outputs/u3_all_rand9.pickle', 'wb') as f:
    pickle.dump(u3, f)

with open('data_outputs/G3_all_rand9.pickle', 'wb') as f:
    pickle.dump(G3, f)

# G3 = G.__class__()
# G3.add_nodes_from(G)
# G3.add_nodes_from(range(L2, L3))
# G3.add_edges_from(G.edges)
#
# x = np.concatenate((x, L_x * np.random.rand(L3-L2)))
#
# # compute distances
# if compute_distance is True and gamma != 0:
#     p_ij = aux.space_distance(x, gamma)
# if compute_distance is True and gamma == 0:
#     p_ij = np.ones((L, L))
# n3 = lil_matrix((L3, L3))
# n3[0:L1, 0:L1] = n
# sum_n = np.array(lil_matrix.sum(n3, axis=0) + np.transpose(lil_matrix.sum(n3, axis=1)))[0]

start3 = time.time()
output3 = mcmc.MCMC(prior, G3, gamma, L3, iter, nburn, p_ij=p_ij3,
                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                    plot=False,
                    sigma=True, c=True, t=True, w0=True, n=True, u=True,
                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                    w0_true=w03, w_true=w3, beta_true=beta3, n_true=n3, u_true=u3,
                    save_every=save_every)
end3 = time.time()
print('minutes to produce the sample (chain 3 rand init): ', round((end3 - start3) / 60, 2))

with open('data_outputs/output3_all_rand9.pickle', 'wb') as f:
    pickle.dump(output3, f)

plt.figure()
w_est = output3[0]
biggest_deg = np.argsort(deg)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(int(iter/save_every))]
plt.plot(biggest_w_est)
biggest_w = w3[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand9/w0_trace_chain3')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(int(nburn/save_every), int(iter/save_every))]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int(iter/save_every) - int(nburn/save_every))], prob=[0.025, 0.975])
    for j in range(L)]
print(sum([emp0_ci_95[i][0] <= w3[i] <= emp0_ci_95[i][1] for i in range(L)])/L)
true0_in_ci = [emp0_ci_95[i][0] <= w3[i] <= emp0_ci_95[i][1] for i in range(L)]
print('posterior coverage of true w in chain 3 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
deg = np.array(list(dict(G3.degree()).values()))
L = len(deg)
num = 50
sort_ind = np.argsort(deg)
ind_big1 = sort_ind[range(L - num, L)]
big_w = w3[ind_big1]
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
small_w = w3[ind_small]
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
small_w = w3[ind_small]
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
plt.savefig('images/all_rand9/w0_CI_chain3')
plt.close()


# ----------------
# Traceplots
# ----------------

# plt.figure()
# plt.plot([i for i in range(0, iter+save_every, save_every)], output2[10], color='cornflowerblue')
# plt.axhline(y=log_post1, label='true', color='r')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log_post')
# plt.savefig('images/all_rand9/log_post1')
# plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[10], color='cornflowerblue')
plt.axhline(y=log_post2, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('log_post')
plt.savefig('images/all_rand9/log_post2')
plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output3[10], color='cornflowerblue')
plt.axhline(y=log_post3, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('log_post')
plt.savefig('images/all_rand9/log_post3')
plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[3], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[3], color='navy', label='L=3k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output3[3], color='blue', label='L=5k')
plt.axhline(y=sigma, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/all_rand9/sigma')
plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[4], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[4], color='navy', label='L=3k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output3[4], color='blue', label='L=5k')
plt.axhline(y=c, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/all_rand9/c')
plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[5], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[4], color='navy', label='L=3k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output3[5], color='blue', label='L=5k')
plt.axhline(y=t, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/all_rand9/t')
plt.close()