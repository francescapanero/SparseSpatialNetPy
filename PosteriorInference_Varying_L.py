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
from itertools import compress
from scipy.sparse import csr_matrix

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

iter = 500000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'

save_every = 2500  # save output every save_every iterations. Must be multiple of 25

# # # ----------------------------------
# # # L = 1000
# # # ----------------------------------
#
# # with open('data_outputs/output1_all_rand8.pickle', 'rb') as f:
# #     output1 = pickle.load(f)
#
# L1 = 1000
#
# # w, w0, beta, x, G, L, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
# #                                             T=T, K=K, L=L1)
# #
# # # compute distances
# # if compute_distance is True and gamma != 0:
# #     p_ij = aux.space_distance(x, gamma)
# #     n = up.update_n(w, G, L, p_ij)
# # if compute_distance is True and gamma == 0:
# #     p_ij = np.ones((L, L))
# #     n = up.update_n(w, G, L, p_ij)
# #
# # # compute auxiliary variables and quantities
# # z = (L * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
# #             (L * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
# # u = tp.tpoissrnd(z * w0)
# # sum_n = np.array(lil_matrix.sum(n, axis=0) + np.transpose(lil_matrix.sum(n, axis=1)))[0]
# # log_post2 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)

# with open('data_outputs/w1_all_rand8.pickle', 'rb') as f:
#     w1 = pickle.load(f)
#
# with open('data_outputs/x1_all_rand8.pickle', 'rb') as f:
#     x1 = pickle.load(f)
#
# with open('data_outputs/n1_all_rand8.pickle', 'rb') as f:
#     n1 = pickle.load(f)
#
# with open('data_outputs/u1_all_rand8.pickle', 'rb') as f:
#     u1 = pickle.load(f)
#
# with open('data_outputs/G1_all_rand8.pickle', 'rb') as f:
#     G1 = pickle.load(f)
#
# sum_n1 = np.array(lil_matrix.sum(n1, axis=0) + np.transpose(lil_matrix.sum(n1, axis=1)))[0]
# p_ij1 = aux.space_distance(x1, gamma)
# w01 = w1
# beta1 = np.ones(L1)
# log_post1 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w1, w01, beta1, n1, u1, p_ij1, a_t, b_t, gamma, sum_n1)

# # start1 = time.time()
# # output1 = mcmc.MCMC(prior, G1, gamma, L1, iter, nburn, p_ij=p_ij1,
# #                     w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
# #                     plot=False,
# #                     sigma=True, c=True, t=True, w0=True, n=True, u=True,
# #                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
# #                     w0_true=w01, w_true=w1, beta_true=beta1, n_true=n1, u_true=u1,
# #                     save_every=save_every)
# # end1 = time.time()
# # print('minutes to produce the sample (chain 1 rand init): ', round((end1 - start1) / 60, 2))
#
with open('data_outputs/output1_all_rand10.pickle', 'rb') as f:
    output1 = pickle.load(f)
#
# # plt.figure()
# # w_est = output1[0]
# # deg = np.array(list(dict(G1.degree()).values()))
# # biggest_deg = np.argsort(deg)[-1]
# # biggest_w_est = [w_est[i][biggest_deg] for i in range(int((iter+save_every)/save_every))]
# # plt.plot([i for i in range(0, iter+save_every, save_every)], biggest_w_est)
# # biggest_w = w1[biggest_deg]
# # plt.axhline(y=biggest_w, label='true')
# # plt.xlabel('iter')
# # plt.ylabel('highest degree w')
# # plt.legend()
# # plt.savefig('images/all_rand10/w0_trace_chain1')
# # plt.close()
# # # plot empirical 95% ci for highest and lowest degrees nodes
# # plt.figure()
# # w_est_fin = [w_est[i] for i in range(int((nburn+save_every)/save_every), int((iter+save_every)/save_every))]
# # emp0_ci_95 = [
# #     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int((iter+save_every)/save_every) -
# #                                                                   int((nburn+save_every)/save_every))],
# #                                   prob=[0.025, 0.975]) for j in range(L1)]
# # print(sum([emp0_ci_95[i][0] <= w1[i] <= emp0_ci_95[i][1] for i in range(L1)])/L1)
# # true0_in_ci = [emp0_ci_95[i][0] <= w1[i] <= emp0_ci_95[i][1] for i in range(L1)]
# # print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
# # deg = np.array(list(dict(G1.degree()).values()))
# # L = len(deg)
# # num = 50
# # sort_ind = np.argsort(deg)
# # ind_big1 = sort_ind[range(L - num, L)]
# # big_w = w1[ind_big1]
# # emp_ci_big = []
# # for i in range(num):
# #     emp_ci_big.append(emp0_ci_95[ind_big1[i]])
# # plt.subplot(1, 3, 1)
# # for i in range(num):
# #     plt.plot((i + 1, i + 1), (emp_ci_big[i][0], emp_ci_big[i][1]), color='cornflowerblue',
# #              linestyle='-', linewidth=2)
# #     plt.plot(i + 1, big_w[i], color='navy', marker='o', markersize=5)
# # plt.ylabel('w')
# # # smallest deg nodes
# # zero_deg = sum(deg == 0)
# # ind_small = sort_ind[range(zero_deg, zero_deg + num)]
# # small_w = w1[ind_small]
# # emp_ci_small = []
# # for i in range(num):
# #     emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
# # plt.subplot(1, 3, 2)
# # for i in range(num):
# #     plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
# #              linestyle='-', linewidth=2)
# #     plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
# # plt.ylabel('log w')
# # # zero deg nodes
# # zero_deg = 0
# # ind_small = sort_ind[range(zero_deg, zero_deg + num)]
# # small_w = w1[ind_small]
# # emp_ci_small = []
# # for i in range(num):
# #     emp_ci_small.append(np.log(emp0_ci_95[ind_small[i]]))
# # plt.subplot(1, 3, 3)
# # for i in range(num):
# #     plt.plot((i + 1, i + 1), (emp_ci_small[i][0], emp_ci_small[i][1]), color='cornflowerblue',
# #              linestyle='-', linewidth=2)
# #     plt.plot(i + 1, np.log(small_w[i]), color='navy', marker='o', markersize=5)
# # plt.ylabel('log w')
# # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
# # plt.savefig('images/all_rand10/w0_CI_chain1')
# # plt.close()

# # ----------------
# # L = 3000
# # ----------------
#
L2 = 4000
#
with open('data_outputs/n2_all_rand10.pickle', 'rb') as f:
    n2 = pickle.load(f)
#
with open('data_outputs/w2_all_rand10.pickle', 'rb') as f:
    w2 = pickle.load(f)
#
with open('data_outputs/G2_all_rand10.pickle', 'rb') as f:
    G2 = pickle.load(f)
#
with open('data_outputs/x2_all_rand10.pickle', 'rb') as f:
    x2 = pickle.load(f)
#
with open('data_outputs/u2_all_rand10.pickle', 'rb') as f:
    u2 = pickle.load(f)
#
#sum_n2 = np.array(csr_matrix.sum(n2, axis=0) + np.transpose(csr_matrix.sum(n2, axis=1)))[0]
#p_ij2 = aux.space_distance(x2, gamma)
w02 = w2
beta2 = np.ones(L2)
#log_post2 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w2, w02, beta2, n2, u2, p_ij2, a_t, b_t, gamma, sum_n2)
#
#
#w2, w02, beta2, x2, G2, L2, deg2 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
#                                                    T=T, K=K, L=L2)

ind = {k: [] for k in G2.nodes}
for i in G2.nodes:
    for j in G2.adj[i]:
        if j > i:
            ind[i].append(j)
selfedge = [i in ind[i] for i in G2.nodes]
selfedge = list(compress(G2.nodes, selfedge))

# compute distances
if compute_distance is True and gamma != 0:
    p_ij2 = aux.space_distance(x2, gamma)
   # n2 = up.update_n(w2, G2, L2, p_ij2, ind, selfedge)
if compute_distance is True and gamma == 0:
    p_ij2 = np.ones((L2, L2))
    #n2 = up.update_n(w2, G2, L2, p_ij2, ind, selfedge)

# compute auxiliary variables and quantities
z2 = (L2 * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (L2 * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
#u2 = tp.tpoissrnd(z2 * w02)
sum_n2 = np.array(csr_matrix.sum(n2, axis=0) + np.transpose(csr_matrix.sum(n2, axis=1)))[0]
log_post2 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w2, w02, beta2, n2, u2, p_ij2, a_t, b_t, gamma, sum_n2)

#with open('data_outputs/w2_all_rand10.pickle', 'wb') as f:
#    pickle.dump(w2, f)

#with open('data_outputs/x2_all_rand10.pickle', 'wb') as f:
#    pickle.dump(x2, f)

#with open('data_outputs/n2_all_rand10.pickle', 'wb') as f:
#    pickle.dump(n2, f)

#with open('data_outputs/u2_all_rand10.pickle', 'wb') as f:
#    pickle.dump(u2, f)

#with open('data_outputs/G2_all_rand10.pickle', 'wb') as f:
#    pickle.dump(G2, f)

#start2 = time.time()
#output2 = mcmc.MCMC(prior, G2, gamma, L2, iter, nburn, p_ij=p_ij2,
#                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
#                    plot=False,
#                    sigma=True, c=True, t=True, w0=True, n=True, u=True,
#                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                    w0_true=w02, w_true=w2, beta_true=beta2, n_true=n2, u_true=u2,
#                    save_every=save_every, ind=ind, selfedge=selfedge)
#end2 = time.time()
#print('minutes to produce the sample (chain 2 rand init): ', round((end2 - start2) / 60, 2))

with open('data_outputs/output2_all_rand10.pickle', 'rb') as f:
    output2=pickle.load(f)

plt.figure()
deg2 = np.array(list(dict(G2.degree()).values()))
w_est = output2[0]
biggest_deg = np.argsort(deg2)[-1]
biggest_w_est = [w_est[i][biggest_deg] for i in range(int(iter/save_every))]
plt.plot(biggest_w_est)
biggest_w = w2[biggest_deg]
plt.axhline(y=biggest_w, label='true')
plt.xlabel('iter')
plt.ylabel('highest degree w')
plt.legend()
plt.savefig('images/all_rand10/w0_trace_chain2')
plt.close()
# # plot empirical 95% ci for highest and lowest degrees nodes
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
# # smallest deg nodes
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
# # zero deg nodes
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
plt.savefig('images/all_rand10/w0_CI_chain2')
plt.close()


# ----------------
# L = 4000
# ----------------

# L3 = 4000

# w3, w03, beta3, x3, G3, L3, deg3 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
#                                                 T=T, K=K, L=L3)
#
# # compute distances
# if compute_distance is True and gamma != 0:
#     p_ij3 = aux.space_distance(x3, gamma)
#     n3 = up.update_n(w3, G3, L3, p_ij3)
# if compute_distance is True and gamma == 0:
#     p_ij3 = np.ones((L3, L3))
#     n3 = up.update_n(w3, G3, L3, p_ij3)
#
# # compute auxiliary variables and quantities
# z3 = (L3 * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
#             (L3 * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
# u3 = tp.tpoissrnd(z3 * w03)
# sum_n3 = np.array(lil_matrix.sum(n3, axis=0) + np.transpose(lil_matrix.sum(n3, axis=1)))[0]
# log_post3 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w3, w03, beta3, n3, u3, p_ij3, a_t, b_t, gamma, sum_n3)
#
# with open('data_outputs/w3_all_rand10.pickle', 'wb') as f:
#     pickle.dump(w3, f)
#
# with open('data_outputs/x3_all_rand10.pickle', 'wb') as f:
#     pickle.dump(x3, f)
#
# with open('data_outputs/n3_all_rand10.pickle', 'wb') as f:
#     pickle.dump(n3, f)
#
# with open('data_outputs/u3_all_rand10.pickle', 'wb') as f:
#     pickle.dump(u3, f)
#
# with open('data_outputs/G3_all_rand10.pickle', 'wb') as f:
#     pickle.dump(G3, f)

# with open('data_outputs/n3_all_rand10.pickle', 'rb') as f:
#     n3 = pickle.load(f)
#
# with open('data_outputs/w3_all_rand10.pickle', 'rb') as f:
#     w3 = pickle.load(f)
#
# with open('data_outputs/G3_all_rand10.pickle', 'rb') as f:
#     G3 = pickle.load(f)
#
# with open('data_outputs/x3_all_rand10.pickle', 'rb') as f:
#     x3 = pickle.load(f)
#
# with open('data_outputs/u3_all_rand10.pickle', 'rb') as f:
#     u3 = pickle.load(f)
#
# sum_n3 = np.array(lil_matrix.sum(n3, axis=0) + np.transpose(lil_matrix.sum(n3, axis=1)))[0]
# p_ij3 = aux.space_distance(x3, gamma)
# w03 = w3
# beta3 = np.ones(L3)
# log_post3 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w3, w03, beta3, n3, u3, p_ij3, a_t, b_t, gamma, sum_n3)
#
# start3 = time.time()
# output3 = mcmc.MCMC(prior, G3, gamma, L3, iter, nburn, p_ij=p_ij3,
#                     w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
#                     plot=False,
#                     sigma=True, c=True, t=True, w0=True, n=True, u=True,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w03, w_true=w3, beta_true=beta3, n_true=n3, u_true=u3,
#                     save_every=save_every)
# end3 = time.time()
# print('minutes to produce the sample (chain 3 rand init): ', round((end3 - start3) / 60, 2))
#
# with open('data_outputs/output3_all_rand10.pickle', 'wb') as f:
#     output3 = pickle.dump(output3, f)
#
# plt.figure()
# deg = np.array(list(dict(G3.degree()).values()))
# w_est = output3[0]
# biggest_deg = np.argsort(deg)[-1]
# biggest_w_est = [w_est[i][biggest_deg] for i in range(int(iter/save_every))]
# plt.plot(biggest_w_est)
# biggest_w = w3[biggest_deg]
# plt.axhline(y=biggest_w, label='true')
# plt.xlabel('iter')
# plt.ylabel('highest degree w')
# plt.legend()
# plt.savefig('images/all_rand10/w0_trace_chain3')
# plt.close()
# # plot empirical 95% ci for highest and lowest degrees nodes
# plt.figure()
# w_est_fin = [w_est[i] for i in range(int(nburn/save_every), int(iter/save_every))]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int(iter/save_every) - int(nburn/save_every))], prob=[0.025, 0.975])
#     for j in range(L3)]
# print(sum([emp0_ci_95[i][0] <= w3[i] <= emp0_ci_95[i][1] for i in range(L3)])/L3)
# true0_in_ci = [emp0_ci_95[i][0] <= w3[i] <= emp0_ci_95[i][1] for i in range(L3)]
# print('posterior coverage of true w in chain 3 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
# deg = np.array(list(dict(G3.degree()).values()))
# L = len(deg)
# num = 50
# sort_ind = np.argsort(deg)
# ind_big1 = sort_ind[range(L - num, L)]
# big_w = w3[ind_big1]
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
# small_w = w3[ind_small]
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
# small_w = w3[ind_small]
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
# plt.savefig('images/all_rand10/w0_CI_chain3')
# plt.close()


# # ----------------
# # L = 7000
# # ----------------
#
# L4 = 7000
#
# w4, w04, beta4, x4, G4, L4, deg4 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, L_x,
#                                                 T=T, K=K, L=L4)
#
# # compute distances
# if compute_distance is True and gamma != 0:
#     p_ij4 = aux.space_distance(x4, gamma)
#     n4 = up.update_n(w4, G4, L4, p_ij4)
# if compute_distance is True and gamma == 0:
#     p_ij4 = np.ones((L4, L4))
#     n4 = up.update_n(w4, G4, L4, p_ij4)
#
# # compute auxiliary variables and quantities
# z4 = (L4 * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
#             (L4 * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
# u4 = tp.tpoissrnd(z4 * w04)
# sum_n4 = np.array(lil_matrix.sum(n4, axis=0) + np.transpose(lil_matrix.sum(n4, axis=1)))[0]
# log_post4 = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w4, w04, beta4, n4, u4, p_ij4, a_t, b_t, gamma, sum_n4)
#
# with open('data_outputs/w4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(w4, f)
#
# with open('data_outputs/x4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(x4, f)
#
# with open('data_outputs/n4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(n4, f)
#
# with open('data_outputs/u4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(u4, f)
#
# with open('data_outputs/G4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(G4, f)
#
# start4 = time.time()
# output4 = mcmc.MCMC(prior, G4, gamma, L4, iter, nburn, p_ij=p_ij4,
#                     w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
#                     plot=False,
#                     sigma=True, c=True, t=True, w0=True, n=True, u=True,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w04, w_true=w4, beta_true=beta4, n_true=n4, u_true=u4,
#                     save_every=save_every)
# end4 = time.time()
# print('minutes to produce the sample (chain 4 rand init): ', round((end4 - start4) / 60, 2))
#
# with open('data_outputs/output4_all_rand10.pickle', 'wb') as f:
#     pickle.dump(output4, f)
#
# plt.figure()
# deg4 = np.array(list(dict(G4.degree()).values()))
# w_est = output4[0]
# biggest_deg = np.argsort(deg4)[-1]
# biggest_w_est = [w_est[i][biggest_deg] for i in range(int(iter/save_every))]
# plt.plot(biggest_w_est)
# biggest_w = w4[biggest_deg]
# plt.axhline(y=biggest_w, label='true')
# plt.xlabel('iter')
# plt.ylabel('highest degree w')
# plt.legend()
# plt.savefig('images/all_rand10/w0_trace_chain4')
# plt.close()
# # plot empirical 95% ci for highest and lowest degrees nodes
# plt.figure()
# w_est_fin = [w_est[i] for i in range(int(nburn/save_every), int(iter/save_every))]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(int(iter/save_every) - int(nburn/save_every))], prob=[0.025, 0.975])
#     for j in range(L4)]
# print(sum([emp0_ci_95[i][0] <= w4[i] <= emp0_ci_95[i][1] for i in range(L4)])/L4)
# true0_in_ci = [emp0_ci_95[i][0] <= w4[i] <= emp0_ci_95[i][1] for i in range(L4)]
# print('posterior coverage of true w in chain 2 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
# deg = np.array(list(dict(G4.degree()).values()))
# L = len(deg)
# num = 50
# sort_ind = np.argsort(deg)
# ind_big1 = sort_ind[range(L4 - num, L4)]
# big_w = w4[ind_big1]
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
# small_w = w4[ind_small]
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
# small_w = w4[ind_small]
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
# plt.savefig('images/all_rand10/w0_CI_chain4')
# plt.close()
#
#
# ----------------
# Traceplots
# ----------------

# plt.figure()
# plt.plot([i for i in range(0, iter+save_every, save_every)], output1[10], color='cornflowerblue')
# plt.axhline(y=log_post1, label='true', color='r')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log_post')
# plt.savefig('images/all_rand10/log_post1')
# plt.close()

plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[10], color='cornflowerblue')
plt.axhline(y=log_post2, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('log_post')
plt.savefig('images/all_rand10/log_post2')
plt.close()
#
# plt.figure()
# plt.plot([i for i in range(0, iter+save_every, save_every)], output3[10], color='cornflowerblue')
# plt.axhline(y=log_post3, label='true', color='r')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log_post')
# plt.savefig('images/all_rand10/log_post3')
# plt.close()
#
# plt.figure()
# plt.plot([i for i in range(0, iter+save_every, save_every)], output4[10], color='cornflowerblue')
# plt.axhline(y=log_post4, label='true', color='r')
# plt.legend()
# plt.xlabel('iter')
# plt.ylabel('log_post')
# plt.savefig('images/all_rand10/log_post4')
# plt.close()
#
plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[3], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[3], color='blue', label='L=4k')
# plt.plot([i for i in range(0, iter+save_every, save_every)], output3[3], color='blue', label='L=4k')
# # plt.plot([i for i in range(0, iter+save_every, save_every)], output4[3], color='navy', label='L=7k')
plt.axhline(y=sigma, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/all_rand10/sigma2')
plt.close()
#
plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[4], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[4], color='blue', label='L=4k')
# plt.plot([i for i in range(0, iter+save_every, save_every)], output3[4], color='navy', label='L=4k')
# # plt.plot([i for i in range(0, iter+save_every, save_every)], output4[4], color='blue', label='L=7k')
plt.axhline(y=c, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/all_rand10/c2')
plt.close()
#
plt.figure()
plt.plot([i for i in range(0, iter+save_every, save_every)], output1[5], color='cornflowerblue', label='L=1k')
plt.plot([i for i in range(0, iter+save_every, save_every)], output2[4], color='blue', label='L=4k')
# plt.plot([i for i in range(0, iter+save_every, save_every)], output3[5], color='navy', label='L=5k')
# # plt.plot([i for i in range(0, iter+save_every, save_every)], output4[4], color='blue', label='L=10k')
plt.axhline(y=t, label='true', color='r')
plt.legend()
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/all_rand10/t2')
plt.close()
