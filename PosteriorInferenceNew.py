import utils.MCMCNew as mcmc
from utils.GraphSamplerNew import *
import utils.TruncPois as tp
import utils.AuxiliaryNew as aux
import utils.UpdatesNew as up
import numpy as np
import pandas as pd
import pymc3 as pm3
import matplotlib.pyplot as plt
import scipy

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 0  # exponent distance in the link probability
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

# ----------------------
# SIMULATE DATA
# ----------------------

w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                            T=T, K=K, L=L)

# compute distances
if compute_distance is True and gamma != 0:
    p_ij = aux.space_distance(x, gamma)
    n = up.update_n(w, G, size, p_ij)
if compute_distance is True and gamma == 0:
    p_ij = np.ones((size, size))
    n = up.update_n(w, G, size, p_ij)

# compute auxiliary variables and quantities
z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
u = tp.tpoissrnd(z * w0)
log_post = aux.log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t)

if reduce is True:
    G, isol = aux.SimpleGraph(G)
    x_red = np.delete(x, isol)
    w_red = np.delete(w, isol)

if check is True:
    temp = aux.check_log_likel_params(prior, sigma, c, t, tau, w0, beta, u)
    log_lik_true = aux.log_likel_params(prior, sigma, c, t, tau, w0, beta, u)
    print('true log likel params = ', log_lik_true)
    print('log likel in max = ', temp[0])
    if prior == 'singlepl':
        print('params for max (sigma, c, t) = ', temp[1])
    if prior == 'doublepl':
        print('params for max (sigma, c, t, tau) = ', temp[1])
    if log_lik_true < temp[0]:
        print('the approximation is not working! Decrease T!')

# ---------------------------
# posterior inference
# ---------------------------

# ---------------------------
# parameters only
# ---------------------------

# iter = 100000
# nburn = int(iter * 0.25)
# sigma_sigma = 0.01
# sigma_c = 0.1
# sigma_t = 0.1
# sigma_tau = 0.01
#
# # inference on hyperparams (all together), with w0, beta, n, u fixed
# output1 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=True,
#                     hyperparams=True,
#                     sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
#                     sigma_init=sigma, c_init=c, tau_init=tau, t_init=t,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)
# output2 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
#                     hyperparams=True,
#                     sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
#                     sigma_init=0.2, c_init=4, tau_init=tau, t_init=t+20,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)
# output3 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
#                     hyperparams=True,
#                     sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
#                     sigma_init=0.5, c_init=1.5, tau_init=tau, t_init=t-30,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)
# output4 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=False,
#                     hyperparams=True,
#                     sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
#                     sigma_init=sigma+0.2, c_init=c+1, tau_init=tau, t_init=t+50,
#                     sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                     w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)

# plt.figure()
# plt.plot(output1[3], color='black')
# plt.plot(output2[3], color='cornflowerblue')
# plt.plot(output3[3], color='navy')
# plt.plot(output4[3], color='blue')
# plt.axhline(y=sigma, label='true', color='r')
# plt.xlabel('iter')
# plt.ylabel('sigma')
# plt.savefig('images/lower_threshold/sigma1')
# plt.figure()
# plt.plot(output1[4], color='black')
# plt.plot(output2[4], color='cornflowerblue')
# plt.plot(output3[4], color='navy')
# plt.plot(output4[4], color='blue')
# plt.axhline(y=c, label='true', color='r')
# plt.xlabel('iter')
# plt.ylabel('c')
# # plt.savefig('images/lower_threshold/c1')
# plt.figure()
# plt.plot(output1[5], color='black')
# plt.plot(output2[5], color='cornflowerblue')
# plt.plot(output3[5], color='navy')
# plt.plot(output4[5], color='blue')
# plt.axhline(y=t, label='true', color='r')
# plt.xlabel('iter')
# plt.ylabel('t')
# # plt.savefig('images/lower_threshold/t1')
# plt.figure()
# plt.plot(output1[9][iter-10000], color='black')
# plt.plot(output2[9][iter-10000], color='cornflowerblue')
# plt.plot(output3[9][iter-10000], color='navy')
# plt.plot(output4[9][iter-10000], color='blue')
# plt.axhline(y=log_post, label='true', color='r')
# plt.xlabel('iter')
# plt.ylabel('log posterior')
# # plt.savefig('images/lower_threshold/logpost1')
#
# lags = np.arange(1, 100)
# plt.subplot(1, 4, 1)
# plt.plot(lags, [pm3.autocorr(np.array(output1[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 2)
# plt.plot(lags, [pm3.autocorr(np.array(output2[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 3)
# plt.plot(lags, [pm3.autocorr(np.array(output3[5][nburn:iter]), l) for l in lags])
# plt.subplot(1, 4, 4)
# plt.plot(lags, [pm3.autocorr(np.array(output4[5][nburn:iter]), l) for l in lags])
#
# n = iter - nburn + 1
# W = (np.array(output2[3][nburn:nburn+iter]).std() ** 2 + np.array(output3[3][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[3][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[3][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[3][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[3][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic sigma: ", np.sqrt(var_theta/W))
# W = (np.array(output2[4][nburn:nburn+iter]).std() ** 2 + np.array(output3[4][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[4][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[4][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[4][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[4][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic c: ", np.sqrt(var_theta/W))
# W = (np.array(output2[5][nburn:nburn+iter]).std() ** 2 + np.array(output3[5][nburn:nburn+iter]).std() ** 2 +
#      np.array(output4[5][nburn:nburn+iter]).std() ** 2) / 3
# mean1 = np.array(output2[5][nburn:nburn+iter]).mean()
# mean2 = np.array(output3[5][nburn:nburn+iter]).mean()
# mean3 = np.array(output4[5][nburn:nburn+iter]).mean()
# mean = (mean1 + mean2 + mean3) / 3
# B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
# var_theta = (1 - 1/n) * W + 1 / n * B
# print("Gelmen-Rubin Diagnostic t: ", np.sqrt(var_theta/W))
#
# plt.figure()
# plt.acorr(output1[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output1[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output1[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
#
# plt.acorr(output2[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output2[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output2[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
#
# plt.acorr(output3[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output3[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.acorr(output3[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
#
# # ----------------
# # w only
# # ----------------

# iter = 200000
# nburn = int(iter * 0.25)
# epsilon = 0.01
# R = 5
# w_inference = 'HMC'
#
# output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
#                    w_inference=w_inference, epsilon=epsilon, R=R,
#                    a_t=a_t, b_t=b_t,
#                    plot=True,
#                    w0=True,
#                    # w0_init=w0,
#                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
#
# w_est = output[0]
# plt.figure()
# w_est_fin = [w_est[i] for i in range(nburn, iter)]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
# plt.savefig('/homes/panero/frappi/Project_CaronRousseau/SparseSpatialNetPy/images/slurm_attempt_trace')
# deg = np.array(list(dict(G.degree()).values()))
# size = len(deg)
# num = 50
# sort_ind = np.argsort(deg)
# ind_big1 = sort_ind[range(size - num, size)]
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
# plt.legend()
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
# plt.legend()
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
# plt.legend()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
# plt.savefig('/homes/panero/frappi/Project_CaronRousseau/SparseSpatialNetPy/images/slurm_attempt_CI')

# # df = pd.DataFrame(output[0])
# # lags = np.arange(1, 100)
# # fig, ax = plt.subplots()
# # ax.plot(lags, [pm3.autocorr(df[1], l) for l in lags])
# # _ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
# # plt.title('Autocorrelation Plot')
# # plt.show()

# -----------------------
# All together
# -----------------------

iter = 500000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'

output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij, plot=True, w_inference='HMC',
                   hyperparams=True,
                   sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
                   sigma_init=sigma, c_init=c, tau_init=tau, t_init=t, w0_init=w0, beta_init=beta,
                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u, log_post_true=log_post)