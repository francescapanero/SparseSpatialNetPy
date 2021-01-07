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

gamma = 0  # exponent distance in the link probability
size_x = 1  # space threshold: [0, size_x]

K = 100  # number of layers, for layers sampler
T = 0.000001  # threshold for simulations of weights from truncated infinite activity CRMs
L = 2000  # tot number of nodes in finite approx of weights simulations (exptiltBFRY)

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
sum_n = np.array(lil_matrix.sum(n[-1], axis=0) + np.transpose(lil_matrix.sum(n[-1], axis=1)))[0]
log_post = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n)

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
#
# iter = 100000
# nburn = int(iter * 0.25)
# epsilon = 0.01
# R = 5
# w_inference = 'HMC'
#
# output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
#                    w_inference=w_inference, epsilon=epsilon, R=R,
#                    a_t=a_t, b_t=b_t,
#                    plot=False,
#                    w0=True,
#                    # hyperparams=True,
#                    # w0_init=w0,
#                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)

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

# true init

iter = 500000
nburn = int(iter * 0.25)
sigma_sigma = 0.01
sigma_c = 0.1
sigma_t = 0.1
sigma_tau = 0.01
epsilon = 0.01
R = 5
w_inference = 'HMC'

# start = time.time()
# output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
#                    w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
#                    plot=False,
#                    sigma=True, c=True, t=True, w0=True,
#                    sigma_init=sigma, c_init=tau, t_init=t, tau_init=tau, w0_init=w0, w_init=w, beta_init=beta,
#                    u_init=u, n_init=n,
#                    sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
#                    w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
# end = time.time()
# print('minutes to produce the sample (true): ', round((end - start) / 60, 2))

start2 = time.time()
output2 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                   w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                   plot=False,
                   sigma=True, c=True, t=True, w0=True,
                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
end2 = time.time()
print('minutes to produce the sample (chain 1 rand): ', round((end2 - start2) / 60, 2))

# w_est = output2[0]
# w_est_fin = [w_est[i] for i in range(nburn, iter)]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')

start3 = time.time()
output3 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                   w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                   plot=False,
                   sigma=True, c=True, t=True, w0=True,
                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
end3 = time.time()
print('minutes to produce the sample (chain 2 rand): ', round((end3 - start3) / 60, 2))

# w_est = output3[0]
# w_est_fin = [w_est[i] for i in range(nburn, iter)]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')

start4 = time.time()
output4 = mcmc.MCMC(prior, G, gamma, size, iter, nburn, p_ij=p_ij,
                   w_inference=w_inference, epsilon=epsilon, R=R, a_t=a_t, b_t=b_t,
                   plot=False,
                   sigma=True, c=True, t=True, w0=True,
                   sigma_true=sigma, c_true=c, t_true=t, tau_true=tau,
                   w0_true=w0, w_true=w, beta_true=beta, n_true=n, u_true=u)
end4 = time.time()
print('minutes to produce the sample (chain 3 rand): ', round((end4 - start4) / 60, 2))

# w_est = output4[0]
# plt.figure()
# w_est_fin = [w_est[i] for i in range(nburn, iter)]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')


# # ----------------------------------
# # All together - TRUE INIT
# # -----------------------------------
#
# # traceplots
#
# plt.figure()
# plt.plot(output[10])
# plt.axhline(y=log_post, color='r')
# plt.xlabel('iter')
# plt.ylabel('log_post')
# plt.savefig('images/all_trueinit3/logpost')
# plt.close()
#
# plt.figure()
# sigma_est = output[3]
# plt.plot(sigma_est, color='blue')
# plt.axhline(y=sigma, label='true', color='r')
# plt.xlabel('iter')
# plt.ylabel('sigma')
# plt.savefig('images/all_trueinit3/sigma')
# plt.close()
#
# plt.figure()
# c_est = output[4]
# plt.plot(c_est, color='blue')
# plt.axhline(y=c, color='r')
# plt.xlabel('iter')
# plt.ylabel('c')
# plt.savefig('images/all_trueinit3/c')
# plt.close()
#
# plt.figure()
# t_est = output[5]
# plt.plot(t_est, color='blue')
# plt.axhline(y=t, color='r')
# plt.xlabel('iter')
# plt.ylabel('t')
# plt.savefig('images/all_trueinit3/t')
# plt.close()
#
# plt.figure()
# w_est = output[0]
# deg = np.array(list(dict(G.degree()).values()))
# biggest_deg = np.argsort(deg)[-1]
# biggest_w_est = [w_est[i][biggest_deg] for i in range(iter)]
# plt.plot(biggest_w_est)
# biggest_w = w[biggest_deg]
# plt.axhline(y=biggest_w, label='true')
# plt.xlabel('iter')
# plt.ylabel('highest degree w')
# plt.legend()
# plt.savefig('images/all_trueinit3/w0_trace')
# plt.close()
# # plot empirical 95% ci for highest and lowest degrees nodes
# plt.figure()
# w_est_fin = [w_est[i] for i in range(nburn, iter)]
# emp0_ci_95 = [
#     scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w (true init) = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
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
# plt.savefig('images/all_trueinit3/w0_CI')
# plt.close()
#
# # u_est = output[8]
# # u_est_fin = [u_est[i] for i in range(nburn, iter)]
# # emp_u_ci_95 = [
# #     scipy.stats.mstats.mquantiles([u_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
# #     for j in range(size)]
# # true_u_in_ci = [emp_u_ci_95[i][0] <= u[i] <= emp_u_ci_95[i][1] for i in range(size)]
# # print('posterior coverage of true u (true init) = ', sum(true_u_in_ci) / len(true_u_in_ci) * 100, '%')
#
# # plot autocorrelations
#
# plt.figure()
# plt.acorr(output[3], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.savefig('images/all_trueinit3/autocor_sigma')
# plt.close()
# plt.figure()
# plt.acorr(output[4], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.savefig('images/all_trueinit3/autocor_c')
# plt.close()
# plt.figure()
# plt.acorr(output[5], detrend=plt.mlab.detrend_mean, maxlags=100)
# plt.xlim(0, 100)
# plt.savefig('images/all_trueinit3/autocor_t')
# plt.close()

# ----------------------------------
# All together - 3 chains random init
# -----------------------------------

# traceplots

plt.figure()
plt.plot(output2[10], color='cornflowerblue')
plt.plot(output3[10], color='navy')
plt.plot(output4[10], color='blue')
plt.axhline(y=log_post, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('log_post')
plt.savefig('images/all_rand3/log_post')
plt.close()

plt.figure()
plt.plot(output2[3], color='cornflowerblue')
plt.plot(output3[3], color='navy')
plt.plot(output4[3], color='blue')
plt.axhline(y=sigma, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('sigma')
plt.savefig('images/all_rand3/sigma')
plt.close()

plt.figure()
plt.plot(output2[4], color='cornflowerblue')
plt.plot(output3[4], color='navy')
plt.plot(output4[4], color='blue')
plt.axhline(y=c, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('c')
plt.savefig('images/all_rand3/c')
plt.close()

plt.figure()
plt.plot(output2[5], color='cornflowerblue')
plt.plot(output3[5], color='navy')
plt.plot(output4[5], color='blue')
plt.axhline(y=t, label='true', color='r')
plt.xlabel('iter')
plt.ylabel('t')
plt.savefig('images/all_rand3/t')
plt.close()

# CHAIN 1

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
plt.savefig('images/all_rand3/w0_trace_chain1')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
    for j in range(size)]
print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]))
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 1 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
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
plt.savefig('images/all_rand3/w0_CI_chain1')
plt.close()

# u_est = output2[8]
# u_est_fin = [u_est[i] for i in range(nburn, iter)]
# emp_u_ci_95 = [
#     scipy.stats.mstats.mquantiles([u_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true_u_in_ci = [emp_u_ci_95[i][0] <= u[i] <= emp_u_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true u in chain 1 = ', sum(true_u_in_ci) / len(true_u_in_ci) * 100, '%')

# CHAIN 2

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
plt.savefig('images/all_rand3/w0_trace_chain2')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
emp0_ci_95 = [
    scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
    for j in range(size)]
true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 2 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
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
plt.savefig('images/all_rand3/w0_CI_chain2')
plt.close()

# u_est = output3[8]
# u_est_fin = [u_est[i] for i in range(nburn, iter)]
# emp_u_ci_95 = [
#     scipy.stats.mstats.mquantiles([u_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true_u_in_ci = [emp_u_ci_95[i][0] <= u[i] <= emp_u_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true u in chain 2 = ', sum(true_u_in_ci) / len(true_u_in_ci) * 100, '%')

# CHAIN 3

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
plt.savefig('images/all_rand3/w0_trace_chain3')
plt.close()
# plot empirical 95% ci for highest and lowest degrees nodes
plt.figure()
w_est_fin = [w_est[i] for i in range(nburn, iter)]
print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]))
# print(sum([emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]))
# true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true w in chain 3 = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
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
plt.savefig('images/all_rand3/w0_CI_chain3')
plt.close()

# u_est = output4[8]
# u_est_fin = [u_est[i] for i in range(nburn, iter)]
# emp_u_ci_95 = [
#     scipy.stats.mstats.mquantiles([u_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
#     for j in range(size)]
# true_u_in_ci = [emp_u_ci_95[i][0] <= u[i] <= emp_u_ci_95[i][1] for i in range(size)]
# print('posterior coverage of true u in chain 3 = ', sum(true_u_in_ci) / len(true_u_in_ci) * 100, '%')

# plot autocorrelations

plt.figure()
plt.acorr(output2[3], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_sigma_chain1')
plt.close()
plt.figure()
plt.acorr(output3[3], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_sigma_chain2')
plt.close()
plt.figure()
plt.acorr(output4[3], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_sigma_chain3')
plt.close()
plt.figure()
plt.acorr(output2[4], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_c_chain1')
plt.close()
plt.figure()
plt.acorr(output3[4], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_c_chain2')
plt.close()
plt.figure()
plt.acorr(output4[4], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_c_chain3')
plt.close()
plt.figure()
plt.acorr(output2[5], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_t_chain1')
plt.close()
plt.figure()
plt.acorr(output3[5], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_t_chain2')
plt.close()
plt.figure()
plt.acorr(output4[5], detrend=plt.mlab.detrend_mean, maxlags=100)
plt.xlim(0, 100)
plt.savefig('images/all_rand3/autocor_t_chain3')
plt.close()

# Gelman Rubin

n = iter - nburn + 1
W = (np.array(output2[3][nburn:nburn+iter]).std() ** 2 + np.array(output3[3][nburn:nburn+iter]).std() ** 2 +
     np.array(output4[3][nburn:nburn+iter]).std() ** 2) / 3
mean1 = np.array(output2[3][nburn:nburn+iter]).mean()
mean2 = np.array(output3[3][nburn:nburn+iter]).mean()
mean3 = np.array(output4[3][nburn:nburn+iter]).mean()
mean = (mean1 + mean2 + mean3) / 3
B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
var_theta = (1 - 1/n) * W + 1 / n * B
print("Gelman-Rubin Diagnostic sigma: ", np.sqrt(var_theta/W))
W = (np.array(output2[4][nburn:nburn+iter]).std() ** 2 + np.array(output3[4][nburn:nburn+iter]).std() ** 2 +
     np.array(output4[4][nburn:nburn+iter]).std() ** 2) / 3
mean1 = np.array(output2[4][nburn:nburn+iter]).mean()
mean2 = np.array(output3[4][nburn:nburn+iter]).mean()
mean3 = np.array(output4[4][nburn:nburn+iter]).mean()
mean = (mean1 + mean2 + mean3) / 3
B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
var_theta = (1 - 1/n) * W + 1 / n * B
print("Gelman-Rubin Diagnostic c: ", np.sqrt(var_theta/W))
W = (np.array(output2[5][nburn:nburn+iter]).std() ** 2 + np.array(output3[5][nburn:nburn+iter]).std() ** 2 +
     np.array(output4[5][nburn:nburn+iter]).std() ** 2) / 3
mean1 = np.array(output2[5][nburn:nburn+iter]).mean()
mean2 = np.array(output3[5][nburn:nburn+iter]).mean()
mean3 = np.array(output4[5][nburn:nburn+iter]).mean()
mean = (mean1 + mean2 + mean3) / 3
B = n / 2 * ((mean1 - mean) ** 2 + (mean2 - mean) ** 2 + (mean3 - mean) ** 2)
var_theta = (1 - 1/n) * W + 1 / n * B
print("Gelman-Rubin Diagnostic t: ", np.sqrt(var_theta/W))