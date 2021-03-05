from utils.GraphSamplerNew import *
import utils.AuxiliaryNew_fast as aux
import numpy as np
import mcmc_chains as chain
import pickle
import _pickle as cPickle
import gzip

# Set parameters for simulating data
t = 100  # ex alpha: time threshold

sigma = 0.4  # shape generalized gamma process
c = 2  # rate generalized gamma process
tau = 5  # only for doublepl

gamma = 1  # exponent distance in the link probability
size_x = 5  # space threshold: [0, size_x]

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

# ----------------------
# SIMULATE DATA
# ----------------------

G = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=1000)

# G1 = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x, a_t, b_t, T=T, K=K, L=2000)

w = np.array([G.nodes[i]['w'] for i in range(G.number_of_nodes())])
w0 = np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])
beta = np.array([G.nodes[i]['beta'] for i in range(G.number_of_nodes())])
x = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
u = np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])
deg = np.array(list(dict(G.degree()).values()))
n = G.graph['counts']
sum_fact_n = G.graph['sum_fact_n']
p_ij = G.graph['distances']
ind = G.graph['ind']
selfedge = G.graph['selfedge']
log_post = G.graph['log_post']

# histdeg = nx.degree_histogram(G)
# plt.loglog(range(1, len(histdeg)), histdeg[1:], 'go-')

# if reduce is True:
#     G, isol = aux.SimpleGraph(G)
#     x_red = np.delete(x, isol)
#     w_red = np.delete(w, isol)
#
# if check is True:
#     temp = aux.check_log_likel_params(prior, sigma, c, t, tau, w0, beta, u)
#     log_lik_true = aux.log_likel_params(prior, sigma, c, t, tau, w0, beta, u)
#     print('true log likel params = ', log_lik_true)
#     print('log likel in max = ', temp[0])
#     if prior == 'singlepl':
#         print('params for max (sigma, c, t) = ', temp[1])
#     if prior == 'doublepl':
#         print('params for max (sigma, c, t, tau) = ', temp[1])
#     if log_lik_true < temp[0]:
#         print('the approximation is not working! Decrease T!')

# ---------------------------
# posterior inference
# ---------------------------
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

iter = 200000
nburn = int(iter * 0.25)

init = {}
init[0] = {}
init[0]['w_init'] = w
init[0]['w0_init'] = w
init[0]['beta_init'] = beta
init[0]['n_init'] = n
init[0]['sum_fact_n'] = sum_fact_n
init[0]['u_init'] = u
init[0]['sigma_init'] = sigma
init[0]['c_init'] = c
init[0]['t_init'] = t
init[0]['tau_init'] = tau
init[0]['x_init'] = x
# init[1] = {}
# init[1]['w_init'] = w_1
# init[1]['w0_init'] = w_1
# init[1]['sigma_init'] = sigma + 0.1
# init[1]['c_init'] = c + 1
# init[1]['t_init'] = t + 40
# init[2] = {}

out = chain.mcmc_chains([G], iter, nburn,
                        sigma=True, c=True, t=True, tau=False,
                        w0=True,
                        n=True,
                        u=True,
                        x=False,
                        beta=False,
                        w_inference='HMC', epsilon=0.01, R=5,
                        sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                        save_every=1000,
                        plot=True, path='all23_trueinit_nox', save_out=False, save_data=False,
                        init=init)

# def load_zipped_pickle(filename):
#     with gzip.open(filename, 'rb') as f:
#         loaded_object = cPickle.load(f)
#         return loaded_object
