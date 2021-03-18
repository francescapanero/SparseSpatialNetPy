import utils.TruncPois as tp
from utils.loggrad import *
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import utils.Auxiliary as aux
from itertools import compress
import scipy.special
import math


# functions to update the parameters sigma, c, t, tau with MH,
# the variables w with gibbs or HMC
# the variables n and u with gibbs


# conditional for counts n: n|w,x \sim Truncated Poisson (2 w_i w_j p_ij)
def update_n(w, G, size, p_ij, ind, selfedge):
    n_ = lil_matrix((size, size))
    for i in G.nodes:
        if ind[i] is not []:
            n_[i, ind[i]] = tp.tpoissrnd(2 * w[i] * w[ind[i]] * p_ij[i, ind[i]])
    n_[selfedge, selfedge] = [tp.tpoissrnd(w[i] ** 2) for i in selfedge]
    n_ = csr_matrix(n_)
    sum_log_fact_n = 0
    # sum_log_fact_n = sum(np.log(scipy.special.factorial(n_[n_ > 1])[0]))
    return n_, sum_log_fact_n


# conditional for the auxiliary var u|w0 \sim Truncated Poisson (z w0)
def posterior_u(lam):
    u = tp.tpoissrnd(lam)
    return u


# function to update the locations x in a sweep of Metropolis Hastings.
# As proposal, I use tilde_x | x \sim Lo Normal (log x, sigma_x^2)
def update_x(x, w, gamma, p_ij, n, sigma_x, acc_distance, prior, sigma, c, t, tau, w0, beta, u, a_t, b_t, sum_n,
             sum_fact_n, log_post, log_post_par):
    # tilde_x = np.exp(np.log(x) + sigma_x * np.random.normal(0, 1, len(x)))
    tilde_x = x
    tilde_x[0] = np.random.normal(x[0], sigma_x)
    tilde_pij = p_ij
    tilde_pij[0, 1:len(x)] = [1 / ((1 + np.abs(tilde_x[0] - x[i])) ** gamma) for i in range(1, len(x))]
    tilde_pij[:, 0] = tilde_pij[0, :]
    # tilde_pij = aux.space_distance(tilde_x, gamma)
    tilde_logpost = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, tilde_pij, a_t, b_t,
                                                 gamma, sum_n, sum_fact_n, log_post_par=log_post_par)
    log_r = tilde_logpost - log_post + sum(np.log(tilde_x) - np.log(x))
    if log_r < 0:
        if np.random.rand(1) < np.exp(log_r):
            x = tilde_x
            p_ij = tilde_pij
            acc_distance += 1
            log_post = tilde_logpost
    else:
        x = tilde_x
        p_ij = tilde_pij
        acc_distance += 1
        log_post = tilde_logpost
    return x, p_ij, acc_distance, log_post


# function to update sigma, c, t, tau in a sweep of Metropolis Hastings. The inputs are:
# prior: singlepl or doublepl
# sigma_prev, c_prev, t_prev, tau_prev, z_prev: the previous values of the parameters
# w0, beta, u: the variables not updated in this step
# log_post: log posterior of precedent step
# accept: how many steps have been accepted so far
# sigma, c, t, tau: boolean vectors saying which params has to be updated
# sigma_sigma, sigma_c, sigma_t, sigma_tau: step-sizes of MH proposals
# a_t, b_t: prior params of Gamma distribution for t
def update_params(prior, sigma_prev, c_prev, t_prev, tau_prev, z_prev, w0, beta, u, log_post, accept,
                  sigma, c, t, tau,
                  sigma_sigma, sigma_c, sigma_t, sigma_tau, a_t, b_t):
    size = len(w0)
    if sigma is True:
        # tilde_sigma = math.inf
        # while math.isinf(tilde_sigma) is True or math.isnan(tilde_sigma) is True or tilde_sigma == 0:
        l = np.exp(np.log(sigma_prev / (1 - sigma_prev)) + sigma_sigma * np.random.normal(0, 1))
        tilde_sigma = l / (1 + l)
    else:
        tilde_sigma = sigma_prev
    # tilde_c = math.inf
    # while math.isinf(tilde_c) is True or math.isnan(tilde_c) is True or tilde_c == 0:
    tilde_c = np.exp(np.log(c_prev) + sigma_c * np.random.normal(0, 1)) if c is True else c_prev
    # tilde_t = math.inf
    # while math.isinf(tilde_t) is True or math.isnan(tilde_t) is True or tilde_t == 0:
    tilde_t = np.exp(np.log(t_prev) + sigma_t * np.random.normal(0, 1)) if t is True else t_prev
    tilde_tau = np.exp(np.log(tau_prev) + sigma_tau * np.random.normal(0, 1)) if tau is True else tau_prev
    tilde_log_post = aux.log_post_params(prior, tilde_sigma, tilde_c, tilde_t, tilde_tau, w0, beta, u, a_t, b_t)
    log_proposal = aux.log_proposal_MH(prior, sigma_prev, tilde_sigma, c_prev, tilde_c, t_prev, tilde_t, tau_prev,
                                       tilde_tau, sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0)
    tilde_log_proposal = aux.log_proposal_MH(prior, tilde_sigma, sigma_prev, tilde_c, c_prev, tilde_t, t_prev,
                                             tilde_tau, tau_prev, sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0)
    log_r = tilde_log_post - log_post + log_proposal - tilde_log_proposal
    if log_r < 0:
        if np.random.rand(1) < min(1, np.exp(log_r)):
            accept = accept + 1
            tilde_z = (size * tilde_sigma / tilde_t) ** (1 / tilde_sigma) if prior == 'singlepl' else \
                      (size * tilde_tau * tilde_sigma ** 2 / (
                        tilde_t * tilde_c ** (tilde_sigma * (tilde_tau - 1)))) ** (1 / tilde_sigma)
            return np.array((tilde_sigma, tilde_c, tilde_t, tilde_tau, tilde_z, accept, tilde_log_post, np.exp(log_r)))
        else:
            return np.array((sigma_prev, c_prev, t_prev, tau_prev, z_prev, accept, log_post, np.exp(log_r)))
    else:
        accept = accept + 1
        tilde_z = (size * tilde_sigma / tilde_t) ** (1 / tilde_sigma) if prior == 'singlepl' else \
            (size * tilde_tau * tilde_sigma ** 2 / (
                    tilde_t * tilde_c ** (tilde_sigma * (tilde_tau - 1)))) ** (1 / tilde_sigma)
        return np.array(
            (tilde_sigma, tilde_c, tilde_t, tilde_tau, tilde_z, accept, tilde_log_post, 1))


# # same as previous function, but this time the update is for sigma, c, z, tau (z instead of t in a change of vars)
# def update_params(prior, sigma_prev, c_prev, t_prev, tau_prev, z_prev, w0, beta, u, log_post, accept,
#                   sigma, c, t, tau,
#                   sigma_sigma, sigma_c, sigma_t, sigma_tau, a_t, b_t):
#     size = len(w0)
#     if sigma is True:
#         l = np.exp(np.log(sigma_prev / (1 - sigma_prev)) + sigma_sigma * np.random.normal(0, 1))
#         tilde_sigma = l / (1 + l)
#     else:
#         tilde_sigma = sigma_prev
#     tilde_c = np.exp(np.log(c_prev) + sigma_c * np.random.normal(0, 1)) if c is True else c_prev
#     tilde_z = np.random.gamma(np.sum(u) - 2 * tilde_sigma, 1 / (np.sum(w0) + sigma_sigma)) if t is True else z_prev  # add
#     tilde_t = len(w0) * tilde_sigma * tilde_z ** (- tilde_sigma)
#     tilde_tau = np.exp(np.log(tau_prev) + sigma_tau * np.random.normal(0, 1)) if tau is True else tau_prev
#     tilde_log_post = aux.log_post_params(prior, tilde_sigma, tilde_c, tilde_t, tilde_tau, w0, beta, u, a_t, b_t)
#     log_proposal = aux.log_proposal_MH(prior, sigma_prev, tilde_sigma, c_prev, tilde_c, t_prev, tilde_t, tau_prev,
#                                        tilde_tau, sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0)
#     tilde_log_proposal = aux.log_proposal_MH(prior, tilde_sigma, sigma_prev, tilde_c, c_prev, tilde_t, t_prev,
#                                              tilde_tau, tau_prev, sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0)
#     log_r = tilde_log_post - log_post + log_proposal - tilde_log_proposal
#     if np.random.rand(1) < min(1, np.exp(log_r)):
#         accept = accept + 1
#         tilde_z = (size * tilde_sigma / tilde_t) ** (1 / tilde_sigma) if prior == 'singlepl' else \
#                   (size * tilde_tau * tilde_sigma ** 2 / (
#                     tilde_t * tilde_c ** (tilde_sigma * (tilde_tau - 1)))) ** (1 / tilde_sigma)
#         return np.array((tilde_sigma, tilde_c, tilde_t, tilde_tau, tilde_z, accept, tilde_log_post, min(1, np.exp(log_r))))
#     else:
#         return np.array((sigma_prev, c_prev, t_prev, tau_prev, z_prev, accept, log_post, min(1, np.exp(log_r))))


# function to update weights w given the rest using a Gibbs step
# w: prev value of weights
# beta, u, n: value of auxiliary variables
# sigma, c, z: the parameters needed for the update
# p_ij: distance matrix
# gamma: exponent for the distance term in the graphon
def gibbs_w(w, beta, sigma, c, z, u, n, p_ij, gamma, sum_n):
    sum_n = sum_n - 2 * n.diagonal()
    shape = - sigma + u + sum_n
    if gamma == 0:
        scale = 1 / (c + z + 2 * (sum(w) - w) / beta)
    if gamma != 0:
        scale = 1 / (c + z + 2 * (np.dot(p_ij, w) - w) / beta)
    w0 = np.random.gamma(shape, scale)
    return w0 / beta, w0


# function to update weights w given the rest using a HMC step
# w, w0: prev value of weights
# beta, u, n: value of auxiliary variables
# sigma, c, t, tau, z: the parameters needed for the update
# gamma: exponent for the distance term in the graphon
# p_ij: distance matrix
# a_t, b_t: prior parameters for t (Gamma distribution)
# epsilon: step size of HMC
# R: number of steps in each HMC sweep
# accept: number of steps previously accepted
# size: size of w (number of active and inactive nodes)
# update_beta: if beta is to be updated or not (in singlepl it will be automatically put to 0)
def HMC_w(prior, w, w0, beta, n, u, sigma, c, t, tau, z, gamma, p_ij, a_t, b_t,
          epsilon, R, accept, size, sum_n, sum_fact_n, log_post, log_post_param, update_beta=True):
    temp1 = sum_n + u - sigma
    temp1_beta = sum_n - sigma * tau
    # first step: propose w0 and beta and auxiliary vars p_w0 p_beta normally distributed
    # pw_outer = np.dot(w, np.dot(p_ij, w)) if gamma != 0 else w * sum(w)
    pw_outer = w * np.dot(p_ij, w) if gamma != 0 else w * sum(w)
    temp2 = (c + z) * w0
    beta_prop = beta
    logbeta_prop = np.log(beta_prop)
    logw0_prop = np.log(w0)
    p_w0 = np.random.normal(0, 1, size)
    p_prop_w0 = p_w0 + epsilon * loggrad(temp1, temp2, pw_outer) / 2
    if prior == 'doublepl' and update_beta is True:
        p_beta = np.random.normal(0, 1, size)
        p_prop_beta = p_beta + epsilon * loggrad(np.negative(temp1_beta), 0, np.negative(pw_outer)) / 2
    # steps j=0...R-1
    for j in range(R):
        logw0_prop = logw0_prop + epsilon * p_prop_w0
        w0_prop = np.exp(logw0_prop)
        if prior == 'doublepl' and update_beta is True:
            logbeta_prop = logbeta_prop + epsilon * p_prop_beta
            beta_prop = np.exp(logbeta_prop)
        logw_prop = logw0_prop - logbeta_prop
        w_prop = np.exp(logw_prop)
        # pw_outer_prop = np.dot(w_prop, np.dot(p_ij, w_prop)) if gamma != 0 else w_prop * sum(w_prop)
        pw_outer_prop = w_prop * np.dot(p_ij, w_prop) if gamma != 0 else w_prop * sum(w_prop)
        temp2_prop = (c + z) * w0_prop
        p_prop_w0 = p_prop_w0 + epsilon * loggrad(temp1, temp2_prop, pw_outer_prop) if j != (R-1) else \
                    - p_prop_w0 - epsilon * loggrad(temp1, temp2_prop, pw_outer_prop) / 2
        if update_beta is True:
            if j != (R-1):
                p_prop_beta = p_prop_beta + epsilon * loggrad(np.negative(temp1_beta), 0, np.negative(pw_outer_prop))
            else:
                p_prop_beta = - p_prop_beta - epsilon * loggrad(np.negative(temp1_beta),0,np.negative(pw_outer_prop))/2
    # compute log accept rate
    log_post_par_prop = aux.log_post_params(prior, sigma, c, t, tau, w0_prop, beta_prop, u, a_t, b_t)
    log_post_prop = aux.log_post_logwbeta_params(prior, sigma, c, t, tau, w_prop, w0_prop, beta_prop, n, u, p_ij, a_t,
                                                 b_t, gamma, sum_n, sum_fact_n, log_post_par=log_post_par_prop)
    log_r = log_post_prop - log_post - sum(p_prop_w0 ** 2 - p_w0 ** 2) / 2
    if update_beta is True:
        log_r = log_r - sum(p_prop_beta ** 2 - p_beta ** 2) / 2

    # accept step
    if log_r < 0:
        rate = min(1, np.exp(log_r))
        if np.random.rand(1) < rate:
            w = w_prop
            w0 = w0_prop
            beta = beta_prop
            accept = accept + 1
            log_post = log_post_prop
            log_post_param = log_post_par_prop
    else:
        rate = 1
        w = w_prop
        w0 = w0_prop
        beta = beta_prop
        accept = accept + 1
        log_post = log_post_prop
        log_post_param = log_post_par_prop
    return w, w0, beta, accept, rate, log_post, log_post_param
