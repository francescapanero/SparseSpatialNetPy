import numpy as np
import scipy
import networkx as nx
import math
from scipy.sparse import lil_matrix


def space_distance(x, gamma):
    size = len(x)
    dist = np.zeros((size, size))
    p_ij = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist[i, j] = np.absolute(x[i] - x[j])
            p_ij[i, j] = 1 / ((1 + dist[i, j]) ** gamma)
    p_ij = p_ij + np.transpose(p_ij)
    for i in range(size):
        p_ij[i,i] = 1
    return p_ij


# log likelihood (sigma, c, t, tau | w0, beta, u, n, x)
def log_likel_params(prior, sigma, c, t, tau, w0, beta, u):
    size = len(w0)
    z = (size * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
        (size * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
    log_likel = size * (np.log(sigma) - np.log(scipy.special.gamma(1 - sigma)) - np.log(
                        (z + c) ** sigma - c ** sigma)) \
                - sigma * sum(np.log(w0)) + np.log(z) * sum(u) - (z + c) * sum(w0)
    if prior == 'doublepl':
        log_likel = log_likel + size * np.log(sigma * tau) + sigma * tau * sum(np.log(beta))
    return log_likel


# ---------------------------------------------------
# LOG POSTERIOR
# ---------------------------------------------------

# # log posterior (sigma, c, t, tau | w0, beta, n, u, a_t, b_t) with improper priors for sigma and c,
# # and Gamma prior for t
# def log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t):
#     log_prior = - np.log(sigma * (1 - sigma)) - np.log(c) + (a_t - 1) * np.log(t) - b_t * t
#     if prior == 'doublepl':
#         log_prior = log_prior - np.log(tau)
#     log_post = log_likel_params(prior, sigma, c, t, tau, w0, beta, u) + log_prior
#     return log_post


# log posterior (sigma, c, t, tau | w0, beta, n, u, a_t, b_t) with proper priors:
# Beta(s_1, s_2) for sigma
# Gammas for t, c and tau
def log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t):
    s_1 = 2
    s_2 = 2
    a_c = 5
    b_c = 1
    log_prior = (s_1 - 1) * np.log(sigma) + (s_2 - 1) * np.log(1 - sigma) + (a_t - 1) * np.log(t) - b_t * t \
                + (a_c - 1) * np.log(t) - b_c * t
    if prior == 'doublepl':
        a_tau = 5
        b_tau = 1
        log_prior = log_prior + (a_tau - 1) * np.log(t) - b_tau * t
    log_post = log_likel_params(prior, sigma, c, t, tau, w0, beta, u) + log_prior
    return log_post


# # log posterior (sigma, c, z, tau | w0, beta, n, u, a_t, b_t) CHANGE OF VARIABLE z instead of t
# # with improper priors for sigma and c and Gamma prior for z
# def log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t):
#     log_prior = - np.log(sigma * (1 - sigma)) - np.log(c) + scipy.stats.gamma.logpdf(1000, 1)
#     if prior == 'doublepl':
#         log_prior = log_prior - np.log(tau)
#     L = len(w0)
#     z = (L * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
#         (L * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
#     log_post = log_likel_params(prior, sigma, c, t, tau, w0, beta, u) + log_prior + np.log(sigma ** 2 * L *
#                                                                                            z ** (- sigma - 1))
#     return log_post


# log posterior (w0, beta, sigma, c, t, tau | x, n, u)
def log_post_wbetaparams(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n):
    if gamma == 0:
        log_post_wbeta = log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t) + \
                         sum(sum_n * np.log(w) - w * sum(w) + u * np.log(w0) - np.log(beta))
    if gamma != 0:
        log_post_wbeta = log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t) + \
                         sum(sum_n * np.log(w) - sum(np.outer(w, np.dot(p_ij, w))) + u * np.log(w0)
                             - np.log(beta))
    return log_post_wbeta


# log posterior (logw0, logbeta, sigma, c, t, tau | x, n, u)
# the change of variables is such that p(logw, logbeta) = w * beta * p(w, beta)
# hence log p(logw, logbeta) = logw + logbeta + log p(w, beta)
def log_post_logwbeta_params(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, **kwargs):
    if 'sum_n' in kwargs:
        sum_n = kwargs['sum_n']
    else:
        sum_n = lil_matrix.sum(n, axis=0)
        sum_n_ = lil_matrix.sum(n, axis=1)
        sum_n = np.array(sum_n + np.transpose(sum_n_))[0]
    log_post_logwbetaparams = log_post_wbetaparams(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n) + \
                              sum(np.log(w0) + np.log(beta))
    return log_post_logwbetaparams


# log posterior (w0, beta, n, u, sigma, c, t, tau | x)
# honestly, I don't think I will ever compute this
def log_post_all(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t):
    log_post = log_post_wbetaparams(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t) - \
                   sum(np.log(math.factorial(n)) + np.log(math.factorial(u)))
    return log_post


# --------------------------------------------------------------
# LOG PROPOSAL for Metropolis Hastings of hyperparameters
# --------------------------------------------------------------

# log proposal for Metropolis Hastings step for hyperparameters sigma, c, t, tau
def log_proposal_MH(prior, sigma, tilde_sigma, c, tilde_c, t, tilde_t, tau, tilde_tau,
                    sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0):
    log_prop = \
    scipy.stats.norm.logpdf(np.log(sigma / (1 - sigma)), np.log(tilde_sigma / (1 - tilde_sigma)), sigma_sigma) - \
    np.log((sigma * (1 - sigma))) + \
    scipy.stats.lognorm.logpdf(c, sigma_c, 0, tilde_c) + \
    scipy.stats.lognorm.logpdf(t, sigma_t, 0, tilde_t)
    if prior == 'doublepl':
        log_prop = log_prop + scipy.stats.lognorm.logpdf(tau, sigma_tau, 0, tilde_tau)
    return log_prop


# # log proposal for Metropolis Hastings step for hyperparameters sigma, c, z, tau. CHANGE OF VARIABLE z instead of t
# def log_proposal_MH(prior, sigma, tilde_sigma, c, tilde_c, t, tilde_t, tau, tilde_tau,
#                     sigma_sigma, sigma_c, sigma_t, sigma_tau, u, w0):
#     log_prop = \
#     scipy.stats.norm.logpdf(np.log(sigma / (1 - sigma)), np.log(tilde_sigma / (1 - tilde_sigma)), sigma_sigma) - \
#     np.log((sigma * (1 - sigma))) + \
#     scipy.stats.lognorm.logpdf(c, sigma_c, 0, tilde_c) + \
#     scipy.stats.gamma.logpdf(sum(u) - 2 * sigma, 1 / (sum(w0) + sigma_sigma))
#     if prior == 'doublepl':
#         log_prop = log_prop + scipy.stats.lognorm.logpdf(tau, sigma_tau, 0, tilde_tau)
#     return log_prop


# --------------------------------------------------------------
# CHECK LOG LIKELIHOOD of PARAMETERS
# --------------------------------------------------------------

def check_log_likel_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t):

    if prior == 'singlepl':
        sigma_ = np.linspace(0.05, sigma+0.3, 20)
        c_ = np.linspace(max(1.1, c - 2), c + 3, 50)
        t_ = np.linspace(t - 30, t + 30, 60)
        mat = [[[np.array((sigma_[i], c_[j], t_[k])) for i in range(len(sigma_))] for j in range(len(c_))]
                for k in range(len(t_))]
        log_post = np.zeros((len(t_), len(c_), len(sigma_)))
        for i in range(len(t_)):
            for j in range(len(c_)):
                for k in range(len(sigma_)):
                    log_post[i, j, k] = log_likel_params(prior, mat[i][j][k][0], mat[i][j][k][1], mat[i][j][k][2], tau,
                                                        w0, beta, u)
        ind_max = np.unravel_index(np.argmax(log_post, axis=None), log_post.shape)
        return log_post[ind_max], mat[ind_max[0]][ind_max[1]][ind_max[2]]

    if prior == 'doublepl':
        sigma_ = np.linspace(0.05, 0.95, 15)
        c_ = np.linspace(max(1.1, c - 2), c + 2, 40)
        t_ = np.linspace(t - 10, t + 10, 30)
        tau_ = np.linspace(max(1.1, tau - 3), tau + 3, 40)
        mat = [[[[np.array((sigma_[i], c_[j], t_[k], tau_[h])) for i in range(len(sigma_))] for j in range(len(c_))]
               for k in range(len(t_))] for h in range(len(tau_))]
        log_post = np.zeros((len(tau_), len(t_), len(c_), len(sigma_)))
        for i in range(len(tau_)):
            for j in range(len(t_)):
                for k in range(len(c_)):
                    for h in range(len(sigma_)):
                        log_post[i, j, k, h] = log_likel_params(prior, mat[i][j][k][h][0], mat[i][j][k][h][1],
                                                               mat[i][j][k][h][2], mat[i][j][k][h][3], w0, beta, u)
        ind_max = np.unravel_index(np.argmax(log_post, axis=None), log_post.shape)
        return log_post[ind_max], mat[ind_max[0]][ind_max[1]][ind_max[2]][ind_max[3]]


# --------------------------------------------------------------
# SIMPLE GRAPH (remove self loops and isolated nodes)
# --------------------------------------------------------------

def SimpleGraph(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    isol = list(nx.isolates(G))
    G.remove_nodes_from(isol)
    return G, isol


# --------------------------------------------------------------
# TUNING of Metropolis Hastings' stepsize
# --------------------------------------------------------------

# tune the parameter of the MCMC proposal according to the acceptance rate, to reach optimal acceptance
# Multidimensional Metropolis - Hastings: ~ 30 %
def tune(acceptance, scale, step):  # need iter multiple of t and > t
    acc_rate = (acceptance[len(acceptance)-1]-acceptance[len(acceptance)-step])/step
    if acc_rate < 0.001:
        scale *= 0.1
    elif acc_rate < 0.05:
        scale *= 0.5
    elif acc_rate < 0.2:
        scale *= 0.9
    elif acc_rate > 0.95:
        scale *= 10.0
    elif acc_rate > 0.75:
        scale *= 2.0
    elif acc_rate > 0.5:
        scale *= 1.1
    return scale