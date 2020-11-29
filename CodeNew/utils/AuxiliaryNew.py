import numpy as np
import scipy
import matplotlib.pyplot as plt
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
    log_likel_params = size * (np.log(sigma) - np.log(scipy.special.gamma(1 - sigma)) - np.log(
                        (z + c) ** sigma - c ** sigma)) \
                        - sigma * sum(np.log(w0)) + np.log(z) * sum(u) - (z + c) * sum(w0) \
                        + sigma * tau * sum(np.log(beta))
    if prior == 'doublepl':
        log_likel_params = log_likel_params + size * np.log(sigma * tau)
    return log_likel_params


# log posterior (sigma, c, t, tau | w0, beta, n, u, a_t, b_t)
def log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t):
    log_prior_params = np.log(sigma * (1 - sigma)) - np.log(c) + (a_t - 1) * np.log(t) - b_t * t
    if prior == 'doublepl':
        log_prior_params = log_prior_params - np.log(tau)
    log_post_params = log_likel_params(prior, sigma, c, t, tau, w0, beta, u) + log_prior_params
    return log_post_params


# log posterior (w0, beta, sigma, c, t, tau | x, n, u)
def log_post_wbetaparams(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t, gamma, sum_n):
    if gamma == 0:
        log_post_wbeta = log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t) + \
                         sum(sum_n * np.log(w) - w * sum(w) + u * np.log(w0) - np.log(beta))
    if gamma != 0:
        log_post_wbeta = log_post_params(prior, sigma, c, t, tau, w0, beta, u, a_t, b_t) + \
                         sum(sum_n * np.log(w) - sum(np.outer(w, np.dot(p_ij, w))) + u * np.log(w0)
                             - np.log(beta))
                         # sum(n * np.log(w) - w * np.dot(p_ij, w) + (u - 1) * np.log(w0) - np.log(beta))
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
    log_post_all = log_post_wbetaparams(prior, sigma, c, t, tau, w, w0, beta, n, u, p_ij, a_t, b_t) - \
                   sum(np.log(math.factorial(n)) + np.log(math.factorial(u)))
    return log_post_all


def check_sample_loglik(prior, sigma, c, t, tau, w0, beta, u):

    sigma_ = np.linspace(0.05, 0.95, 100)
    b = [log_likel_params(prior, sigma_[i], c, t, tau, w0, beta, u) for i in range(len(sigma_))]
    plt.plot(sigma_, b)
    plt.axvline(x=sigma, label='true')
    plt.xlabel('sigma')
    b_ind = b.index(max(b))
    plt.axvline(x=sigma_[b_ind], color='r', label='max')
    plt.legend()

    plt.figure()
    c_ = np.linspace(max(0.9, c-3), c + 3, 200)
    d = [log_likel_params(prior, sigma, c_[i], t, tau, w0, beta, u) for i in range(len(c_))]
    plt.plot(c_, d)
    plt.axvline(x=c, label='true')
    plt.xlabel('c')
    d_ind = d.index(max(d))
    plt.axvline(x=c_[d_ind], color='r', label='max')
    plt.legend()

    plt.figure()
    t_ = np.linspace(t - 10, t + 10, 101)
    e = [log_likel_params(prior, sigma, c, t_[i], tau, w0, beta, u) for i in range(len(t_))]
    plt.plot(t_, e)
    plt.axvline(x=t, label='true')
    plt.xlabel('t')
    e_ind = e.index(max(e))
    plt.axvline(x=t_[e_ind], color='r', label='max')
    plt.legend()

    if prior == 'doublepl':
        plt.figure()
        tau_ = np.linspace(1.1, tau + 3, 100)
        f = [log_likel_params(prior, sigma, c, t, tau_[i], w0, beta, u) for i in range(len(tau_))]
        plt.plot(tau_, f)
        plt.axvline(x=tau, label='true')
        plt.xlabel('tau')
        f_ind = np.where(f == max(f))
        plt.axvline(x=tau_[f_ind], color='r', label='max')
        plt.legend()
        return np.array((sigma_[b_ind], c_[d_ind], t_[e_ind], tau_[f_ind]))

    return np.array((sigma_[b_ind], c_[d_ind], t_[e_ind]))


def SimpleGraph(G):
    G.remove_edges_from(nx.selfloop_edges(G))
    isol = list(nx.isolates(G))
    G.remove_nodes_from(isol)
    return G, isol


# tune the parameter of the MCMC proposal according to the acceptance rate, to reach optimal acceptance
# Multidimensional Metropolis - Hastings: ~ 30 %
def tune(acceptance, scale, step):  # need iter multiple of t and > t
    acc_rate = (acceptance[len(acceptance)-1]-acceptance[len(acceptance)-step])/step
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1
    return scale