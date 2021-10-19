from utils.GraphSampler import *
import utils.TruncPois as tp
import utils.Auxiliary as aux
import utils.Updates as up
import utils.PlotMCMC as PlotMCMC
import numpy as np
import os
from itertools import compress
from scipy.sparse import csr_matrix
import _pickle as cPickle
import gzip
import scipy


# wrapper function used for debugging the inference of locations x.
# G: networkx graph
# iter, nburn: number of iterations, burn in
# save_every: the chain saves the values once every save_every iterations
# sigma_x: stepsize of MH proposal for locations
# index: the index of the locations we want to update (could be a subset of the total. the rest would be fixed to their
#           true values)
# init: initialisation values for x

def mcmc_debug_x(G, iter, nburn, save_every, sigma_x, index, init=0):

    out = mcmc(G, iter, nburn,
               sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False,
               x=True,
               beta=False, wnu=False, hyperparams=False, all=False,
               w_inference='HMC', epsilon=0.05, R=5, save_every=save_every,
               sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_x=sigma_x,
               sigma_tau=0.01, a_t=200, b_t=1,
               init=init, index=index)

    return out


# This code runs the posterior inference in parallel (hopefully) for UP TO THREE CHAINS
# G = [G1, G2, G3] networkx graphs (of course, you can specify only one if you don't need the 3 chains!
# iter  = number of iterations
# prior = 'singlepl' or 'doublepl'
# gamma = exponent distance
# size_x = locations x sampled from Unif[0, size_x]
# **kwargs: very long story

def mcmc_chains(G, iter, nburn, index,
                sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False, x=False, beta=False,
                wnu=False, hyperparams=False, all=False,
                sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
                epsilon=0.01, R=5, w_inference='HMC',
                init='none',
                save_out=False, save_data=False, plot=False, path=False, save_every=1000,):

    if plot is True:
        os.mkdir(os.path.join('images', path))
    if save_out is True:
        os.mkdir(os.path.join('data_outputs', path))

    if save_data is True:
        # with open(os.path.join('data_outputs', path, 'G.pickle'), 'wb') as f:
        #     pickle.dump(G, f)
        save_zipped_pickle(G, os.path.join('data_outputs', path, 'G.pickle'))

    nchain = len(G)
    out = {}

    for i in range(nchain):

        start = time.time()
        out[i] = mcmc(G[i], iter, nburn,
                      sigma=sigma, c=c, t=t, tau=tau, w0=w0, n=n, u=u, x=x, beta=beta,
                      wnu=wnu, hyperparams=hyperparams, all=all,
                      w_inference=w_inference, epsilon=epsilon, R=R, save_every=save_every,
                      sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_x=sigma_x,
                      sigma_tau=sigma_tau, a_t=a_t, b_t=b_t,
                      init=init[i], index=index)
        end = time.time()

        print('minutes to perform posterior inference (chain ', i+1, '): ', round((end - start) / 60, 2))

        if save_out is True:
            # with open(os.path.join('data_outputs', path, 'out.pickle'), 'wb') as f:
            #     compressed_pickle.dump(out[11], f)
            np.savetxt(os.path.join('data_outputs', path, 'x_%i.txt' % i), out[i][11][-1], delimiter=",")

    if plot is True:

        PlotMCMC.plot(out, G, path,
                      sigma, c, tau, t, w0, x,
                      iter, nburn, save_every)

        if x is True:
            PlotMCMC.plot_space_debug(out, G, iter, nburn, save_every, index, path)

    return out


def init_var(G, size, gamma, init, w0, beta, n, u, sigma, c, t, tau, x, hyperparams, wnu, all, prior, a_t, b_t, size_x):

    if hyperparams is True or all is True:
        sigma = c = t = tau = True
        if prior == 'singlepl':
            tau = False
    if wnu is True or all is True:
        w0 = beta = n = u = x = True
        if prior == 'singlepl':
            beta = False
    if sigma is True:
        sigma_est = [init['sigma']] if 'sigma' in init else [float(np.random.rand(1))]
    else:
        sigma_est = [G.graph['sigma']]
    if c is True:
        c_est = [init['c']] if 'c' in init else [float(5 * np.random.rand(1) + 1)]
    else:
        c_est = [G.graph['c']]
    if t is True:
        t_est = [init['t']] if 't' in init else [float(np.random.gamma(a_t, 1 / b_t))]
    else:
        t_est = [G.graph['t']]
    if prior == 'doublepl':
        if tau is True:
            tau_est = [init['tau']] if 'tau' in init else [float(5 * np.random.rand(1) + 1)]
        else:
            tau_est = [G.graph['tau']]
    else:
        tau_est = [0]

    z_est = [(size * sigma_est[0] / t_est[0]) ** (1 / sigma_est[0])] if G.graph['prior'] == 'singlepl' else \
        [(size * tau_est[0] * sigma_est[0] ** 2 / (t_est[0] * c_est[0] ** (sigma_est[0] * (tau_est[0] - 1)))) ** \
         (1 / sigma_est[0])]

    if w0 is True:
        if 'w0' in init:
            w0_est = [init['w0']]
        else:
            g = np.random.gamma(1 - sigma_est[0], 1, size)
            unif = np.random.rand(size)
            w0_est = [np.multiply(g, np.power(((z_est[0] + c_est[0]) ** sigma_est[0]) * (1 - unif) +
                                              (c_est[0] ** sigma_est[0]) * unif, -1 / sigma_est[0]))]
    else:
        w0_est = [np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])]
    if prior == 'doublepl' and beta is True:
        beta_est = [init['beta']] if 'beta' in init else [float(np.random.beta(sigma_est[0] * tau_est[0], 1))]
    if prior == 'singlepl' or beta is False:
        beta_est = [np.array([G.nodes[i]['beta'] for i in range(G.number_of_nodes())])] if 'beta' in G.nodes[0] \
            else [np.ones((size))]
    if u is True:
        u_est = [init['u']] if 'u' in init else [tp.tpoissrnd(z_est[0] * w0_est[0])]
    else:
        u_est = [np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])]
    if x is True:
        x_est = [init['x']] if 'x' in init else [size_x * np.random.rand(size)]  # uniform prior
        # x_est = [init['x']] if 'x' in init else [scipy.stats.norm.rvs(3, 0.1, size)]  # normal prior
        p_ij_est = [aux.space_distance(x_est[-1], gamma)]
    else:
        if gamma != 0:
            x_est = [np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])]
            p_ij_est = [G.graph['distances']]
        else:
            x_est = [np.ones(G.number_of_nodes())]
            p_ij_est = [np.ones((G.number_of_nodes(), G.number_of_nodes()))]
    if 'ind' in G.graph:
        ind = G.graph['ind']
    else:
        ind = {k: [] for k in G.nodes}
        for i in G.nodes:
            for j in G.adj[i]:
                if j >= i:
                    ind[i].append(j)
    if 'selfedge' in G.graph:
        selfedge = G.graph['selfedge']
    else:
        selfedge = [i in ind[i] for i in G.nodes]
        selfedge = list(compress(G.nodes, selfedge))
    if n is True:
        if 'n' in init:
            n_est = [init['n']]
        else:
            out_n = up.update_n(w0_est[0], G, size, p_ij_est[-1], ind, selfedge)
            n_est = [out_n[0]]
    else:
        n_est = [G.graph['counts']]

    w_est = [np.exp(np.log(w0_est[0]) - np.log(beta_est[0]))]

    # ## speed up - only x
    # x_est = [x_est[-1][index]]
    # n_est = [n_est[-1][index, :]]
    # n_est = [n_est[-1][:, index]]
    # p_ij_est = [p_ij_est[-1][:, index]]
    # p_ij_est = [p_ij_est[-1][index, :]]
    # adj = adj[index, :]
    # adj = adj[:, index]
    # w_est = [w_est[-1][index]]
    # ## speed up - only x

    return sigma_est, c_est, t_est, tau_est, w_est, w0_est, beta_est, n_est, x_est, p_ij_est, u_est, z_est, ind, \
           selfedge


# main code to run the mcmc
def mcmc(G, iter, nburn,
         w0=False, beta=False, n=False, u=False, sigma=False, c=False, t=False, tau=False, x=False,
         hyperparams=False, wnu=False, all=False,
         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
         epsilon=0.01, R=5, w_inference='HMC',
         save_every=1000,
         init='none',
         index=None):

    size = G.number_of_nodes()
    prior = G.graph['prior'] if 'prior' in G.graph else print('You must specify a prior as attribute of G')
    gamma = G.graph['gamma'] if 'gamma' in G.graph else print('You must specify spatial exponent gamma as G attribute')
    size_x = G.graph['size_x'] if 'size_x' in G.graph else print('You must specify size_x as attribute of G')

    sigma_est, c_est, t_est, tau_est, w_est, w0_est, beta_est, n_est, x_est, p_ij_est, u_est, z_est, ind, selfedge = \
     init_var(G, size, gamma, init, w0, beta, n, u, sigma, c, t, tau, x, hyperparams, wnu, all, prior, a_t, b_t, size_x)

    accept_params = [0]
    accept_hmc = 0
    accept_distance = [0]
    rate = [0]
    rate_p = [0]
    step = 100
    nadapt = 1000

    sigma_prev = sigma_est[-1]
    c_prev = c_est[-1]
    t_prev = t_est[-1]
    tau_prev = tau_est[-1]
    w_prev = w_est[-1]
    w0_prev = w0_est[-1]
    beta_prev = beta_est[-1]
    n_prev = n_est[-1]
    x_prev = x_est[-1]
    p_ij_prev = p_ij_est[-1]
    u_prev = u_est[-1]
    z_prev = z_est[-1]
    sum_n = np.array(csr_matrix.sum(n_prev, axis=0) + np.transpose(csr_matrix.sum(n_prev, axis=1)))[0]
    adj = n_prev > 0
    log_post_param_prev = aux.log_post_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                              w0_prev, beta_prev, u_prev, a_t, b_t)
    log_post_prev = aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev,
                                                 tau_prev, w_prev, w0_prev, beta_prev,
                                                 n_prev, u_prev, p_ij_prev, a_t, b_t, gamma,
                                                 sum_n, adj, x_prev, log_post_par=log_post_param_prev)

    log_post_param_est = [log_post_param_prev]
    log_post_est = [log_post_prev]

    for i in range(iter):

        # update hyperparameters if at least one of them demands the update
        if sigma is True or c is True or t is True or tau is True:
            sigma_prev, c_prev, t_prev, tau_prev, z_prev, accept_param_prev, log_post_param_prev, rate_p_prev \
                = up.update_params(prior, sigma_prev, c_prev, t_prev, tau_prev, z_prev,
                                   w0_prev, beta_prev, u_prev, log_post_param_prev, accept_params[-1],
                                   sigma=sigma, c=c, t=t, tau=tau,
                                   sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t,
                                   sigma_tau=sigma_tau, a_t=a_t, b_t=b_t)
            accept_params.append(accept_param_prev)
            rate_p.append(rate_p_prev)
            # if you only have to update hyperparams, then log_post = log_post_param, otherwise you need to update that
            if w0 is True or n is True or u is True or x is True:
                log_post_prev = aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev,
                                                             tau_prev, w_prev, w0_prev, beta_prev,
                                                             n_prev, u_prev, p_ij_prev, a_t, b_t, gamma,
                                                             sum_n, adj, x_prev, log_post_par=log_post_param_prev)
            if (i + 1) % save_every == 0 and i != 0:
                sigma_est.append(sigma_prev)
                c_est.append(c_prev)
                t_est.append(t_prev)
                tau_est.append(tau_prev)
                z_est.append(z_prev)
                log_post_param_est.append(log_post_param_prev)
                if w0 is True or n is True or u is True or x is True:
                    log_post_est.append(log_post_prev)
            if i % 1000 == 0:
                print('update hyperparams iteration ', i)
                print('acceptance rate hyperparams = ', round(accept_params[-1] / (i+1) * 100, 1), '%')
            if (i % step) == 0 and i != 0 and i < nburn:
                if sigma is True:
                    sigma_sigma = aux.tune(accept_params, sigma_sigma, step)
                if c is True:
                    sigma_c = aux.tune(accept_params, sigma_c, step)
                if t is True:
                    sigma_t = aux.tune(accept_params, sigma_t, step)
                if tau is True:
                    sigma_tau = aux.tune(accept_params, sigma_tau, step)

        # update w and beta if at least one of them is True
        if w0 is True:

            if w_inference == 'gibbs':
                w_prev, w0_prev = up.gibbs_w(w_prev, beta_prev, sigma_prev, c_prev, z_prev,
                                             u_prev, n_prev, p_ij_prev, gamma, sum_n)
                log_post_param_prev = aux.log_post_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                          w0_prev, beta_prev, u_prev, a_t, b_t)
                log_post_prev = aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev,
                                                             tau_prev, w_prev, w0_prev, beta_prev,
                                                             n_prev, u_prev, p_ij_prev, a_t, b_t,
                                                             gamma, sum_n, adj, x_prev,
                                                             log_post=log_post_param_prev)
                if (i + 1) % save_every == 0 and i != 0:
                    w_est.append(w_prev)
                    w0_est.append(w0_prev)
                    beta_est.append(beta_prev)
                    log_post_est.append(log_post_prev)
                    log_post_param_est.append(log_post_param_prev)
                if i % 1000 == 0 and i != 0:
                    print('update w iteration ', i)
            if w_inference == 'HMC':
                w_prev, w0_prev, beta_prev, accept_hmc, rate_prev, log_post_prev, log_post_param_prev \
                            = up.HMC_w(prior, w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                       sigma_prev, c_prev, t_prev, tau_prev, z_prev, gamma,
                                       p_ij_prev, a_t, b_t, epsilon, R, accept_hmc, size, sum_n, adj, x_prev,
                                       log_post_prev, log_post_param_prev, update_beta=beta)
                rate.append(rate_prev)
                if (i + 1) % save_every == 0 and i != 0:
                    w_est.append(w_prev)
                    w0_est.append(w0_prev)
                    beta_est.append(beta_prev)
                    log_post_est.append(log_post_prev)
                    log_post_param_est.append(log_post_param_prev)
                if i % 100 == 0 and i != 0:
                    # if i < nadapt:
                    if i >= step:
                        # epsilon = np.exp(np.log(epsilon) + 0.01 * (np.mean(rate) - 0.6))
                        epsilon = np.exp(np.log(epsilon) + 0.01 * (np.mean(rate[i-step:i]) - 0.6))
                if i % 1000 == 0:
                    print('update w and beta iteration ', i)
                    print('acceptance rate HMC = ', round(accept_hmc / (i + 1) * 100, 1), '%')
                    print('epsilon = ', epsilon)

        # update n
        step_n = 25
        if n is True and (i + 1) % step_n == 0:
            n_prev, rubbish = up.update_n(w_prev, G, size, p_ij_prev, ind, selfedge)
            sum_n = np.array(csr_matrix.sum(n_prev, axis=0) + np.transpose(csr_matrix.sum(n_prev, axis=1)))[0]
            log_post_prev = aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                         w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                                         p_ij_prev, a_t, b_t, gamma, sum_n, adj, x_prev,
                                                         log_post_par=log_post_param_prev)
            if (i + 1) % save_every == 0 and i != 0:
                n_est.append(n_prev)
                log_post_param_est.append(log_post_param_prev)
                log_post_est.append(log_post_prev)
            if i % 1000 == 0:
                print('update n iteration ', i)

        # update u
        if u is True:
            u_prev = up.posterior_u(z_prev * w0_prev)
            log_post_param_prev = aux.log_post_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                      w0_prev, beta_prev, u_prev, a_t, b_t)
            log_post_prev = aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                         w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                                         p_ij_prev, a_t, b_t, gamma, sum_n, adj, x_prev,
                                                         log_post_par=log_post_param_prev)
            if (i + 1) % save_every == 0 and i != 0:
                u_est.append(u_prev)
                log_post_param_est.append(log_post_param_prev)
                log_post_est.append(log_post_prev)
            if i % 1000 == 0:
                print('update u iteration ', i)

        step_x = 1
        if x is True and (i + 1) % step_x == 0:
            x_prev, p_ij_prev, accept_distance_prev, log_post_prev = \
                up.update_x(x_prev, w_prev, gamma, p_ij_prev, n_prev, sigma_x, accept_distance[-1], prior,
                            sigma_prev, c_prev, t_prev, tau_prev, w0_prev, beta_prev, u_prev, a_t, b_t, sum_n, adj,
                            log_post_prev, log_post_param_prev, index)
            accept_distance.append(accept_distance_prev)
            if (i + 1) % save_every == 0 and i != 0:
                p_ij_est.append(p_ij_prev)
                x_est.append(x_prev)
                log_post_param_est.append(log_post_param_prev)
                log_post_est.append(log_post_prev)
            if i % 1000 == 0:
                print('update x iteration ', i)
                print('acceptance rate x = ', round(accept_distance[-1] * 100 * step_x / (i+1), 1), '%')
                print('sigma_x = ', sigma_x)
            if (i % (step/step_x)) == 0 and i != 0 and i < nburn:
                    sigma_x = aux.tune(accept_distance, sigma_x, int(step/step_x))

    if gamma != 0:
        return w_est, w0_est, beta_est, sigma_est, c_est, t_est, tau_est, n_est, u_est, \
                log_post_param_est, log_post_est, p_ij_est, x_est
    else:
        return w_est, w0_est, beta_est, sigma_est, c_est, t_est, tau_est, n_est, u_est, \
                log_post_param_est, log_post_est, p_ij_est


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)