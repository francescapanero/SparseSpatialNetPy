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
import os
from itertools import compress
from scipy.sparse import csr_matrix
import pickle
import _pickle as cPickle
import gzip


# This code runs the posterior inference in parallel (hopefully) for UP TO THREE CHAINS
# G = [G1, G2, G3] networkx graphs (of course, you can specify only one if you don't need the 3 chains!
# iter  = number of iterations
# prior = 'singlepl' or 'doublepl'
# gamma = exponent distance
# size_x = locations x sampled from Unif[0, size_x]
# **kwargs: very long story

def mcmc_chains(G, iter, nburn,
                sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False, x=False, beta=False,
                wnu=False, hyperparams=False, all=False,
                sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
                epsilon=0.01, R=5, w_inference='HMC',
                init='none',
                save_out=False, save_data=False, plot=False, path=False, save_every=1000):

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
                      init=init[i])

        end = time.time()

        print('minutes to perform posterior inference (chain ', i+1, '): ', round((end - start) / 60, 2))

    if plot is True:

        for i in range(nchain):
            plt.figure()
            plt.plot(out[i][10], label='chain %i' % i)
            if 'log_post' in G[i].graph:
                plt.axhline(y=G[i].graph['log_post'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('log_post')
            plt.savefig(os.path.join('images', path, 'log_post%i' % i))
            plt.close()

        if sigma is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][3], label='chain %i' % i)
            if 'sigma' in G[i].graph:
                plt.axhline(y=G[i].graph['sigma'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('sigma')
            plt.savefig(os.path.join('images', path, 'sigma'))
            plt.close()

        if c is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][4], label='chain %i' % i)
            if 'c' in G[i].graph:
                plt.axhline(y=G[i].graph['c'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('c')
            plt.savefig(os.path.join('images', path, 'c'))
            plt.close()

        if t is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][5], label='chain %i' % i)
            if 't' in G[i].graph:
                plt.axhline(y=G[i].graph['t'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('t')
            plt.savefig(os.path.join('images', path, 't'))
            plt.close()

        if tau is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][6], label='chain %i' %i)
            if 'tau' in G[i].graph:
                plt.axhline(y=G[i].graph['tau'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('tau')
            plt.savefig(os.path.join('images', path, 'tau'))
            plt.close()

        if w0 is True:
            for i in range(nchain):
                plt.figure()
                w_est = out[i][0]
                deg = np.array(list(dict(G[i].degree()).values()))
                size = len(deg)
                biggest_deg = np.argsort(deg)[-1]
                biggest_w_est = [w_est[i][biggest_deg] for i in range(int((iter+save_every)/save_every))]
                plt.plot([j for j in range(0, iter+save_every, save_every)], biggest_w_est)
                if 'w' in G[i].nodes[0]:
                    w = np.array([G[i].nodes[j]['w'] for j in range(size)])
                    biggest_w = w[biggest_deg]
                    plt.axhline(y=biggest_w, label='true')
                plt.xlabel('iter')
                plt.ylabel('highest degree w')
                plt.legend()
                plt.savefig(os.path.join('images', path, 'w_trace_chain%i' % i))
                plt.close()

                w_est_fin = [w_est[k] for k in range(int((nburn+save_every)/save_every),
                                                     int((iter+save_every)/save_every))]
                emp0_ci_95 = [
                    scipy.stats.mstats.mquantiles([w_est_fin[k][j] for k in range(int((iter+save_every)/save_every) -
                                                                                  int((nburn+save_every)/save_every))],
                                                  prob=[0.025, 0.975]) for j in range(size)]
                if 'w' in G[i].nodes[0]:
                    w = np.array([G[i].nodes[j]['w'] for j in range(size)])
                    true0_in_ci = [emp0_ci_95[j][0] <= w[j] <= emp0_ci_95[j][1] for j in range(size)]
                    print('posterior coverage of true w (chain %i' % i, ') = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
                num = 50
                sort_ind = np.argsort(deg)
                ind_big1 = sort_ind[range(size - num, size)]
                emp_ci_big = []
                for j in range(num):
                    emp_ci_big.append(emp0_ci_95[ind_big1[j]])
                plt.figure()
                plt.subplot(1, 3, 1)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_big[j][0], emp_ci_big[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    if 'w' in G[i].nodes[0]:
                        plt.plot(j + 1, w[ind_big1][j], color='navy', marker='o', markersize=5)
                plt.ylabel('w')
                # smallest deg nodes
                zero_deg = sum(deg == 0)
                ind_small = sort_ind[range(zero_deg, zero_deg + num)]
                emp_ci_small = []
                for j in range(num):
                    emp_ci_small.append(np.log(emp0_ci_95[ind_small[j]]))
                plt.subplot(1, 3, 2)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_small[j][0], emp_ci_small[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    if 'w' in G[i].nodes[0]:
                        plt.plot(j + 1, np.log(w[ind_small][j]), color='navy', marker='o', markersize=5)
                plt.ylabel('log w')
                # zero deg nodes
                zero_deg = 0
                ind_small = sort_ind[range(zero_deg, zero_deg + num)]
                emp_ci_small = []
                for j in range(num):
                    emp_ci_small.append(np.log(emp0_ci_95[ind_small[j]]))
                plt.subplot(1, 3, 3)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_small[j][0], emp_ci_small[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    if 'w' in G[i].nodes[0]:
                        plt.plot(j + 1, np.log(w[ind_small][j]), color='navy', marker='o', markersize=5)
                plt.ylabel('log w')
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
                plt.savefig(os.path.join('images', path, 'w_CI_chain%i' % i))
                plt.close()

        if x is True:
            for i in range(nchain):
                num = 10
                deg = np.array(list(dict(G[i].degree()).values()))
                size = len(deg)
                sort_ind = np.argsort(deg)
                ind_big1 = sort_ind[range(size - num, size)]
                p_ij_est = out[i][11]
                p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn+save_every)/save_every),
                                                     int((iter+save_every)/save_every))] for j in ind_big1]
                emp_ci_95_big = []
                for j in range(num):
                    emp_ci_95_big.append(
                        [scipy.stats.mstats.mquantiles(
                            [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                                  int((nburn + save_every) / save_every))],
                            prob=[0.025, 0.975]) for l in range(size)])
                if 'distances' in G[i].graph:
                    p_ij = G[i].graph['distances']
                    true_in_ci = [[emp_ci_95_big[j][k][0] <= p_ij[ind_big1[j], k] <= emp_ci_95_big[j][k][1]
                                  for k in range(size)] for j in range(num)]
                    print('posterior coverage of true p_ij for highest deg nodes (chain %i' % i, ') = ',
                          [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
                for j in range(num):
                    plt.figure()
                    for k in range(num):
                        plt.plot((k + 1, k + 1), (emp_ci_95_big[j][ind_big1[k]][0], emp_ci_95_big[j][ind_big1[k]][1]),
                             color='cornflowerblue', linestyle='-', linewidth=2)
                        if 'distances' in G[i].graph:
                            plt.plot(k + 1, p_ij[ind_big1[j], ind_big1[k]], color='navy', marker='o', markersize=5)
                    plt.savefig(os.path.join('images', path, 'p_ij_ci_highdeg%i_chain%i' % (j, i)))
                    plt.close()
                if 'distances' in G[i].graph:
                    # smallest deg nodes
                    zero_deg = sum(deg == 0)
                    ind_small = sort_ind[range(zero_deg, zero_deg + num)]
                    p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
                                                                      int((iter + save_every) / save_every))] for j in
                                    ind_small]
                    emp_ci_95_small = []
                    for j in range(num):
                        emp_ci_95_small.append(
                            [scipy.stats.mstats.mquantiles(
                                [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                                      int((nburn + save_every) / save_every))],
                                prob=[0.025, 0.975]) for l in range(size)])
                    true_in_ci = [
                        [emp_ci_95_small[j][k][0] <= p_ij[ind_small[j], k] <= emp_ci_95_small[j][k][1] for k in range(size)]
                        for j in range(num)]
                    print('posterior coverage of true p_ij for smallest deg nodes (chain %i' % i, ') = ',
                          [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
                    # zero deg nodes
                    ind_zero = sort_ind[range(num)]
                    p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
                                                                      int((iter + save_every) / save_every))] for j in
                                    ind_zero]
                    emp_ci_95_zero = []
                    for j in range(num):
                        emp_ci_95_zero.append(
                            [scipy.stats.mstats.mquantiles(
                                [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                                      int((nburn + save_every) / save_every))],
                                prob=[0.025, 0.975]) for l in range(size)])
                    true_in_ci = [
                        [emp_ci_95_zero[j][k][0] <= p_ij[ind_zero[j], k] <= emp_ci_95_zero[j][k][1] for k in range(size)]
                        for j in range(num)]
                    print('posterior coverage of true p_ij for zero deg nodes (chain %i' % i, ') = ',
                          [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')

        # if n is True:
        #     for i in range(nchain):
        #         num = 10
        #         deg = np.array(list(dict(G[i].degree()).values()))
        #         size = len(deg)
        #         n_est = out[i][7]
        #         sort_ind = np.argsort(deg)
        #         ind_big1 = sort_ind[range(size - num, size)]
        #         n_est_fin = [[n_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
        #                                                     int((iter + save_every) / save_every))] for j in ind_big1]
        #         emp_ci_95_big = []
        #         for j in range(num):
        #             emp_ci_95_big.append(
        #                 [scipy.stats.mstats.mquantiles(
        #                     [n_est_fin[j][k][0, l] for k in range(int((iter + save_every) / save_every) -
        #                                                        int((nburn + save_every) / save_every))],
        #                     prob=[0.025, 0.975]) for l in range(size)])
        #         if G[i].graph['counts'] is not None:
        #             n = G[i].graph['counts']
        #             true_in_ci = [[emp_ci_95_big[j][k][0] <= n[ind_big1[j], k] <= emp_ci_95_big[j][k][1] for k in range(size)]
        #                           for j in range(num)]
        #             print('posterior coverage of true n for highest deg nodes (chain %i' % i, ') = ',
        #                   [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
        #         # smallest deg nodes
        #         zero_deg = sum(deg == 0)
        #         ind_small = sort_ind[range(zero_deg, zero_deg + num)]
        #         n_est_fin = [[n_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
        #                                                     int((iter + save_every) / save_every))] for j in
        #                      ind_small]
        #         emp_ci_95_small = []
        #         for j in range(num):
        #             emp_ci_95_small.append(
        #                 [scipy.stats.mstats.mquantiles(
        #                     [n_est_fin[j][k][0, l] for k in range(int((iter + save_every) / save_every) -
        #                                                        int((nburn + save_every) / save_every))],
        #                     prob=[0.025, 0.975]) for l in range(size)])
        #         if G[i].graph['counts'] is not None:
        #             true_in_ci = [
        #                 [emp_ci_95_small[j][k][0] <= n[ind_small[j], k] <= emp_ci_95_small[j][k][1] for k in range(size)]
        #                 for j in range(num)]
        #             print('posterior coverage of true n for smallest deg nodes (chain %i' % i, ') = ',
        #                   [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
        #         # zero deg nodes
        #         ind_zero = sort_ind[range(num)]
        #         n_est_fin = [[n_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
        #                                                     int((iter + save_every) / save_every))] for j in
        #                      ind_zero]
        #         emp_ci_95_zero = []
        #         for j in range(num):
        #             emp_ci_95_zero.append(
        #                 [scipy.stats.mstats.mquantiles(
        #                     [n_est_fin[j][k][0, l] for k in range(int((iter + save_every) / save_every) -
        #                                                        int((nburn + save_every) / save_every))],
        #                     prob=[0.025, 0.975]) for l in range(size)])
        #         if G[i].graph['counts'] is not None:
        #             true_in_ci = [
        #                 [emp_ci_95_zero[j][k][0] <= n[ind_zero[j], k] <= emp_ci_95_zero[j][k][1] for k in range(size)]
        #                 for j in range(num)]
        #             print('posterior coverage of true n for zero deg nodes (chain %i' % i, ') = ',
        #                   [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')

    if save_out is True:
        # with open(os.path.join('data_outputs', path, 'out.pickle'), 'wb') as f:
        #     compressed_pickle.dump(out, f)
        save_zipped_pickle(out, os.path.join('data_outputs', path, 'out.pickle'))

    return out


def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol)


def mcmc(G, iter, nburn,
         w0=False, beta=False, n=False, u=False, sigma=False, c=False, t=False, tau=False, x=False,
         hyperparams=False, wnu=False, all=False,
         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
         epsilon=0.01, R=5, w_inference='HMC',
         save_every=1000,
         init='none'):

    size = G.number_of_nodes()
    prior = G.graph['prior'] if 'prior' in G.graph else print('You must specify a prior as attribute of G')
    gamma = G.graph['gamma'] if 'gamma' in G.graph else print('You must specify spatial exponent gamma as attribute of G')
    size_x = G.graph['size_x'] if 'size_x' in G.graph else print('You must specify size_x as attribute of G')

    if hyperparams is True or all is True:
        sigma = c = t = tau = True
        if prior == 'singlepl':
            tau = False
    if wnu is True or all is True:
        w0 = beta = n = u = x = True
        if prior == 'singlepl':
            beta = False
    if sigma is True:
        sigma_est = [init['sigma_init']] if 'sigma_init' in init else [float(np.random.rand(1))]
    else:
        sigma_est = [G.graph['sigma']]
    if c is True:
        c_est = [init['c_init']] if 'c_init' in init else [float(5 * np.random.rand(1) + 1)]
    else:
        c_est = [G.graph['c']]
    if t is True:
        t_est = [init['t_init']] if 't_init' in init else [float(np.random.gamma(a_t, 1 / b_t))]
    else:
        t_est = [G.graph['t']]
    if prior == 'doublepl':
        if tau is True:
            tau_est = [init['tau_init']] if 'tau_init' in init else [float(5 * np.random.rand(1) + 1)]
        else:
            tau_est = [G.graph['tau']]
    else:
        tau_est = [0]

    z_est = [(size * sigma_est[0] / t_est[0]) ** (1 / sigma_est[0])] if G.graph['prior'] == 'singlepl' else \
                 [(size * tau_est[0] * sigma_est[0] ** 2 / (t_est[0] * c_est[0] ** (sigma_est[0] * (tau_est[0] - 1)))) ** \
                 (1 / sigma_est[0])]

    if w0 is True:
        if 'w0_init' in init:
            w0_est = [init['w0_init']]
        else:
            g = np.random.gamma(1 - sigma_est[0], 1, size)
            unif = np.random.rand(size)
            w0_est = [np.multiply(g, np.power(((z_est[0] + c_est[0]) ** sigma_est[0]) * (1 - unif) +
                                          (c_est[0] ** sigma_est[0]) * unif, -1 / sigma_est[0]))]
    else:
        w0_est = [np.array([G.nodes[i]['w0'] for i in range(G.number_of_nodes())])]
    if prior == 'doublepl' and beta is True:
        beta_est = [init['beta_init']] if 'beta_init' in init else [float(np.random.beta(sigma_est[0] * tau_est[0], 1))]
    if prior == 'singlepl' or beta is False:
        beta_est = [np.array([G.nodes[i]['beta'] for i in range(G.number_of_nodes())])] if 'beta' in G.nodes[0] \
            else [np.ones((size))]
    if u is True:
        u_est = [init['u_init']] if 'u_init' in init else [tp.tpoissrnd(z_est[0] * w0_est[0])]
    else:
        u_est = [np.array([G.nodes[i]['u'] for i in range(G.number_of_nodes())])]
    if x is True:
        x_est = init['x_init'] if 'x_init' in init else size_x * np.random.rand(size)
        p_ij_est = [aux.space_distance(x_est, gamma)]
    else:
        x_est = np.array([G.nodes[i]['x'] for i in range(G.number_of_nodes())])
        p_ij_est = [aux.space_distance(x_est, gamma)]
    if 'ind' in G.graph:
        ind = G.graph['ind']
    else:
        ind = {k: [] for k in G.nodes}
        for i in G.nodes:
            for j in G.adj[i]:
                if j > i:
                    ind[i].append(j)
    if 'selfedge' in G.graph:
        selfedge = G.graph['selfedge']
    else:
        selfedge = [i in ind[i] for i in G.nodes]
        selfedge = list(compress(G.nodes, selfedge))
    if n is True:
        if 'n_init' in init:
            n_est = [init['n_init']]
            sum_fact_n = init['sum_fact_n']
        else:
            out_n = up.update_n(w0_est[0], G, size, p_ij_est[-1], ind, selfedge)
            n_est = [out_n[0]]
            sum_fact_n = out_n[1]
    else:
        n_est = [G.graph['counts']]
        sum_fact_n = G.graph['sum_fact_n']

    w_est = [np.exp(np.log(w0_est[0]) - np.log(beta_est[0]))]

    log_post_param_est = [aux.log_post_params(prior, sigma_est[-1], c_est[-1], t_est[-1], tau_est[-1],
                                        w0_est[-1], beta_est[-1], u_est[-1], a_t, b_t)]
    sum_n = np.array(csr_matrix.sum(n_est[-1], axis=0) + np.transpose(csr_matrix.sum(n_est[-1], axis=1)))[0]
    log_post_est = [aux.log_post_logwbeta_params(prior, sigma_est[-1], c_est[-1], t_est[-1], tau_est[-1], w_est[-1],
                                                 w0_est[-1], beta_est[-1], n_est[-1], u_est[-1], p_ij_est[-1], a_t, b_t,
                                                 gamma, sum_n, sum_fact_n, log_post_par=log_post_param_est[-1])]

    accept_params = [0]
    accept_hmc = 0
    accept_distance = 0
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
    x_prev = x_est
    p_ij_prev = p_ij_est[-1]
    u_prev = u_est[-1]
    z_prev = z_est[-1]

    for i in range(iter):

        # update hyperparameters if at least one of them demands the update
        if sigma is True or c is True or t is True or tau is True:
            output_params = up.update_params(prior, sigma_prev, c_prev, t_prev, tau_prev, z_prev,
                                             w0_prev, beta_prev, u_prev, log_post_param_est[-1],
                                             accept_params[-1],
                                             sigma=sigma, c=c, t=t, tau=tau,
                                             sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t,
                                             sigma_tau=sigma_tau, a_t=a_t, b_t=b_t)
            sigma_prev = output_params[0]
            c_prev = output_params[1]
            t_prev = output_params[2]
            tau_prev = output_params[3]
            z_prev = output_params[4]
            accept_params.append(output_params[5])
            log_post_param_est.append(output_params[6])
            rate_p.append(output_params[7])
            if (i + 1) % save_every == 0 and i != 0:
                sigma_est.append(sigma_prev)
                c_est.append(c_prev)
                t_est.append(t_prev)
                tau_est.append(tau_prev)
                z_est.append(z_prev)
            if i % 1000 == 0:
                print('update hyperparams iteration = ', i)
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
            if accept_params[-1] == 0:
                log_post_est.append(log_post_est[-1])
            if accept_params[-1] == 1:
                log_post_est.append(aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev,
                                                                 tau_prev, w_prev, w0_prev, beta_prev,
                                                                 n_prev, u_prev, p_ij_prev, a_t, b_t, gamma,
                                                                 sum_n, sum_fact_n, log_post_par=log_post_param_est[-1]))
            if w_inference == 'gibbs':
                output_gibbs = up.gibbs_w(w_prev, beta_prev, sigma_prev, c_prev, z_prev,
                                          u_prev, n_prev, p_ij_prev, gamma, sum_n)
                w_prev = output_gibbs[0]
                w0_prev = output_gibbs[1]
                log_post_param_est.append(
                    aux.log_post_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                        w0_prev, beta_prev, u_prev, a_t, b_t))
                log_post_est.append(aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev,
                                                                 tau_prev, w_prev, w0_prev, beta_prev,
                                                                 n_prev, u_prev, p_ij_prev, a_t, b_t,
                                                                 gamma, sum_n, sum_fact_n,
                                                                 log_post=log_post_param_est[-1]))
                if (i + 1) % save_every == 0 and i != 0:
                    w_est.append(w_prev)
                    w0_est.append(w0_prev)
                    beta_est.append(beta_prev)
                if i % 1000 == 0 and i != 0:
                    print('update w iteration = ', i)
            if w_inference == 'HMC':
                output_hmc = up.HMC_w(prior, w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                      sigma_prev, c_prev, t_prev, tau_prev, z_prev, gamma,
                                      p_ij_prev, a_t, b_t, epsilon, R, accept_hmc, size, sum_n, sum_fact_n,
                                      log_post_est[-1], log_post_param_est[-1], update_beta=beta)
                w_prev = output_hmc[0]
                w0_prev = output_hmc[1]
                beta_prev = output_hmc[2]
                accept_hmc = output_hmc[3]
                rate.append(output_hmc[4])
                log_post_est.append(output_hmc[5])
                log_post_param_est.append(output_hmc[6])
                if (i + 1) % save_every == 0 and i != 0:
                    w_est.append(w_prev)
                    w0_est.append(w0_prev)
                    beta_est.append(beta_prev)
                if i % 100 == 0 and i != 0:
                    # if i < nadapt:
                    if i >= step:
                        # epsilon = np.exp(np.log(epsilon) + 0.01 * (np.mean(rate) - 0.6))
                        epsilon = np.exp(np.log(epsilon) + 0.01 * (np.mean(rate[i-step:i]) - 0.6))
                if i % 1000 == 0:
                    print('update w and beta iteration = ', i)
                    print('acceptance rate HMC = ', round(accept_hmc / (i + 1) * 100, 1), '%')
                    print('epsilon = ', epsilon)

        # update n
        if n is True and (i + 1) % 25 == 0:
            out_n = up.update_n(w_prev, G, size, p_ij_prev, ind, selfedge)
            n_prev = out_n[0]
            sum_fact_n = out_n[1]
            sum_n = np.array(csr_matrix.sum(n_prev, axis=0) + np.transpose(csr_matrix.sum(n_prev, axis=1)))[0]
            log_post_param_est.append(log_post_param_est[-1])
            log_post_est.append(aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                             w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                                             p_ij_prev, a_t, b_t, gamma, sum_n, sum_fact_n,
                                                             log_post_par=log_post_param_est[-1]))
            if (i + 1) % save_every == 0 and i != 0:
                n_est.append(n_prev)
            if i % 1000 == 0:
                print('update n iteration = ', i)

        # update u
        if u is True:
            u_prev = up.posterior_u(z_prev * w0_prev)
            log_post_param_est.append(aux.log_post_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                          w0_prev, beta_prev, u_prev, a_t, b_t))
            log_post_est.append(aux.log_post_logwbeta_params(prior, sigma_prev, c_prev, t_prev, tau_prev,
                                                             w_prev, w0_prev, beta_prev, n_prev, u_prev,
                                                             p_ij_prev, a_t, b_t, gamma, sum_n, sum_fact_n,
                                                             log_post_par=log_post_param_est[-1]))
            if (i + 1) % save_every == 0 and i != 0:
                u_est.append(u_prev)
            if i % 1000 == 0:
                print('update u iteration = ', i)

        if x is True:  # and (i + 1) % 25 == 0:
            out = up.update_x(x_prev, w_prev, gamma, p_ij_prev, n_prev, sigma_x, accept_distance, prior, sigma_prev,
                              c_prev, t_prev, tau_prev, w0_prev, beta_prev, u_prev, a_t, b_t, sum_n, sum_fact_n,
                              log_post_est[-1], log_post_param_est[-1])
            x_prev = out[0]
            p_ij_prev = out[1]
            accept_distance = out[2]
            log_post_est.append(out[3])
            if (i + 1) % save_every == 0 and i != 0:
                p_ij_est.append(p_ij_prev)
            if i % 1000 == 0:
                print('update x iteration = ', i)
                # print('acceptance rate x = ', round(accept_distance * 25 * 100 / iter, 1), '%')
                print('acceptance rate x = ', round(accept_distance * 100 / iter, 1), '%')

    return w_est, w0_est, beta_est, sigma_est, c_est, t_est, tau_est, n_est, u_est, \
           log_post_param_est, log_post_est, p_ij_est
