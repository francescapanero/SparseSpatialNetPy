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
from itertools import compress
from scipy.sparse import csr_matrix


# This code runs the posterior inference in parallel (hopefully) for UP TO THREE CHAINS
# G = [G1, G2, G3] networkx graphs (of course, you can specify only one if you don't need the 3 chains!
# iter  = number of iterations
# prior = 'singlepl' or 'doublepl'
# gamma = exponent distance
# size_x = locations x sampled from Unif[0, size_x]
# **kwargs: very long story

def mcmc_chains(G, iter, nburn,
                sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False, x=False, beta=False,
                prior='singlepl', gamma=1, size_x=1, nchain=1, w_inference='HMC',
                sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
                epsilon=0.01, R=5, save_every=1000, plot=False,
                init='none'):

    out = {}

    for i in range(nchain):

        start = time.time()

        if G[i].graph['ground_truth'] == 1:
            out[i] = mcmc_groundtruth(G[i], iter, nburn,
                                      sigma=sigma, c=c, t=t, tau=tau, w0=w0, n=n, u=u, x=x, beta=beta,
                                      w_inference=w_inference, epsilon=epsilon, R=R, save_every=save_every,
                                      sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_x=sigma_x,
                                      sigma_tau=sigma_tau, plot=False, init=init[i])

        else:
            out[i] = mcmc_nogroundtruth(G[i], iter, nburn,
                                        sigma=sigma, c=c, t=t, tau=tau, w0=w0, n=n, u=u, x=x, beta=beta,
                                        prior=prior, gamma=gamma, size_x=size_x,
                                        w_inference=w_inference, epsilon=epsilon, R=R, save_every=save_every,
                                        sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_x=sigma_x,
                                        sigma_tau=sigma_tau, a_t=a_t, b_t=b_t, plot=False, init=init[i])

        end = time.time()
        print('minutes to perform posterior inference (chain ', i+1, '): ', round((end - start) / 60, 2))

    if plot is True:

        plt.figure()
        for i in range(nchain):
            plt.plot(out[i][10], label='chain %i' % i)
        if G[i].graph['log_post']:
            plt.axhline(y=G[i].graph['log_post'], label='true', color='r')
        plt.legend()
        plt.xlabel('iter')
        plt.ylabel('log_post')
<<<<<<< HEAD
        plt.savefig('images/all_rand11/log_post2')
=======
        plt.savefig('images/all_rand11/log_post1')
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
        plt.close()

        if sigma is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][3], label='chain %i' % i)
            if G[i].graph['sigma']:
                plt.axhline(y=G[i].graph['sigma'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('sigma')
<<<<<<< HEAD
            plt.savefig('images/all_rand11/sigma2')
=======
            plt.savefig('images/all_rand11/sigma1')
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
            plt.close()

        if c is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][4], label='chain %i' % i)
            if G[i].graph['c']:
                plt.axhline(y=G[i].graph['c'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('c')
<<<<<<< HEAD
            plt.savefig('images/all_rand11/c2')
=======
            plt.savefig('images/all_rand11/c1')
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
            plt.close()

        if t is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][5], label='chain %i' % i)
            if G[i].graph['t']:
                plt.axhline(y=G[i].graph['t'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('t')
<<<<<<< HEAD
            plt.savefig('images/all_rand11/t2')
=======
            plt.savefig('images/all_rand11/t1')
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
            plt.close()

        if tau is True:
            plt.figure()
            for i in range(nchain):
                plt.plot([i for i in range(0, iter+save_every, save_every)], out[i][6], label='chain %i' %i)
            if G[i].graph['tau']:
                plt.axhline(y=G[i].graph['tau'], label='true', color='r')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('tau')
<<<<<<< HEAD
            plt.savefig('images/all_rand11/tau2')
=======
            plt.savefig('images/all_rand11/tau1')
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
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
                if G[i].nodes[0]['w'] is not None:
                    w = np.array([G[i].nodes[j]['w'] for j in range(size)])
                    biggest_w = w[biggest_deg]
                    plt.axhline(y=biggest_w, label='true')
                plt.xlabel('iter')
                plt.ylabel('highest degree w')
                plt.legend()
<<<<<<< HEAD
                plt.savefig('images/all_rand11/w2_trace_chain%i' % i)
=======
                plt.savefig('images/all_rand11/w1_trace_chain%i' % i)
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
                plt.close()

                w_est_fin = [w_est[k] for k in range(int((nburn+save_every)/save_every),
                                                     int((iter+save_every)/save_every))]
                emp0_ci_95 = [
                    scipy.stats.mstats.mquantiles([w_est_fin[k][j] for k in range(int((iter+save_every)/save_every) -
                                                                                  int((nburn+save_every)/save_every))],
                                                  prob=[0.025, 0.975]) for j in range(size)]
                if G[i].nodes[0]['w'] is not None:
                    w = np.array([G[i].nodes[j]['w'] for j in range(size)])
                    true0_in_ci = [emp0_ci_95[j][0] <= w[j] <= emp0_ci_95[j][1] for j in range(size)]
                    print('posterior coverage of true w (chain %i' % i, ') = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
                num = 50
                sort_ind = np.argsort(deg)
                ind_big1 = sort_ind[range(size - num, size)]
                big_w = w[ind_big1]
                emp_ci_big = []
                for j in range(num):
                    emp_ci_big.append(emp0_ci_95[ind_big1[j]])
                plt.figure()
                plt.subplot(1, 3, 1)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_big[j][0], emp_ci_big[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    plt.plot(j + 1, big_w[j], color='navy', marker='o', markersize=5)
                plt.ylabel('w')
                # smallest deg nodes
                zero_deg = sum(deg == 0)
                ind_small = sort_ind[range(zero_deg, zero_deg + num)]
                small_w = w[ind_small]
                emp_ci_small = []
                for j in range(num):
                    emp_ci_small.append(np.log(emp0_ci_95[ind_small[j]]))
                plt.subplot(1, 3, 2)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_small[j][0], emp_ci_small[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    plt.plot(j + 1, np.log(small_w[j]), color='navy', marker='o', markersize=5)
                plt.ylabel('log w')
                # zero deg nodes
                zero_deg = 0
                ind_small = sort_ind[range(zero_deg, zero_deg + num)]
                small_w = w[ind_small]
                emp_ci_small = []
                for j in range(num):
                    emp_ci_small.append(np.log(emp0_ci_95[ind_small[j]]))
                plt.subplot(1, 3, 3)
                for j in range(num):
                    plt.plot((j + 1, j + 1), (emp_ci_small[j][0], emp_ci_small[j][1]), color='cornflowerblue',
                             linestyle='-', linewidth=2)
                    plt.plot(j + 1, np.log(small_w[j]), color='navy', marker='o', markersize=5)
                plt.ylabel('log w')
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
<<<<<<< HEAD
                plt.savefig('images/all_rand11/w02_CI_chain%i' % i)
=======
                plt.savefig('images/all_rand11/w01_CI_chain%i' % i)
>>>>>>> f7cac6fbfd64d5d57e2d9be497dfda23bed312e1
                plt.close()

    if x is True:
        for i in range(nchain):
            num = 10
            deg = np.array(list(dict(G[i].degree()).values()))
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
            if G[i].graph['distances'] is not None:
                p_ij = G[i].graph['distances']
                true_in_ci = [[emp_ci_95_big[j][k][0] <= p_ij[ind_big1[j], k] <= emp_ci_95_big[j][k][1] for k in range(size)]
                              for j in range(num)]
                print('posterior coverage of true p_ij for highest deg nodes (chain %i' % i, ') = ',
                      [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')
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
            if G[i].graph['distances'] is not None:
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
            if G[i].graph['distances'] is not None:
                true_in_ci = [
                    [emp_ci_95_zero[j][k][0] <= p_ij[ind_zero[j], k] <= emp_ci_95_zero[j][k][1] for k in range(size)]
                    for j in range(num)]
                print('posterior coverage of true p_ij for zero deg nodes (chain %i' % i, ') = ',
                      [round(sum(true_in_ci[j]) / size * 100, 1) for j in range(num)], '%')

    return out


def mcmc_groundtruth(G, iter, nburn,
                     sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False, x=False, beta=False,
                     w_inference='HMC', epsilon=0.01, R=5, save_every=1000,
                     sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01,
                     plot=False,
                     init='none'):

    size = G.number_of_nodes()
    w_true = np.array([G.nodes[i]['w'] for i in range(size)])
    w0_true = np.array([G.nodes[i]['w0'] for i in range(size)])
    beta_true = np.array([G.nodes[i]['beta'] for i in range(size)])
    x_true = np.array([G.nodes[i]['x'] for i in range(size)])
    u_true = np.array([G.nodes[i]['u'] for i in range(size)])
    n_true = G.graph['counts']
    p_ij_true = G.graph['distances']
    ind = G.graph['ind']
    selfedge = G.graph['selfedge']
    log_post_true = G.graph['log_post']
    sigma_true = G.graph['sigma']
    c_true = G.graph['c']
    t_true = G.graph['t']
    tau_true = G.graph['tau']
    gamma = G.graph['gamma']
    size_x = G.graph['size_x']
    prior = G.graph['prior']
    a_t = G.graph['a_t']
    b_t = G.graph['b_t']

    true = {}
    true['sigma_true'] = sigma_true
    true['c_true'] = c_true
    true['t_true'] = t_true
    true['tau_true'] = tau_true
    true['w_true'] = w_true
    true['w0_true'] = w0_true
    true['u_true'] = u_true
    true['beta_true'] = beta_true
    true['n_true'] = n_true
    true['x_true'] = x_true
    true['p_ij_true'] = p_ij_true
    true['log_post_true'] = log_post_true

    output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, size_x,
                       sigma=sigma, c=c, t=t, tau=tau, w0=w0, n=n, u=u, x=x, beta=beta,
                       sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, sigma_x=sigma_x,
                       w_inference=w_inference, epsilon=epsilon, R=R,
                       a_t=a_t, b_t=b_t,
                       plot=plot,
                       ind=ind, selfedge=selfedge,
                       save_every=save_every,
                       init=init, true=true)

    return output


def mcmc_nogroundtruth(G, iter, nburn, prior='singlepl',
                       sigma=False, c=False, t=False, tau=False, w0=False, n=False, u=False, x=False, beta=False,
                       gamma=1, size_x=1,
                       w_inference='HMC', epsilon=0.01, R=5, save_every=1000,
                       sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, sigma_x=0.01, a_t=200, b_t=1,
                       plot=False, init='none'):

    size = G.number_of_nodes()

    output = mcmc.MCMC(prior, G, gamma, size, iter, nburn, size_x,
                       sigma=sigma, c=c, t=t, tau=tau, w0=w0, n=n, u=u, x=x, beta=beta,
                       sigma_sigma=sigma_sigma, sigma_c=sigma_c, sigma_t=sigma_t, sigma_tau=sigma_tau, sigma_x=sigma_x,
                       w_inference=w_inference, epsilon=epsilon, R=R,
                       a_t=a_t, b_t=b_t,
                       plot=plot,
                       save_every=save_every,
                       init=init)

    return output
