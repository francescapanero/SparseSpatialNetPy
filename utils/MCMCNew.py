import utils.UpdatesNew as up
import matplotlib.pyplot as plt
import utils.TruncPois as tp
import numpy as np
import utils.AuxiliaryNew as aux
import scipy


def MCMC(prior, G, gamma, size, iter, nburn, w_inference='none', p_ij='None', epsilon=0.01, R=5,
         w0=False, beta=False, n=False, u=False, sigma=False, c=False, t=False, tau=False,
         hyperparams=False, wnu=False, all=False,
         sigma_sigma=0.01, sigma_c=0.01, sigma_t=0.01, sigma_tau=0.01, a_t=200, b_t=1,
         plot=True, **kwargs):

    if hyperparams is True or all is True:
        sigma = c = t = tau = True
        if prior == 'singlepl':
            tau = False
    if wnu is True or all is True:
        w0 = beta = n = u = True
        if prior == 'singlepl':
            beta = False
    if sigma is True:
        sigma_est = [kwargs['sigma_init']] if 'sigma_init' in kwargs else [np.random.rand(1)]
    else:
        sigma_est = [kwargs['sigma_true']]
    if c is True:
        c_est = [kwargs['c_init']] if 'c_init' in kwargs else [5 * np.random.rand(1) + 1]
    else:
        c_est = [kwargs['c_true']]
    if t is True:
        t_est = [kwargs['t_init']] if 't_init' in kwargs else [np.random.gamma(a_t, 1 / b_t)]
    else:
        t_est = [kwargs['t_true']]
    if prior == 'doublepl':
        if tau is True:
            tau_est = [kwargs['tau_init']] if 'tau_init' in kwargs else [5 * np.random.rand(1) + 1]
        else:
            tau_est = [kwargs['tau_true']]
    else:
        tau_est = [0]

    z_est = [(size * sigma_est[0] / t_est[0]) ** (1 / sigma_est[0])] if prior == 'singlepl' else \
                 [(size * tau_est[0] * sigma_est[0] ** 2 / (t_est[0] * c_est[0] ** (sigma_est[0] * (tau_est[0] - 1)))) ** \
                 (1 / sigma_est[0])]

    if w0 is True:
        if 'w0_init' in kwargs:
            w0_est = [kwargs['w0_init']]
        else:
            g = np.random.gamma(1 - sigma_est[0], 1, size)
            unif = np.random.rand(size)
            w0_est = [np.multiply(g, np.power(((z_est[0] + c_est[0]) ** sigma_est[0]) * (1 - unif) +
                                          (c_est[0] ** sigma_est[0]) * unif, -1 / sigma_est[0]))]
    else:
        w0_est = [kwargs['w0_true']]
    if prior == 'doublepl' and beta is True:
        beta_est = [kwargs['beta_init']] if 'beta_init' in kwargs else [np.random.beta(sigma_est[0] * tau_est[0], 1)]
    if prior == 'singlepl' or beta is False:
        beta_est = [kwargs['beta_true']]
    if u is True:
        u_est = [kwargs['u_init']] if 'u_init' in kwargs else [tp.tpoissrnd(z_est[0] * w0_est[0])]
    else:
        u_est = [kwargs['u_true']]
    if n is True:
        n_est = [kwargs['n_init']] if 'n_init' in kwargs else [up.update_n(w0_est[0], G, size, p_ij)]
    else:
        n_est = [kwargs['n_true']]

    w_est = [np.exp(np.log(w0_est[0]) - np.log(beta_est[0]))]
    log_post_est = [aux.log_post_params(prior, sigma_est[0], c_est[0], t_est[0], tau_est[0],
                                        w0_est[0], beta_est[0], u_est[0], a_t, b_t)]

    accept_params = [0]
    accept_hmc = 0
    rate = [0]
    rate_p = [0]
    step = 100
    nadapt = 1000

    for i in range(iter):

        # update hyperparameters if at least one of them demands the update
        if sigma is True or c is True or t is True or tau is True:
            output_params = up.update_params(prior, sigma_est[-1], c_est[-1], t_est[-1], tau_est[-1], z_est[-1],
                                             w0_est[-1], beta_est[-1], u_est[-1], log_post_est[-1], accept_params[-1],
                                             sigma=sigma, c=c, t=t, tau=tau,
                                             sigma_sigma=sigma_sigma,  sigma_c=sigma_c, sigma_t=sigma_t,
                                             sigma_tau=sigma_tau, a_t=a_t, b_t=b_t)
            sigma_est.append(output_params[0])
            c_est.append(output_params[1])
            t_est.append(output_params[2])
            tau_est.append(output_params[3])
            z_est.append(output_params[4])
            accept_params.append(output_params[5])
            log_post_est.append(output_params[6])
            rate_p.append(output_params[7])
            print('sigma = ', sigma_est[-1]) 
            if i % 1000 == 0:
                print('update hyperparams iteration = ', i)
                print('acceptance rate hyperparams = ', round(accept_params[-1] / (i+1) * 100, 1), '%')
                print('z = ', output_params[4])
                print('painful term = ', (z_est[-1] + c_est[-1]) ** sigma_est[-1] - c_est[-1] ** sigma_est[-1])
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
                output_gibbs = up.gibbs_w(w_est[-1], beta_est[-1], sigma_est[-1], c_est[-1], z_est[-1],
                                          u_est[-1], n_est[-1], p_ij, gamma)
                w_est.append(output_gibbs[0])
                w0_est.append(output_gibbs[1])
                beta_est.append(beta_est[0])  # beta is not updated in the gibbs version!
                if i % 1000 == 0 and i != 0:
                    print('update w iteration = ', i)
            if w_inference == 'HMC':
                output_hmc = up.HMC_w(prior, w_est[-1], w0_est[-1], beta_est[-1], n_est[-1], u_est[-1],
                                      sigma_est[-1], c_est[-1], t_est[-1], tau_est[-1], z_est[-1], gamma,
                                      p_ij, a_t, b_t, epsilon, R, accept_hmc, size, update_beta=beta)
                w_est.append(output_hmc[0])
                w0_est.append(output_hmc[1])
                beta_est.append(output_hmc[2])
                accept_hmc = output_hmc[3]
                rate.append(output_hmc[4])
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
        if n is True:
            n_est.append(up.update_n(w_est[-1], G, size, p_ij))
            if i % 1000 == 0:
                print('update n iteration = ', i)
        # update u
        if u is True:
            u_est.append(up.posterior_u(z_est[-1] * w0_est[-1]))
            if i % 1000 == 0:
                print('update u iteration = ', i)

    if plot is True:
        plot_MCMC(prior, iter, nburn, size, G,
                  w0=w0, beta=beta, n=n, u=u, sigma=sigma, c=c, t=t, tau=tau,
                  sigma_est=sigma_est, c_est=c_est, t_est=t_est, tau_est=tau_est, w_est=w_est, beta_est=beta_est,
                  n_est=n_est, u_est=u_est, log_post_est=log_post_est, **kwargs)

    return w_est, w0_est, beta_est, sigma_est, c_est, t_est, tau_est, n_est, u_est, log_post_est


def plot_MCMC(prior, iter, nburn, size, G,
              w0=False, beta=False, n=False, u=False, sigma=False, c=False, t=False, tau=False,
              **kwargs):

    if 'log_post_true' in kwargs:
        plt.figure()
        plt.plot(kwargs['log_post_est'])
        plt.axhline(y=kwargs['log_post_true'], color='r')
        plt.xlabel('iter')
        plt.ylabel('log_post')
        plt.savefig('images/all/all_logpost')
    if sigma is True:
        plt.figure()
        sigma_est = kwargs['sigma_est']
        plt.plot(sigma_est, color='blue')
        if 'sigma_true' in kwargs:
            plt.axhline(y=kwargs['sigma_true'], label='true', color='r')
        plt.xlabel('iter')
        plt.ylabel('sigma')
        plt.savefig('images/all/all_sigma')
    if c is True:
        plt.figure()
        c_est = kwargs['c_est']
        plt.plot(c_est, color='blue')
        if 'c_true' in kwargs:
            plt.axhline(y=kwargs['c_true'], color='r')
        plt.xlabel('iter')
        plt.ylabel('c')
        plt.savefig('images/all/all_c')
    if t is True:
        plt.figure()
        t_est = kwargs['t_est']
        plt.plot(t_est, color='blue')
        if 't_true' in kwargs:
            plt.axhline(y=kwargs['t_true'], color='r')
        plt.xlabel('iter')
        plt.ylabel('t')
        plt.savefig('images/all/all_t')
    if prior == 'doublepl' and tau is True:
        plt.figure()
        tau_est = kwargs['tau_est']
        plt.plot(tau_est, color='blue')
        if 'tau_true' in kwargs:
            plt.axhline(y=kwargs['tau_true'], color='r')
        plt.xlabel('iter')
        plt.ylabel('tau')
        plt.savefig('images/all/all_t')
    if w0 is True:
        plt.figure()
        w_est = kwargs['w_est']
        deg = np.array(list(dict(G.degree()).values()))
        biggest_deg = np.argsort(deg)[-1]
        biggest_w_est = [w_est[i][biggest_deg] for i in range(iter)]
        plt.plot(biggest_w_est)
        if 'w_true' in kwargs:
            w = kwargs['w_true']
            biggest_w = w[biggest_deg]
            plt.axhline(y=biggest_w, label='true')
        plt.xlabel('iter')
        plt.ylabel('highest degree w')
        plt.legend()
        plt.savefig('images/wannacry_trace2')
        if 'w_true' in kwargs:  # plot empirical 95% ci for highest and lowest degrees nodes
            plt.figure()
            w_est_fin = [w_est[i] for i in range(nburn, iter)]
            emp0_ci_95 = [
                scipy.stats.mstats.mquantiles([w_est_fin[i][j] for i in range(iter - nburn)], prob=[0.025, 0.975])
                for j in range(size)]
            true0_in_ci = [emp0_ci_95[i][0] <= w[i] <= emp0_ci_95[i][1] for i in range(size)]
            print('posterior coverage of true w = ', sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
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
            plt.legend()
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
            plt.legend()
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
            plt.legend()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
            plt.savefig('images/wannacry_CI2')
