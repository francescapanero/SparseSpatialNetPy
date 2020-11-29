import numpy as np
import matplotlib.pyplot as plt
import scipy

def sanity_check(prior, w, u, betaw, G, deg, p_ij, alpha_true, sigma_true, tau_true):

    size = len(w)

    if prior == 'GGP' or prior == 'exptiltBFRY':
        sum_w = sum(w)
        sum_logw = sum(np.log(w))
        sum_u = sum(u)

        # sigma
        sigma = np.linspace(0.05, sigma_true+0.05, 20)
        b = [sum_u*np.log((size*sigma[i]/alpha_true)**(1/sigma[i])) - ((size*sigma[i]/alpha_true)**(1/sigma[i]))*sum_w
             - size*np.log((tau_true+(size*sigma[i]/alpha_true)**(1/sigma[i]))**sigma[i] - tau_true**sigma[i])
             + size*np.log(sigma[i]) + (-1-sigma[i])*sum_logs - size*np.log(scipy.special.gamma(1-sigma[i]))
             for i in range(len(sigma))]
        plt.plot(sigma, b)
        plt.axvline(x=sigma_true, label='true')
        plt.xlabel('sigma')
        plt.ylabel('sum log p(w,u|sigma)')
        b_ind = b.index(max(b))
        plt.axvline(x=sigma[b_ind],  color='r', label='max')
        plt.legend()
        plt.savefig('sigma')

        # tau
        tau = np.linspace(0.1, tau_true+5, 50)
        d = [- size*np.log((tau[i]+(size*sigma_true/alpha_true)**(1/sigma_true))**sigma_true - tau[i]**sigma_true)
             - tau[i]*sum_s for i in range(len(tau))]
        plt.plot(tau, d)
        plt.axvline(x=tau_true, label='true')
        plt.xlabel('tau')
        plt.ylabel('sum log p(w,u|tau)')
        d_ind = d.index(max(d))
        plt.axvline(x=tau[d_ind],  color='r', label='max')
        plt.legend()

        # alpha
        alpha = np.linspace(alpha_true-10, alpha_true+10, 101)
        a = [sum_u*np.log((size*sigma_true/alpha[i])**(1/sigma_true)) - ((size*sigma_true/alpha[i])**(1/sigma_true))*sum_w
             - size*np.log((tau_true+(size*sigma_true/alpha[i])**(1/sigma_true))**sigma_true - tau_true**sigma_true)
             for i in range(len(alpha))]
        plt.plot(alpha, a)
        plt.axvline(x=alpha_true, label='true')
        plt.xlabel('alpha')
        plt.ylabel('sum log p(w,u|alpha)')
        a_ind = a.index(max(a))
        plt.axvline(x=alpha[a_ind],  color='r', label='max')
        plt.legend()
        plt.savefig('alpha')


    if prior == 'doublepl':
        sum_w = sum(w)
        sum_logw = sum(np.log(w))
        sum_u = sum(u)
        sum_w0 = sum(w0)
        sum_logw0 = sum(np.log(w0))
        sum_logbeta = sum(np.log(betaw))

        # sigma
        sigma = np.linspace(sigma_true-0.1, sigma_true+0.3, 100)
        t = np.power(size*sigma*tau_true/(alpha_true*np.power(c,tau_true-sigma)), 1/sigma)
        b = [size*np.log(sigma[i]/(scipy.special.gamma(1-sigma[i])*((t[i]+c)**sigma[i]-c**sigma[i]))) +
             (-1-sigma[i])*sum_logw0 + np.log(t[i])*sum_u - t[i]*sum_w0 for i in range(len(sigma))]
        plt.plot(sigma, b)
        plt.axvline(x=sigma_true, label='true')
        plt.xlabel('sigma')
        plt.ylabel('sum log p(w,u|sigma)')
        b_ind = b.index(max(b))
        plt.axvline(x=sigma[b_ind],  color='r', label='max')
        plt.legend()
        plt.savefig('sigma')

        # tau
        tau = np.linspace(1.1, tau_true+2, 80)
        t = (size*sigma_true*tau/(alpha_true*np.power(c,tau-sigma_true)))**(1/sigma_true)
        d = [size*np.log(scipy.special.gamma(1+tau[i])/(scipy.special.gamma(tau[i])*((t[i]+c)**sigma_true)-c**sigma_true))
             +np.log(t[i])*sum_u - t[i]*sum_w0+(tau[i]-1)*sum_logbeta for i in range(len(tau))]
        plt.plot(tau, d)
        plt.axvline(x=tau_true, label='true')
        plt.xlabel('tau')
        plt.ylabel('sum log p(s_i,u_i|tau)')
        d_ind = d.index(max(d))
        plt.axvline(x=tau[d_ind],  color='r', label='max')
        plt.legend()

        # w_i, but I'm only doing it for the highest deg node
        ind_big = np.argmax(deg)
        big_w = w[ind_big]
        w_linspace = np.linspace(big_w-0.5, big_s+0.5, 100)
        w_sum = []
        for i in range(len(w_linspace)):
            w_sum.append(0)
            for j in range(size):
                w_sum[i] = w_linspace[i] + w[j]
        w_outer = []
        for i in range(len(w_linspace)):
            w_outer.append(np.outer(w_linspace[i], w))
        param = []
        for i in range(len(w_linspace)):
            param.append(2*p_ij[ind_big, :]*w_outer[i])
        sum_loglik = []
        for i in range(len(w_linspace)):
            sum_loglik.append(0)
            for j in range(size):
                edges_j = G.edges(j)
                if [ind_big, j] in edges_j: # CHECK THIS
                    sum_loglik[i] = sum_loglik[i] + np.log(1-np.exp(-param[i][0][j]))
                else:
                    sum_loglik[i] = sum_loglik[i] - param[i][0][j]
            sum_loglik[i] = sum_loglik[i] + (-1-sigma_true+u_true[ind_big])*np.log(w_linspace[i]) -\
                            (tau_true+t_true)*w_linspace[i]
        plt.plot(w_linspace, sum_loglik)
        plt.axvline(x=big_w, label='true')
        plt.xlabel('w')
        plt.ylabel('sum log p(w|rest)')
        d_ind = sum_loglik.index(max(sum_loglik))
        plt.axvline(x=w_linspace[d_ind],  color='r', label='max')
        plt.legend()
        plt.savefig('w_max_deg')


    # STILL TO CHECK
    degsort = np.sort(deg)

    ind_big = np.where(deg == degsort[size-1])
    big_w = w[ind_big]
    w_linspace = np.linspace(max(0.1,big_w-5), big_w+5, 200)
    # m_i = np.sum(N_true, axis=0)[ind_big] + np.sum(N_true, axis=1)[ind_big]
    # n_i = N_true[ind_big,ind_big]
    # p_i = p_ij[i,]
    # pw = np.dot(p_i, s_true)
    betawi = betaw[ind_big]
    # w0i = w0[ind_big]
    # ui = u_true[ind_big]
    # loglik = np.zeros(len(s_linspace))
    # for i in range(len(s_linspace)):
    #     s_true_modif = np.concatenate((s_true[1:ind_big], [s_linspace[i]], s_true[ind_big+1:len(s_true)]))
    #     pwmodif = pw - s_true[ind_big] + s_linspace[i]
    #     loglik[i] = 2*np.log(s_linspace[i]*betawi)*(m_i-n_i) - s_linspace[i]*(2*pwmodif-s_linspace[i]) + \
    #     (ui-1-sigma_true)*np.log(s_linspace[i]*betawi) - (c+t_true)*s_linspace[i]*betawi

    s_sum = []
    for i in range(len(s_linspace)):
        s_sum.append(0)
        for j in range(size):
            s_sum[i] = s_linspace[i] + w[j]
    log_s = np.log(w)
    s_outer = []
    for i in range(len(s_linspace)):
        s_outer.append(np.outer(s_linspace[i], w))
    param = []
    for i in range(len(s_linspace)):
        param.append(2*p_ij[ind_big, :]*s_outer[i])
    sum_loglik = []
    for i in range(len(s_linspace)):
        sum_loglik.append(0)
        for j in range(size):
            if Z[ind_big, j]>0:
                sum_loglik[i] = sum_loglik[i] + np.log(1-np.exp(-param[i][0][0][j]))
            else:
                sum_loglik[i] = sum_loglik[i] - param[i][0][0][j]
        sum_loglik[i] = sum_loglik[i] + (-1-sigma_true+u_true[ind_big])*np.log(s_linspace[i]*betawi) - (c+t_true)*s_linspace[i]*betawi




    plt.plot(s_linspace, sum_loglik)
    plt.axvline(x=big_s, label='true')
    plt.xlabel('w')
    plt.ylabel('sum log p(w|rest)')
    d_ind = np.where(sum_loglik==max(sum_loglik))
    plt.axvline(x=s_linspace[d_ind],  color='r', label='max')
    plt.legend()
