import matplotlib.pyplot as plt
import os
import numpy as np
import scipy


def plot(out, G, path,
         sigma, c, tau, t, w0, x,
         iter, nburn, save_every):

    nchain = len(out)

    for i in range(nchain):
        plt.figure()
        if w0 is False and x is False:
            plt.plot(out[i][9], label='chain %i' % i)
            if 'log_post_param' in G[i].graph:
                plt.axhline(y=G[i].graph['log_post_param'], label='true', color='r')
        else:
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
            plt.plot([i for i in range(0, iter + save_every, save_every)], out[i][3], label='chain %i' % i)
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
            plt.plot([i for i in range(0, iter + save_every, save_every)], out[i][4], label='chain %i' % i)
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
            plt.plot([i for i in range(0, iter + save_every, save_every)], out[i][5], label='chain %i' % i)
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
            plt.plot([i for i in range(0, iter + save_every, save_every)], out[i][6], label='chain %i' % i)
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
            biggest_w_est = [w_est[i][biggest_deg] for i in range(int((iter + save_every) / save_every))]
            plt.plot([j for j in range(0, iter + save_every, save_every)], biggest_w_est)
            if 'w' in G[i].nodes[0]:
                w = np.array([G[i].nodes[j]['w'] for j in range(size)])
                biggest_w = w[biggest_deg]
                plt.axhline(y=biggest_w, label='true')
            plt.xlabel('iter')
            plt.ylabel('highest degree w')
            plt.legend()
            plt.savefig(os.path.join('images', path, 'w_trace_chain%i' % i))
            plt.close()

            w_est_fin = [w_est[k] for k in range(int((nburn + save_every) / save_every),
                                                 int((iter + save_every) / save_every))]
            emp0_ci_95 = [
                scipy.stats.mstats.mquantiles([w_est_fin[k][j] for k in range(int((iter + save_every) / save_every) -
                                                                              int((nburn + save_every) / save_every))],
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
                if j < 10:
                    print(emp_ci_big[-1])
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

    # if x is True:
    #     for i in range(nchain):
    #         num = 10
    #         deg = np.array(list(dict(G[i].degree()).values()))
    #         size = len(deg)
    #         sort_ind = np.argsort(deg)
    #         ind_big1 = sort_ind[range(size - num, size)]
    #         x_est = out[i][12]
    #         if 'x' in G[i].nodes[0]:
    #             x = np.array([G[i].nodes[j]['x'] for j in range(size)])
    #         for j in ind_big1:
    #             plt.figure()
    #             plt.plot([k for k in range(0, iter + save_every, save_every)],
    #                      [x_est[k][j] for k in range(int((iter + save_every) / save_every))])
    #             if 'x' in G[i].nodes[0]:
    #                 plt.axhline(y=x[j], label='true')
    #             plt.xlabel('iter')
    #             plt.ylabel('x')
    #             plt.savefig(os.path.join('images', path, 'x_highdeg%i_chain%i' % (j, i)))
    #             plt.close()
    #         plt.figure()
    #         for j in ind_big1:
    #             plt.plot([k for k in range(0, iter + save_every, save_every)],
    #                      [x_est[k][j] for k in range(int((iter + save_every) / save_every))],
    #                      label='%i' % j)
    #             plt.xlabel('iter')
    #             plt.ylabel('x')
    #             plt.legend()
    #             plt.savefig(os.path.join('images', path, 'x_highdeg_chain%i' % i))
    #         plt.close()
    #
    #         # need to take into account step
    #         x_est_fin = [x_est[k] for k in range(int((nburn + save_every) / save_every),
    #                                              int((iter + save_every) / save_every))]
    #         emp0_ci_95 = [
    #             scipy.stats.mstats.mquantiles(
    #                 [x_est_fin[k][j] for k in range(int((iter + save_every) / save_every) -
    #                                                 int((nburn + save_every) / save_every))],
    #                 prob=[0.025, 0.975]) for j in range(size)]
    #         if 'x' in G[i].nodes[0]:
    #             true0_in_ci = [emp0_ci_95[j][0] <= x[j] <= emp0_ci_95[j][1] for j in range(size)]
    #             print('posterior coverage of true x (chain %i' % i, ') = ',
    #                   sum(true0_in_ci) / len(true0_in_ci) * 100, '%')
    #
    #         p_ij_est = out[i][11]
    #         p_ij_est_fin = [[p_ij_est[k][j, :] for k in range(int((nburn + save_every) / save_every),
    #                                              int((iter+save_every)/save_every))] for j in range(size)]
    #         emp_ci_95_big = []
    #         for j in range(size):
    #             emp_ci_95_big.append(
    #                 [scipy.stats.mstats.mquantiles(
    #                     [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
    #                                                           int((nburn + save_every) / save_every))],
    #                     prob=[0.025, 0.975]) for l in range(size)])
    #         if 'distances' in G[i].graph:
    #             p_ij = G[i].graph['distances']
    #             true_in_ci = [[emp_ci_95_big[j][k][0] <= p_ij[[j], k] <= emp_ci_95_big[j][k][1]
    #                           for k in range(size)] for j in range(size)]
    #             print('posterior coverage of true p_ij (chain %i' % i, ') = ',
    #                   np.mean([sum(true_in_ci[j]) / size * 100 for j in range(size)]), '%')


# debugging plots for locations x.
# Specify the name of the folder (you'll find it in 'images') in which they're saved.
# Outputs:
# - traceplot of log posterior
# - traceplots of x[index]
# - p_ij 95% posterior c.i. wrt true valued for some of x[index] (max 20 of them)
def plot_space_debug(out, G, iter, nburn, save_every, index, path):

    # os.mkdir(os.path.join('images', path))
    nchain = len(out)

    for l in range(nchain):

        x = np.array([G[l].nodes[i]['x'] for i in range(G[l].number_of_nodes())])
        deg = np.array(list(dict(G[l].degree()).values()))
        p_ij = G[l].graph['distances']

        # plot some location traceplots

        x_est = out[l][12]
        if np.isscalar(index):
            plt.figure()
            plt.plot([x_est[i][index] for i in range(int(iter / save_every))])
            plt.axhline(y=x[index])
            plt.title('Traceplot location of node with deg %i' % deg[index])
            plt.savefig(os.path.join('images', path, 'trace_deg%i_chain%i' % (deg[index], l)))
            plt.close()
        else:
            if len(index) < 20:
                for j in index:
                    plt.figure()
                    plt.plot([x_est[i][j] for i in range(int(iter / save_every))])
                    plt.axhline(y=x[j])
                    plt.title('Traceplot location of node with deg %i' % deg[j])
                    plt.savefig(os.path.join('images', path, 'trace_deg%i_chain%i' % (deg[j], l)))
                    plt.close()
            if len(index) > 19:
                # plot for 10 lowest and 10 highest deg nodes
                for j in np.concatenate((index[0:10], index[(len(index)-10):len(index)])):
                    plt.figure()
                    plt.plot([x_est[i][j] for i in range(int(iter / save_every))])
                    plt.axhline(y=x[j])
                    plt.title('Traceplot location of node with deg %i' % deg[j])
                    plt.savefig(os.path.join('images', path, 'trace_deg%i_chain%i' % (deg[j], l)))
                    plt.close()

        # plot p_ij and print posterior coverage for 10 lowest and 10 highest deg nodes
        if not np.isscalar(index):

            if len(index) > 19:
                index = np.concatenate((index[0:10], index[len(index)-10: len(index)]))

            size = G[l].number_of_nodes()
            p_ij_est = out[l][11]
            p_ij_est_fin = [[p_ij_est[k][h, :] for k in range(int((nburn + save_every) / save_every),
                                                              int((iter + save_every) / save_every))] for h in index]
            emp_ci = []
            true_in_ci = []

            for j in range(len(index)):
                # compute posterior coverage of these nodes
                emp_ci.append(
                    [scipy.stats.mstats.mquantiles(
                        [p_ij_est_fin[j][k][l] for k in range(int((iter + save_every) / save_every) -
                                                              int((nburn + save_every) / save_every))],
                        prob=[0.025, 0.975]) for l in range(size)])
                true_in_ci.append([emp_ci[j][k][0] <= p_ij[index[j], k] <= emp_ci[j][k][1]
                              for k in range(size)])
                # plot p_ij across these nodes
                plt.figure()
                for k in range(len(index)):
                    plt.plot((k + 1, k + 1), (emp_ci[j][index[k]][0], emp_ci[j][index[k]][1]),
                         color='cornflowerblue', linestyle='-', linewidth=2)
                    plt.plot(k + 1, p_ij[index[j], index[k]], color='navy', marker='o', markersize=5)
                plt.savefig(os.path.join('images', path, 'pij_deg%i_chain%i' % (deg[index[j]], l)))
                plt.close()
                print('posterior coverage in chain %i' % l, ' of true p_ij for node with deg %i' % deg[index[j]], ' = ',
                        round(sum(true_in_ci[j]) / size * 100, 1), '%')