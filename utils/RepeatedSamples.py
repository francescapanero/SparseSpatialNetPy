from utils.Limits import *
from utils.PlotGraph import *
from utils.GraphSamplerNew import *

# this function contains various 'mode's that require to take multiple samples (with varying t)
# for asymptotic studies, or others where t but very big
# - 'sparsity': number of nodes and edges for increasing t (which needs to be an array).
#               Other than usual params, you need to specify n= the number of samples you want to take
# - 'clustering': clustering coefficients for increasing t (which needs to be an array).
#                 Other than usual params, you need to specify n= the number of samples you want to take
# - 'large_deg': large degrees for big t, just for doublepl method


def repeated_samples(mode, prior, approximation, sampler, t, sigma, c, tau, gamma, size_x, T=0.001, K=100, L=5000,
                     n=3, plot=False):
    
    if mode == 'sparsity':
        nodes = np.zeros((n, len(t)))
        edges = np.zeros((n, len(t)))

        for i in range(len(t)):
            for j in range(n):
                w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma,
                                                            size_x, T=T, K=K, L=L)
                isol = list(nx.isolates(G))
                G.remove_nodes_from(isol)
                nodes[j, i] = len(G.nodes())
                edges[j, i] = len(G.edges())
                print('sample number ', j)
            print('t = ', int(t[i]))

        if plot is True:
            plt.figure()
            for i in range(n - 1):
                plt.plot(t, np.log(nodes[i, ]), 'bo-')
            plt.plot(t, np.log(nodes[n - 1, ]), 'bo-')
            plt.xlabel('t')
            plt.ylabel('number of nodes')
            plt.figure()
            for i in range(n - 1):
                plt.plot(t, np.log(edges[i, ]), 'bo-')
            plt.plot(t, np.log(edges[n - 1, ]), 'bo-')
            plt.xlabel('t')
            plt.ylabel('number of edges')

        return nodes, edges

    if mode == 'clustering':

        avg_loc_clust = np.zeros((n, len(t)))
        glob_clust = np.zeros((n, len(t)))
        tmax = max(t)

        for j in range(n):
            w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, tmax, tau, gamma,
                                                        size_x, T=T, K=K, L=L)
            isol = list(nx.isolates(G))
            w = np.delete(w, isol)
            thetamax = tmax * np.random.rand(len(w))
            G.remove_nodes_from(isol)
            G.remove_edges_from(nx.selfloop_edges(G))
            Gadj = nx.to_numpy_matrix(G)
            G1 = np.square(Gadj)
            deg = np.squeeze(np.asarray(np.matrix.sum(G1, 1)))
            cn = np.diag(G1 * np.triu(G1) * G1).copy()
            # cn = np.diag(G1 * G1 * G1)
            c0 = np.zeros(len(deg))
            a = np.multiply(deg, (deg - 1))
            for h in range(len(a)):
                if a[h] != 0:
                    c0[h] = 2 * cn[h] / a[h]
            glob_clust[j, len(t) - 1] = sum(cn) / sum(deg[deg > 1] * (deg[deg > 1] - 1) / 2)
            avg_loc_clust[j, len(t)-1] = np.mean(c0[deg > 1])
            print('t = %i' % tmax)
            for i in reversed(range(len(t)-1)):
                ind = np.where(thetamax < t[i])
                G2 = Gadj[ind[0], :]
                G2 = G2[:, ind[0]]
                G2 = G2 - np.diag(G2)
                G1 = np.square(G2)
                deg = np.squeeze(np.asarray(np.matrix.sum(G1, 1)))
                cn = np.diag(G1 * np.triu(G1) * G1).copy()
                # cn = np.diag(G1 * G1 * G1)
                c0 = np.zeros(len(deg))
                a = np.multiply(deg, (deg - 1))
                for h in range(len(a)):
                    if a[h] != 0:
                        c0[h] = 2 * cn[h] / a[h]
                glob_clust[j, i] = sum(cn) / sum(deg[deg > 1] * (deg[deg > 1] - 1) / 2)
                avg_loc_clust[j, i] = np.mean(c0[deg > 1])
                print('t = %i' % t[i])
            print('sample number %i' % j)

            [limit_glob, limit_loc] = limit_clustering(prior, sigma, c, gamma, size_x, tau)

        if plot is True:
            plt.figure()
            for i in range(n - 1):
                plt.plot(t, avg_loc_clust[i, ], 'b-', t, glob_clust[i, ], 'r-', linewidth=0.6)
            plt.plot(t, avg_loc_clust[n - 1, ], 'b-', label='avg local', linewidth=0.6)
            plt.plot(t, glob_clust[n - 1, ], 'r-', label='global', linewidth=0.6)
            plt.hlines(limit_glob, min(t), max(t), color='r', linestyles='dashed', label='limit glob')
            plt.hlines(limit_loc, min(t), max(t), color='b', linestyles='dashed', label='limit avg loc')
            plt.xlabel('t')
            plt.ylabel('clustering coefficient')
            plt.legend()

        return glob_clust, avg_loc_clust, limit_glob, limit_loc

    # if mode == 'clustering_space':
    #
    #     avg_loc_clust = []
    #     glob_clust = []
    #     n = kwargs['n']
    #
    #     for j in range(n):
    #
    #         temp_avgloc = np.zeros(len(size_x))
    #         temp_glob = np.zeros(len(size_x))
    #
    #         for i in range(len(size_x)):
    #
    #             if typeweights == 'doublepl':
    #                 [w, x, G, size, betaw, w0] = GraphSampler(typeweights, typesampler, t, sigma, c, beta,
    #                                                           size_x[i], **kwargs)
    #             if typeweights == 'GGP' or typeweights == 'exptiltBFRY':
    #                 [w, x, G, size] = GraphSampler(typeweights, typesampler, t, sigma, c, beta, size_x[i],
    #                                                **kwargs)
    #
    #             isol = list(nx.isolates(G))
    #             G.remove_nodes_from(isol)
    #             G.remove_edges_from(nx.selfloop_edges(G))
    #             Gadj = nx.to_numpy_matrix(G)
    #             G1 = np.square(Gadj)
    #             deg = np.squeeze(np.asarray(np.matrix.sum(G1, 1)))
    #             cn = np.diag(G1 * np.triu(G1) * G1).copy()
    #             c = np.zeros(len(deg))
    #             a = np.multiply(deg, (deg - 1))
    #             for h in range(len(a)):
    #                 if a[h] != 0:
    #                     c[h] = 2 * cn[h] / a[h]
    #             temp_glob[i] = sum(cn) / sum(deg[deg > 1] * (deg[deg > 1] - 1) / 2)
    #             temp_avgloc[i] = np.mean(c[deg > 1])
    #
    #             print('size_x: %i' % size_x[i])
    #
    #         print('sample %i' % j)
    #
    #         avg_loc_clust.append(temp_avgloc)
    #         glob_clust.append(temp_glob)
    #
    #     return glob_clust, avg_loc_clust

    if mode == 'large_deg':

        if prior == 'doublepl':
            w, w0, beta, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma,
                                                        size_x, T=T, K=K, L=L)
        if prior == 'singlepl':
            print('this holds only for the doublepl!')
            return

        deg = list(dict(G.degree()).values())
        deg_ = pandas.Series(deg)
        freq = deg_.value_counts()
        freq = dict(freq)
        j = list(freq.keys())
        nj = list(freq.values())
        freq_keys = np.arange(max(j))
        # compute constants of \bar{\rho}
        c_1 = beta**(sigma*(c-1))/(sigma**2*(c-1)*scipy.special.gamma(1-sigma))
        c_0 = scipy.special.gamma(sigma*(c-1))/(sigma*c*scipy.special.gamma(1-sigma))
        # compute the asymptotic value of the number of nodes with degree j. The approximation holds for j\to\infty
        exp_nj = [c * t ** (c + 1) * c_0 / ((2 ** (sigma * c)) * (c_1 ** c) * (scipy.special.gamma(1 - sigma) ** c) \
                                            * i ** (1 + c)) for i in j]

        if plot is True:
            plt_large_deg_nodes(j, nj, exp_nj)

        return j, nj, exp_nj




