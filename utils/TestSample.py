from numpy import savetxt
from utils.RepeatedSamples import *
from utils.GraphSamplerNew import *
from utils.TruncPois import *
import utils.AuxiliaryNew as aux

# --------------------------
# function to test the model. Up to now, I have implemented the following 'type' of tests:
# - 'layers_vs_naive': compare main outputs of two graphs simulated with layers and naive methods,
#                    having fixed hyperparameters and w, x
# - 'sampler': plots and prints main outputs of a sample from layers or naive method
# - 'clustering_coefficient': compares multiple samples (need to specify number n) from layers and naive methods
#                             against the asymptotic limit
# - 'sparsity': compares multiple samples' (need to specify number n) number of nodes and edges from layers and
#               naive methods
# --------------------------

# inputs:
# - type: described above
# - prior: GGP, exptiltBFRY or doublepl
# - sigma, tau, t: usual params of the models
# - size_x: bound for the space of locations
# - gamma: exponent tuning effect of distance in the probability of connections: Delta_x^-gamma
# - T: threshold for finite simulation of weights for infinite activity CRMs (w>T)
# - typesampler: to be specified if in 'sampler' mode
# - **kwargs: K (size of the grid for layers method), c (param of doublepl), n (number of repeated samples to take in
#             'clustering_coefficient' or 'sparsity' types


def unit_test(type, prior, approximation, sigma, c, t, tau, gamma, size_x, sampler='na', n=0, T=0.00001, K=100,
              L=10000, **kwargs):

    if type == 'layers_vs_naive':

        nodes_L = []
        edges_L = []
        nodes_N = []
        edges_N = []

        for i in range(n):

            # layers method
            start_l = time.time()
            w, w0, beta, x, G_l, size, deg_l = GraphSampler(prior, approximation, 'layers', sigma, c, t, tau,
                                                                      gamma, size_x, T=T, K=K, L=L)

            end_l = time.time()
            if 'save_vars' in kwargs:
                savetxt('w.csv', w, delimiter=',')
                savetxt('x.csv', x, delimiter=',')

            # plot check: see if the loglik is maximised in sigma, tau and t (these do not depend on w!)
            if prior == 'GGP' or prior == 'exptiltBFRY':
                z = (sigma * size / t) ** (1 / sigma)
                u = tpoissrnd(z * w)
            if prior == 'doublepl':
                t_dpl = t * (c ** (sigma * tau - sigma)) / (tau * sigma)
                z = (sigma * size / t_dpl) ** (1 / sigma)
                u = tpoissrnd(z * w0)
            # check_sample_loglik(prior, sigma, tau, t, size_x, gamma, w, u, t, **kwargs)

            # remove self loop and isolated vertices
            G_l, isol_l = aux.SimpleGraph(G_l)
            x_l_red = np.delete(x, isol_l)
            w_l_red = np.delete(w, isol_l)
            # number nodes and edges
            nodes_l = nx.number_of_nodes(G_l)
            edges_l = nx.number_of_edges(G_l)

            # naive method
            start_n = time.time()
            w, w0, beta, x, G_n, size, deg_n = GraphSampler(prior, approximation, 'layers', sigma, c, t, tau, gamma,
                                                         size_x, T=T, K=K, L=L, w=w, w0=w0, beta=beta, x=x)
            end_n = time.time()

            # remove self loop and isolated vertices
            G_n, isol_n = aux.SimpleGraph(G_n)
            x_n_red = np.delete(x, isol_n)
            w_n_red = np.delete(w, isol_n)
            # number of nodes and edges
            nodes_n = nx.number_of_nodes(G_n)
            edges_n = nx.number_of_edges(G_n)

            nodes_L.append(nodes_l)
            edges_L.append(edges_l)
            nodes_N.append(nodes_n)
            edges_N.append(edges_n)

        print('mean nodes layers ', np.mean(nodes_L))
        print('mean nodes naive ', np.mean(nodes_N))
        print('mean edges layers ', np.mean(edges_L))
        print('mean edges naive ', np.mean(edges_N))

        print('var nodes layers ', np.var(nodes_L))
        print('var nodes naive ', np.var(nodes_N))
        print('var edges layers ', np.var(edges_L))
        print('var edges naive ', np.var(edges_N))

        # plot adjacency matrices
        plt_space_adj(G_l, x_l_red)
        plt.title('Adjacency matrix layers')
        plt_space_adj(G_n, x_n_red)
        plt.title('Adjacency matrix naive')

        # plot the two degree distribution
        plt_deg_distr(deg_l, prior=prior, sigma=sigma, binned=False)
        plt.title('Degree distribution layers')
        plt_deg_distr(deg_n, prior=prior, sigma=sigma, binned=False)
        plt.title('Degree distribution naive')

        # plot ccdf of degrees
        plt_ccdf(deg_l, sigma=sigma, tau=tau, prior=prior)
        plt.title('Degree ccdf layers')
        plt_ccdf(deg_n, sigma=sigma, tau=tau, prior=prior)
        plt.title('Degree ccdf naive')

        # plot ranked degrees
        plt_rank(deg_l)
        plt.title('Ranked degrees layers')
        plt_rank(deg_n)
        plt.title('Ranked degrees naive')

    if type == 'sampler':
        start = time.time()
        w, w0, gamma, x, G, size, deg = GraphSampler(prior, approximation, sampler, sigma, c, t, tau, gamma, size_x,
                                                     T=T, K=K, L=L)
        end = time.time()
        if 'save_vars' in kwargs:
            savetxt('w.csv', w, delimiter=',')
            savetxt('x.csv', x, delimiter=',')
        # remove self loop and isolated vertices
        G, isol = aux.SimpleGraph(G)
        x_red = np.delete(x, isol)
        w_red = np.delete(w, isol)
        # number nodes and edges
        nodes = nx.number_of_nodes(G)
        edges = nx.number_of_edges(G)
        # degrees
        deg = np.array(list(dict(G.degree()).values()))

        print('minutes to produce the sample: ', round((end - start) / 60, 2))
        print('number of nodes: ', nodes)
        print('number of edges: ', edges)

        # plot adjacency matrices
        plt_space_adj(G, x_red)
        plt.title('Adjacency matrix')

        # plot the two degree distribution
        plt_deg_distr(deg, prior=prior, sigma=sigma, binned=False)
        plt.title('Degree distribution')

        # plot ccdf of degrees
        plt_ccdf(deg, sigma=sigma, tau=tau, prior=prior)
        plt.title('Degree ccdf')

        # plot ranked degrees
        plt_rank(deg)
        plt.title('Ranked degrees')

        if prior == 'doublepl':
            repeated_samples('large_deg_nodes', prior, sampler, t, sigma, tau, gamma, size_x, plot=True, **kwargs)

    if type == 'clustering_coefficient':
        [glob_l, loc_l, glob_lim, loc_lim] = repeated_samples('clustering', prior, approximation, 'layers', t, sigma, c,
                                                              tau, gamma, size_x, T=T, K=K, L=L, plot=True)
        plt.title('Clustering layers')
        [glob_n, loc_n, glob_lim, loc_lim] = repeated_samples('clustering', prior, approximation, 'naive', t, sigma, c,
                                                              tau, gamma, size_x, T=T, K=K, L=L, plot=True)
        plt.title('Clustering naive')
        # plot them together
        plt_compare_clustering(glob_n, loc_n, glob_l, loc_l, glob_lim, loc_lim, t)
        plt.title('Clustering comparison')
        return glob_l, loc_l, glob_n, loc_n, glob_lim, loc_lim

    if type == 'sparsity':
        [nodes_n, edges_n] = repeated_samples('sparsity', prior, approximation, 'naive', t, sigma, c,
                                              tau, gamma, size_x, T=T, K=K, L=L, plot=True)
        [nodes_l, edges_l] = repeated_samples('sparsity', prior, approximation, 'layers', t, sigma, c,
                                              tau, gamma, size_x, T=T, K=K, L=L, plot=True)
        # plot them together
        plt_compare_sparsity(nodes_n, edges_n, nodes_l, edges_l, t)







