import time
from numpy import savetxt
from utils.SimpleGraph import *
from utils.RepeatedSamples import *
from utils.GraphSampler import *
from utils.CheckLikelParams import *
from utils.TruncPois import *

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
# - sigma, tau, alpha: usual params of the models
# - size_x: bound for the space of locations
# - beta: exponent tuning effect of distance in the probability of connections: Delta_x^-beta
# - T: threshold for finite simulation of weights for infinite activity CRMs (w>T)
# - typesampler: to be specified if in 'sampler' mode
# - **kwargs: K (size of the grid for layers method), c (param of doublepl), n (number of repeated samples to take in
#             'clustering_coefficient' or 'sparsity' types


def unit_test(type, prior, sigma, tau, alpha, beta, size_x, T, typesampler='na', n=0, **kwargs):

    if type == 'layers_vs_naive':

        nodes_L = []
        edges_L = []
        nodes_N = []
        edges_N = []

        for i in range(n):

            K = kwargs['K']
            # layers method
            start_l = time.time()
            output = GraphSampler(prior, "layers", alpha, sigma, tau, beta, size_x, T=T, **kwargs)
            end_l = time.time()
            w, x, G_l, size, deg_l = output[0:5]
            if prior == 'doublepl':
                betaw, w0 = output[5:7]
            if 'save_vars' in kwargs:
                savetxt('w.csv', w, delimiter=',')
                savetxt('x.csv', x, delimiter=',')

            # plot check: see if the loglik is maximised in sigma, tau and alpha (these do not depend on w!)
            if prior == 'GGP' or prior == 'exptiltBFRY':
                t = (sigma*size/alpha)**(1/sigma)
                u = tpoissrnd(t*w)
            if prior == 'doublepl':
                c = kwargs['c']
                alpha_dpl = alpha*(c**(sigma*tau-sigma))/(tau*sigma)
                t = (sigma*size/alpha_dpl)**(1/sigma)
                u = tpoissrnd(t*w0)
            # check_sample_loglik(prior, sigma, tau, alpha, size_x, beta, w, u, t, **kwargs)

            # remove self loop and isolated vertices
            G_l, isol_l = SimpleGraph(G_l)
            x_l_red = np.delete(x, isol_l)
            w_l_red = np.delete(w, isol_l)
            # number nodes and edges
            nodes_l = nx.number_of_nodes(G_l)
            edges_l = nx.number_of_edges(G_l)
            # degrees
            deg_l = np.array(list(dict(G_l.degree()).values()))

            # naive method
            start_n = time.time()
            if prior == 'doublepl':
                output = GraphSampler(prior, "naive", alpha, sigma, tau, beta, size_x, T=T, w=w, x=x, w0=w0,
                                      betaw=betaw, **kwargs)
            if prior == 'GGP' or prior == 'exptiltBFRY':
                output = GraphSampler(prior, "naive", alpha, sigma, tau, beta, size_x, T=T, w=w, x=x)
            end_n = time.time()
            w, x, G_n, size, deg_n = output[0:5]
            if prior == 'doublepl':
                betaw_n, w0_n = output[5:7]

            # remove self loop and isolated vertices
            G_n, isol_n = SimpleGraph(G_n)
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
        output = GraphSampler(prior, typesampler, alpha, sigma, tau, beta, size_x, T=T, **kwargs)
        end = time.time()
        w, x, G, size = output[0:4]
        if prior == 'doublepl':
            betaw, w0 = output[4:6]
        if 'save_vars' in kwargs:
            savetxt('w.csv', w, delimiter=',')
            savetxt('x.csv', x, delimiter=',')
        # remove self loop and isolated vertices
        G, isol = SimpleGraph(G)
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
            repeated_samples('large_deg_nodes', prior, typesampler, alpha, sigma, tau, beta, size_x, plot=True, **kwargs)

    if type == 'clustering_coefficient':
        [glob_l, loc_l, glob_lim, loc_lim] = repeated_samples('clustering', prior, 'layers', alpha, sigma, tau, beta,
                                                          size_x, T=T, plot=True, **kwargs)
        plt.title('Clustering layers')
        [glob_n, loc_n, glob_lim, loc_lim] = repeated_samples('clustering', prior, 'naive', alpha, sigma, tau, beta,
                                                          size_x, T=T, plot=True, **kwargs)
        plt.title('Clustering naive')
        # plot them together
        plt_compare_clustering(glob_n, loc_n, glob_l, loc_l, glob_lim, loc_lim, alpha)
        plt.title('Clustering comparison')
        return glob_l, loc_l, glob_n, loc_n, glob_lim, loc_lim

    if type == 'sparsity':
        [nodes_n, edges_n] = repeated_samples('sparsity', prior, 'naive', alpha, sigma, tau, beta, size_x, T=T,
                                              plot=False, **kwargs)
        [nodes_l, edges_l] = repeated_samples('sparsity', prior, 'layers', alpha, sigma, tau, beta, size_x, T=T,
                                              plot=False, **kwargs)
        # plot them together
        plt_compare_sparsity(nodes_n, edges_n, nodes_l, edges_l, alpha)







