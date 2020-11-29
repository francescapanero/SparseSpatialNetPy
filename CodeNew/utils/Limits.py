import scipy
from utils.LocationsSampler import *

# --------------------------
# computes the asymptotic limits of clustering coefficients (global and local) for different priors (GGP and doublepl)
# the doublepl looks wrong :|
# the integrals are computed with monte carlo.
# --------------------------


def limit_clustering(prior, sigma, tau, beta, size_x, **kwargs):

    n = 1000000
    loc_1 = LocationsSampler(size_x, n)
    loc_2 = LocationsSampler(size_x, n)
    loc_3 = LocationsSampler(size_x, n)

    if prior == 'singlepl':

        w_1 = np.random.gamma(1 - sigma, 1 / tau, n)
        w_2 = np.random.gamma(1 - sigma, 1 / tau, n)
        w_3 = np.random.gamma(1 - sigma, 1 / tau, n)

        num = 1 / n * np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
                              (1 - np.exp(-2 * w_2 * w_3 / ((1 + np.absolute(loc_2 - loc_3)) ** beta))) * \
                              (1 - np.exp(-2 * w_1 * w_3 / ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * \
                              (w_1 * w_2 * w_3)**(-1) * tau ** (3*sigma-3) * size_x**3)
        den = 1 / n * np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
                             (1 - np.exp(-2 * w_1 * w_3 / ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * \
                             (w_1 * w_2 * w_3) ** (-1) * tau ** (3 * sigma - 3) * size_x ** 3)
        num_l = 1 / n * np.sum((1 - np.exp(-2 * w_2 * w_3 / ((1 + np.absolute(loc_2 - loc_3)) ** beta))) * \
                               w_2 * w_3 / (((1 + np.absolute(loc_1 - loc_2)) ** beta) * (
                    ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * \
                               (w_2 * w_3) ** (-1) * tau ** (2 * sigma - 2) * size_x ** 3)
        den_l = 1 / n * np.sum(
            w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta) * w_2 ** (-1) * tau ** (sigma - 1) * size_x ** 2)

    if prior == 'doublepl':

        w_1 = np.random.gamma(1 - sigma, 1 / (sigma*tau), n)
        w_2 = np.random.gamma(1 - sigma, 1 / (sigma*tau), n)
        w_3 = np.random.gamma(1 - sigma, 1 / (sigma*tau), n)
        c = kwargs['c']

        num = 1 / n * np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
                             (1 - np.exp(-2 * w_2 * w_3 / ((1 + np.absolute(loc_2 - loc_3)) ** beta))) * \
                             (1 - np.exp(-2 * w_1 * w_3 / ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * \
                             (w_1 * w_2 * w_3)**(- 1 - sigma * tau + sigma) * (sigma * tau) ** (3 * sigma - 3) * size_x ** 3 * \
                             np.exp(sigma * tau * (w_1 + w_2 + w_3)) * \
                             scipy.special.gammainc(sigma * (tau - 1), c * w_1) * scipy.special.gammainc(sigma * (tau- 1 ), c * w_2) * \
                             scipy.special.gammainc(sigma * (tau - 1), c * w_3) * scipy.special.gamma(sigma * (tau - 1)) ** 3)
        den = 1 / n * np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
                             (1 - np.exp(-2 * w_1 * w_3 / ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * \
                             (w_1 * w_2 * w_3)**(- 1 - sigma * tau + sigma) * (sigma * tau) ** (3 * sigma - 3) * size_x ** 3 * \
                             np.exp(sigma * tau * (w_1 + w_2 + w_3)) * \
                             scipy.special.gammainc(sigma * (tau - 1), c * w_1) * scipy.special.gammainc(sigma * (tau - 1), c * w_2) * \
                             scipy.special.gammainc(sigma * (tau - 1), c * w_3) * scipy.special.gamma(sigma * (tau - 1)) ** 3)
        num_l = 1 / n * np.sum((1 - np.exp(-2 * w_2 * w_3 / ((1 + np.absolute(loc_2 - loc_3)) ** beta))) * \
                               (w_2 * w_3) ** (sigma - sigma*tau) / (((1 + np.absolute(loc_1 - loc_2)) ** beta) * (
                               ((1 + np.absolute(loc_1 - loc_3)) ** beta))) * np.exp(sigma * tau * (w_2 + w_3)) * \
                               (sigma * tau) ** (2 * sigma - 2) * size_x ** 3 * \
                               scipy.special.gammainc(sigma * (tau - 1), c * w_2) * \
                               scipy.special.gammainc(sigma * (tau - 1), c * w_3) * scipy.special.gamma(sigma * (tau - 1)) ** 2)
        den_l = 1 / n * np.sum(w_2 ** (sigma - sigma * tau) / ((1 + np.absolute(loc_1 - loc_2)) ** beta) * \
                               (sigma * tau) ** (sigma - 1) * size_x ** 2 * np.exp(sigma * tau * w_2) * \
                                scipy.special.gammainc(sigma * (tau - 1), c * w_2) * scipy.special.gamma(sigma * (tau - 1)))

    return num/den, num_l/(den_l**2)


# don't look at this
def lim_num_nodes(prior, alpha, sigma, tau, beta, size_x):

    if prior == 'GGP':

        n = 100000
        w_1 = np.random.gamma(1-sigma, 1/tau, n)
        w_2 = np.random.gamma(1-sigma, 1/tau, n)
        loc_1 = LocationsSampler(size_x, n)
        loc_2 = LocationsSampler(size_x, n)

        # m = 2 * 1/n * (np.sum(w_1 ** (- sigma) / scipy.special.gamma(1 - sigma) * np.exp(-tau * w_1)))
        m = 2 * 1 / n * (np.sum(w_1 ** (- sigma - 1) * tau ** (sigma-1)))
        ell = (2*m)**sigma/(sigma*scipy.special.gamma(1-sigma))
        lim_n_alpha = [ell * i ** (1 + sigma) for i in alpha]

        # w_bar = 1 / n * (np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
        #                 w_1 ** (-1 - sigma) / scipy.special.gamma(1 - sigma) * np.exp(-tau * w_1) * \
        #                 w_2 ** (-1 - sigma) / scipy.special.gamma(1 - sigma) * np.exp(-tau * w_2)))
        w_bar = 1 / n * (np.sum((1 - np.exp(-2 * w_1 * w_2 / ((1 + np.absolute(loc_1 - loc_2)) ** beta))) * \
                        w_1 ** (-1 - sigma -1) * tau ** (sigma-1) * \
                        w_2 ** (-1 - sigma -1) * tau ** (sigma-1)) * size_x**2)
        lim_n_e = [w_bar / 2 * i**2 for i in alpha]

    return lim_n_alpha, lim_n_e