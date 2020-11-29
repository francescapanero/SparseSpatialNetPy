from utils.GGPrnd import GGPrnd
from utils.exptiltBFRY import exptiltBFRY
import numpy as np

# --------------------------
# sample weights according to different priors: 'singlepl' or 'doublepl'
# need to specify the type of approximation to use for w0: 'finite' (etBFRY) or 'truncated' (Generalized Gamma Process
# with weights > T)
# --------------------------


def WeightsSampler(prior, approximation, t, sigma, c, tau, **kwargs):

    # sample w0
    if approximation == 'finite':
        L = kwargs['L'] if 'L' in kwargs else 10000
        z = (L * sigma / t) ** (1 / sigma) if prior == 'singlepl' else \
            (L * tau * sigma ** 2 / (t * c ** (sigma * (tau - 1)))) ** (1 / sigma)
        w0 = exptiltBFRY(sigma, z, c, L)
    if approximation == 'truncated':
        T = kwargs['T'] if 'T' in kwargs else 0.00001
        if prior == 'doublepl':
            t = t * c ** (sigma * (tau - 1)) / (sigma * tau)
        w0 = GGPrnd(t, sigma, c, T)

    # sample beta
    beta = np.ones(len(w0)) if prior == 'singlepl' else np.random.beta(sigma*tau, 1, len(w0))

    # w = w0 / beta (note that for singlepl beta=1 so w=w0)
    w = w0 / beta
    return w, w0, beta


def WeightLayers(w):
    w0 = min(w)
    wmax = max(w)
    J = np.ceil(np.log(wmax/w0)/np.log(2))
    w_layers = [w0 * (2 ** j) for j in range(int(J)+1)]  # layers
    w_layers[-1] = wmax + 0.000001
    _, bin = np.histogram(w, w_layers)
    layer = np.digitize(w, bin)
    layer = np.array(layer)
    lay = [w_layers]
    lay.append(layer)
    return lay
