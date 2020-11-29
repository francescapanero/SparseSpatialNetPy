from GGPrnd import GGPrnd
from GGPdoublepl import GGPdoublepl
from exptiltBFRY import exptiltBFRY

# currently supports types GGP, exponentially tilted BFRY and double power law

# for type "GGP": **args: T =
# for "exptiltBFRY": **args: L =
# for "doublepl": **args: T = , c =


def WeightsSampler(type, alpha, sigma, tau, **kwargs):

    if type == "GGP":  # power law with exp sigma
        T = kwargs['T']  # threshold for GGP and exponential
        w = GGPrnd(alpha, sigma, tau, T)
        return w
    elif type == "exptiltBFRY":  # power law with exp sigma but fixed numb of nodes L
        L = kwargs['L']  # number of nodes
        w = exptiltBFRY(alpha, sigma, tau, L)
        return w
    elif type == "doublepl":  # double power law
        T = kwargs['T']
        c = kwargs['c']
        [w, betaw, w0] = GGPdoublepl(alpha, sigma, tau, T, c)
        return w, betaw, w0