import numpy as np


def loggrad(temp1, temp2, pw_outer):
    grad = np.sum([temp1, np.negative(temp2), np.negative(np.multiply(2, pw_outer))], axis=0)
    # grad = np.sum([temp1, np.negative(temp2), np.negative(ps_outer), np.negative(w**2)], axis=0)
    return grad


# def loggrad_beta(temp1, pw_outer):
#     grad = np.sum([temp1, np.negative(np.multiply(2, pw_outer))], axis=0)
#     # grad = np.sum([temp1, np.negative(temp2), np.negative(pw_outer)], axis=0)
#     return grad
