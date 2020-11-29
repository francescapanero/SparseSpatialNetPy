import numpy as np

# --------------------------
# function to generate weights from exptiltBFRY distribution: etBFRY(sigma, z, c)
# mind that z varies depending if the prior is single or double power law
# --------------------------


def exptiltBFRY(sigma, z, c, size):

    # simulate sociabilities from exponentially tilted BFRY(sigma, z, c)
    g = np.random.gamma(1 - sigma, 1, size)
    unif = np.random.rand(size)
    w0 = np.multiply(g, np.power(((z + c) ** sigma) * (1 - unif) + (c ** sigma) * unif, -1 / sigma))

    return w0