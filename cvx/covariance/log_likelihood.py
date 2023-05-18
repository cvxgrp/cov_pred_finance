import numpy as np

def log_likelihood(returns, Sigmas, means=None):
    """
    Computes the log likelihhod assuming Gaussian returns with covariance matrix
    Sigmas and mean vector means

    param returns: numpy array where rows are vector of asset returns
    param Sigmas: numpy array of covariance matrix
    param means: numpy array of mean vector; if None, assumes zero mean
    """
    if means is None:
        means = np.zeros_like(returns)

    T, n = returns.shape

    returns = returns.reshape(T,n,1)
    means = means.reshape(T,n,1)

    dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
    Sigma_invs = np.linalg.inv(Sigmas)

    return (-n/2*np.log(2*np.pi) - 1/2*np.log(dets) - 1/2 * np.transpose(returns-means, axes=(0, 2,1)) @ Sigma_invs @ (returns-means)).flatten()




