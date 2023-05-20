import numpy as np
import pandas as pd

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


def rolling_window(returns, memory, min_periods=20):
    min_periods = max(min_periods, 1)

    times = returns.index
    assets = returns.columns

    returns = returns.values


    Sigmas = np.zeros((returns.shape[0], returns.shape[1], returns.shape[1]))
    Sigmas[0] = np.outer(returns[0], returns[0])

    for t in range(1, returns.shape[0]):
        alpha_old = 1 / min(t+1, memory)
        alpha_new = 1 / min(t+2, memory)

        if t>= memory:
            Sigmas[t] = alpha_new/alpha_old * Sigmas[t-1] + alpha_new*(np.outer(returns[t], returns[t]) - np.outer(returns[t-memory], returns[t-memory]))
        else:
            Sigmas[t] = alpha_new/alpha_old * Sigmas[t-1] + alpha_new*(np.outer(returns[t], returns[t]))
    
    Sigmas = Sigmas[min_periods-1:]
    times = times[min_periods-1:]
        
    return {times[t]:pd.DataFrame(Sigmas[t], index=assets, columns=assets) for t in range(len(times))}


def from_row_to_covariance(row, n):
    """
    Convert upper diagonal part of covariance matrix to a covariance matrix
    """
    Sigma = np.zeros((n,n))

    # set upper triangular part
    upper_mask = np.triu(np.ones((n,n)), k=0).astype(bool)
    Sigma[upper_mask] = row

    # set lower triangular part
    lower_mask = np.tril(np.ones((n,n)), k=0).astype(bool)
    Sigma[lower_mask] = Sigma.T[lower_mask]
    return Sigma

def from_row_matrix_to_covariance(M, n):
    """
    Convert Tx(n(n+1)/2) matrix of upper diagonal parts of covariance matrices to a Txnxn matrix of covariance matrices
    """
    Sigmas = []
    T = M.shape[0]
    for t in range(T):
        Sigmas.append(from_row_to_covariance(M[t], n))
    return np.array(Sigmas)


def regularize_covariance(Sigmas, lamda):
    """
    Adds lamda*diag(Sigma) to each covariance (Sigma) matrix in Sigmas

    param Sigmas: dictionary of covariance matrices
    param lamda: scalar
    """
    for key in Sigmas.keys():
        Sigmas[key] = Sigmas[key] + lamda*np.diag(np.diag(Sigmas[key]))

    return Sigmas

def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    return x, y

