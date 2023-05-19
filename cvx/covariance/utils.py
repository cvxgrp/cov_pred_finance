import numpy as np

def _regularize_correlation(R, r):
    """
    param Rs: Txnxn numpy array of correlation matrices
    param r: float, rank of low rank component 

    returns: low rank + diag approximation of R\
        R_hat = sum_i^r lambda_i q_i q_i' + E, where E is diagonal,
        defined so that R_hat has unit diagonal; lamda_i, q_i are eigenvalues
        and eigenvectors of R (the r first, in descending order)
    """
    eig_decomps = np.linalg.eigh(R)
    Lamda = eig_decomps[0]
    Q = eig_decomps[1]

    # Sort eigenvalues in descending order
    Lamda = Lamda[:, ::-1]
    Q = Q[:, :, ::-1]

    # Make Lamdas diagonal
    Lamda = np.stack([np.diag(lamda) for lamda in Lamda])

    # Get low rank component
    Lamda_r = Lamda[:, :r, :r]
    Q_r = Q[:, :, :r]
    R_lo = Q_r @ Lamda_r @ Q_r.transpose(0, 2, 1)

    # Get diagonal component 
    D = np.stack([np.diag(np.diag(R[i, :, :]-R_lo[i, :, :])) for i in range(R.shape[0])])
    
    # Create low rank approximation
    return R_lo + D