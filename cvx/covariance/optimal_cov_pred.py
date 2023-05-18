import cvxpy as cp
import numpy as np

def get_Sigmas_from_Lts(Lts, n):
    """
    Returns covariance matrices given array of overtriangular part of L^T 
        where L*L^T = Sigma^{-1}
    """
    Sigmas = []
    for Lt in Lts:
        Lt_temp = np.zeros((n, n))
        mask = np.triu(np.ones((n,n)), k=0).astype(bool)
        Lt_temp[mask] = Lt
        Sigma  = np.linalg.solve(Lt_temp.T@Lt_temp, np.eye(n))
        Sigmas.append(Sigma)
    
    return np.array(Sigmas)

def get_l_multi(R, Lts):
    """
    param R: pandas dataframe of returns, T x n
    param Lts: list of cvxpy transpose of Cholesky of precision matrix varbiables
    """
    n = R.shape[1]
    T = R.shape[0]

    # Initialize list of log-likelihoods
    l = - 1/2 * cp.square(cp.sum(cp.multiply(Lts[:, :n], R), axis=1)) + cp.log(Lts[:, 0]) 


    count = n
    for i in range(1, n):
        start = count
        end = count + n - i
        l += - 1/2 * cp.square(cp.sum(cp.multiply(Lts[:, start:end], R[:, i:]), axis=1)) + cp.log(Lts[:, count])
        count += n - i
    


    l = cp.reshape(l, (l.shape[0], 1)) - n/2*np.log(2*np.pi)
    return l

def get_prob_multi(T_half, R, lamda, monthly_update=False, quarterly_updates=False, frob_lim=1000):
    T = R.shape[0]
    n = R.shape[1]
    Lts = []
    const = []

    Lts = cp.Variable((T, int(n*(n+1)/2)))
    # Add smoothness constraint on Lts
    const += [cp.norm(Lts[1:] - Lts[:-1], "fro") <= frob_lim * Lts.shape[0]] 
    # const += [cp.norm(Lts[1:] - Lts[:-1], axis=1) <= frob_lim] 

    # const += [cp.sum(cp.square(Lts[1:] - Lts[:-1]), axis=1) <= frob_lim]

    l = get_l_multi(R.values, Lts)

    # Get monthly average log-likelihoods
    # monthly_counts = R.resample("M").count().iloc[:,0].values
    # months =  R.resample("M").count().index
    # L = cp.Variable(len(monthly_counts))

    # t = 0
    # starts = []
    # ends = []
    # for i in range(len(monthly_counts)):
    #     start = t
    #     end = t + monthly_counts[i]
    #     starts.append(start)
    #     ends.append(end)

    #     # const += [L[i] <= cp.sum(l[start:end]) / (end-start)]
    #     t += monthly_counts[i]
    # starts = np.array(starts)
    # ends = np.array(ends)
    # const += [L <= cp.sum(l[starts:ends], axis=1) / monthly_counts]

    # # Maximize monthly average log-likelihood
    # obj = cp.Maximize(cp.sum(L)/len(monthly_counts))
    
    # prob = cp.Problem(obj, const)
    # print(prob.is_dcp())
    # prob.solve()

    # return prob, l, Lts, months

    # For maximizing average log-likelihood
    obj = cp.Maximize(1/T*cp.sum(l))

    prob = cp.Problem(obj, const)
    prob.solve()

    return prob, l, Lts


    
    beta = np.exp(-np.log(2)/T_half)

    a1s = np.array([(beta-beta**t) / (1-beta**t) for t in range(2,T+1)]).reshape(-1,1)
    a2s = np.array([(1-beta) / (1-beta**t) for t in range(2,T+1)]).reshape(-1,1)

    L = cp.Variable((T,1))

    const = [L[1:T] <= cp.multiply(a1s, L[0:T-1]) + cp.multiply(a2s, l[1:T])]
    const += [L[0] <= l[0]]
    # Uncomment next line to add constraint on change in Lts
    # const += [cp.norm(Lts[1:] - Lts[:-1], "fro") <= 1000]

    row_vars = []
    if monthly_update:
        n_months = int(np.floor(T/21))
        for i in range(n_months):
            start = i*21
            end = (i+1)*21

            # Create ith temporary variable
            row_var = cp.Variable((1, int(n*(n+1)/2)))
            const += [Lts[start:end, :] == row_var]
            row_vars.append(row_var)
    elif quarterly_updates:
        n_quarts = int(np.floor(T/63))
        for i in range(n_quarts):
            start = i*63
            end = (i+1)*63

            # Create ith temporary variable
            row_var = cp.Variable((1, int(n*(n+1)/2)))
            const += [Lts[start:end, :] == row_var]
            row_vars.append(row_var)

    obj = cp.Maximize(1/T*cp.sum(L))
    const += [cp.norm(Lts[1:] - Lts[:-1], "fro") <= frob_lim * Lts.shape[0]]
    # const += [cp.square(cp.sum(Lts[1:] - Lts[:-1], axis=1)) <= 0.001]


    # obj = cp.Maximize(-1/lamda * cp.log_sum_exp(-lamda*L + np.log(T)))

    prob = cp.Problem(obj, const)

    return prob, L, l, Lts, row_vars

