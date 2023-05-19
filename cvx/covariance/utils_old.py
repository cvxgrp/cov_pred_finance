import numpy as np
import multiprocessing as mp
import cvxpy as cp
from tqdm import trange
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from arch import arch_model
from matplotlib import pyplot as plt



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

def get_turnover(ws):
        turnovers = []
        for t in range(1,len(ws)):
            w_tm = ws[t-1]
            w_t = ws[t]
            turnovers.append(np.linalg.norm(w_tm-w_t, ord=1))
        return np.mean(turnovers)


def Gaussian_likelihood(x, Sigma, mu=None):
    """
    Computes the Gaussian likelihood of x given mu and Sigma. 
    """
    if mu is None:
        mu = np.zeros(x.shape[0])
    n = x.shape[0]
    return (1 / ((2 * np.pi)**(n/2) * np.linalg.det(Sigma)**(1/2))) * np.exp(-1/2 * (x - mu).T @ np.linalg.inv(Sigma) @ (x - mu))

def get_posterior_probs(x, Sigmas, mus=None, weights=None):
    """
    Computes the posterior probabilities of each Gaussian distribution given x. 
    """
    n = x.shape[0]
    if mus is None:
        mus = np.zeros((Sigmas.shape[0], n))
    if weights is None:
        weights = np.ones(Sigmas.shape[0]) / Sigmas.shape[0]
    probs = np.zeros(Sigmas.shape[0])
    for i in range(Sigmas.shape[0]):
        probs[i] = weights[i] * Gaussian_likelihood(x, Sigmas[i], mus[i])
    return probs / np.sum(probs)

# def get_Lt()

def get_frob_change(X):
    """
    Computes frobenius norm of change in X and divides by number of rows
    """
    return np.sqrt(np.sum((X[1:] - X[:-1])**2)) / X.shape[0]

def get_ar_format(X, l):
    """
    Returns AR(l) representation of X
    """
    T = X.shape[0]
    n = X.shape[1]

    # Initialize AR(l) representation
    X_ar = np.zeros((T-l, n*l))

    for i in range(l):
        X_ar[:, i*n:(i+1)*n] = X[i:T-l+i]

    y_ar = X[l:]
        
    return X_ar, y_ar


def plot_ewma_log_likelihoods(predictors, start_date, end_date=None, T_half=63, labels=None, with_opt=False):

    if end_date is None:
        end_date = predictors[0].R.index[-1]
    dates = pd.to_datetime(predictors[0].R.loc[start_date:end_date].index)

    log_likes = []
    for predictor in predictors:
        log_likes.append(predictor.get_log_likelihoods(start_date, end_date))

    ewma_log_likes = []

    for log_like in log_likes:
        ewma_log_likes.append(get_ewmas(log_like, T_half))


    for i, ewma_log_like in enumerate(ewma_log_likes):
        if i==0 and with_opt:
            plt.plot(dates, ewma_log_like, c="k")
        else:
            plt.scatter(dates, ewma_log_like, s=2)
    



    # Place legend outside to right of plot
    if labels is not None:
        # Make legend dots bigger
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', labels=labels, scatterpoints=1, markerscale=5);

def plot_ecdf(datas, labels=None):
    """Compute ECDF for a one-dimensional array of measurements."""

    for i, data in enumerate(datas):
        # Number of data points: n
        n = len(data)

        # x-data for the ECDF: x
        x = np.sort(data)

        # y-data for the ECDF: y
        y = np.arange(1, n+1) / n

        if labels:
            plt.plot(x, y, label=labels[i])
        else:
            plt.plot(x, y)

        # Plot 5th and 10th percentile as dashed line
        # plt.plot(x, np.ones(n)*0.025, '--', color='gray', label='2.5th percentile')

        # plt.plot(x, np.ones(n)*0.05, '--', color='gray', label='5th percentile')
        # plt.plot(x, np.ones(n)*0.1, '--', color='gray', label='10th percentile')

    # Put legend to right of plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def get_metrics(log_likelihoods):
    """
    param log_likelihoods: list of log likelihoods

    returns: metrics
    """
    avg = np.mean(log_likelihoods)
    std = np.std(log_likelihoods)
    lowest = np.min(log_likelihoods)

    # 1st, 5th and 10th percentile
    p1 = np.percentile(log_likelihoods, 1)
    p5 = np.percentile(log_likelihoods, 5)
    p10 = np.percentile(log_likelihoods, 10)

    return np.round(avg,2), np.round(std,2), np.round(lowest,2), np.round(p1,2), np.round(p5,2), np.round(p10,2)

def get_yearly_garch_params(R, update_freq=2):
    """
    returns a dict of "yearly" GARCH parameters for each asset,
        strictly speaking updated every update_freq years
    """
    window_size = update_freq*250

    # Create a list of years in the data
    years = R.index.year.unique()

    all_params = {}

    for i, asset in enumerate(R.columns):
        # Initialize an empty dict to store the results of the rolling window estimation
        yearly_params = {}

        # Loop over the years
        for year in years[update_freq:]:
            # Get the data for the two previous years
            data_temp = R[R.index.year < year].values[:, i] 
            am = arch_model(data_temp, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
            res = am.fit(disp=False)
            yearly_params[year] = res.params

        for i in range(update_freq):
            yearly_params[years[i]] = yearly_params[years[update_freq+1]]

        all_params[asset] = yearly_params

    return all_params


def moving_average(X, n):
    """
    param R: pd.DataFrame of shape (T, n) where T is the number of time steps and n is the number of assets
    param n: window size

    returns the moving average of X with window size n
    """
    ret = np.cumsum(X, axis=0)
    ma_part = (ret[n:] - ret[:-n]) / n
    return np.vstack([ret[:n]/n, ma_part])


def get_hi(X, sigmas):
    """
    return hi feature
    """
    return np.maximum((X - sigmas), 0)

def get_lo(X, sigmas):
    """
    return lo feature
    """
    return np.maximum(-(X - sigmas), 0)

def get_pos(X):
    """
    return positive part 
    """

    return np.maximum(X, 0)

def get_neg(X):
    """
    return negative part
    """
    return np.maximum(-X, 0)


def get_features(R, val_end, n=8):
    """
    feature engineers the data
    """
    X = R.values

    ### Define features
    sigmas = np.std(X[:val_end], axis=0).reshape(1,-1)
    hi = get_hi(X, sigmas)
    lo = get_lo(X, sigmas)
    pos = get_pos(X)
    neg = get_neg(X)

    ma2 = moving_average(X, 2)
    sigmas2 = np.std(ma2[:val_end], axis=0).reshape(1,-1)
    hi2 = get_hi(ma2, sigmas2)
    lo2 = get_lo(ma2, sigmas2)
    pos2 = get_pos(ma2)
    neg2 = get_neg(ma2)

    ma4 = moving_average(X, 4)
    sigmas4 = np.std(ma4[:val_end], axis=0).reshape(1,-1)
    hi4 = get_hi(ma4, sigmas4)
    lo4 = get_lo(ma4, sigmas4)
    pos4 = get_pos(ma4)
    neg4 = get_neg(ma4)

    ma8 = moving_average(X, 8)
    sigmas8 = np.std(ma8[:val_end], axis=0).reshape(1,-1)
    hi8 = get_hi(ma8, sigmas8)
    lo8 = get_lo(ma8, sigmas8)
    pos8 = get_pos(ma8)
    neg8 = get_neg(ma8)

    ma16 = moving_average(X, 16)
    sigmas16 = np.std(ma16[:val_end], axis=0).reshape(1,-1)
    hi16 = get_hi(ma16, sigmas16)
    lo16 = get_lo(ma16, sigmas16)
    pos16 = get_pos(ma16)
    neg16 = get_neg(ma16)

    ma32 = moving_average(X, 32)
    sigmas32 = np.std(ma32[:val_end], axis=0).reshape(1,-1)
    hi32 = get_hi(ma32, sigmas32)
    lo32 = get_lo(ma32, sigmas32)
    pos32 = get_pos(ma32)
    neg32 = get_neg(ma32)


    if n == 32:
        features = np.hstack([hi, lo, pos, neg, ma2, ma4, ma8, ma16, ma32, hi2, hi4, hi8, hi16, hi32, lo2, lo4, lo8, lo16, lo32,\
            pos2, pos4, pos8, pos16, pos32, neg2, neg4, neg8, neg16, neg32])
    elif n== 8:
        features = np.hstack([hi, lo, pos, neg, ma2, ma4, ma8, hi2, hi4, hi8, lo2, lo4, lo8,\
            pos2, pos4, pos8, neg2, neg4, neg8])
    elif n== 16:
        features = np.hstack([hi, lo, pos, neg, ma2, ma4, ma8, ma16, hi2, hi4, hi8, hi16, lo2, lo4, lo8, lo16,\
            pos2, pos4, pos8, pos16, neg2, neg4, neg8, neg16])
    if n==4:
        features = np.hstack([hi, lo, pos, neg, ma2, ma4, hi2, hi4, lo2, lo4,\
            pos2, pos4, neg2, neg4])
    if n=="ma2":
        return ma2


    return pd.DataFrame(features, index=R.index, columns=[str(i) for i in range(features.shape[1])])

def get_R(R_whitenened, Lt_invs):
    """
    param R_whitenened: numpy array of shape (T, n, 1) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days
    param Lt_invs: numpy array of shape (T, n, n) where T is the number of time steps and n is the number of assets
        these are the inverse of the whitening matrices
    
    returns: numpy array of shape (T, n) where T is the number of time steps and n is the number of assets
    """
    T = R_whitenened.shape[0]
    n = R_whitenened.shape[1]
    R = (Lt_invs @ R_whitenened).reshape(T, n)

    return R


def get_V_hats(vols):
    """
    param vols: numpy array of shape (T, n) where T is the number of time steps and n is the number of assets
        these are the predicted volatilities over the T days

    returns: numpy array of shape (T, n, n) where T is the number of time steps and n is the number of assets
        these are the predicted diagonal volatility matrices over the T days
    """
    T = vols.shape[0]
    n = vols.shape[1]
    V_hats = np.zeros((T, n, n))
    for t in range(T):
        V_hats[t] = np.diag(vols[t])

    return V_hats

def cp_get_log_likelihood(r, Theta):
    """
    param R: array of shape (n,1) where n is the number of assets
    param Theta: precision matrix of shape (n,n) where n is the number of assets

    returns: log likelihood of R given Theta
    """
    return cp.log_det(Theta) - cp.quad_form(r, Theta) 


def max_log_likelihood_const_L(prob, w, Lt_params, Lt_times_R_vec_params, Lt_hat, Lt_times_R_vec, prob_num, ignore_dpp=False):
    """
    param whiteners: array of shape (M, n, n) where M is the number of predictor candidates, and n is the number of assets
        whiteners[i] = Li^T, where Li*Li^T=Sigma_i, the precission matrix of the i-th predictor
    param R: numpy array of shape (T, n, 1) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days

    returns the weights that maximize the log likelihood of the data given M constant whiteners 
    """
    try: # TODO: just a quick fix for with_pooling=False case
        if prob_num[0] % int(prob_num[1] / 10) == 0:
            print(f"{prob_num[0]/prob_num[1]:.0%} done...", end=" ")
    except:
        pass

    n_predictors = len(Lt_params)
    for i in range(n_predictors):
        Lt_params[i].value = Lt_hat[i]
        Lt_times_R_vec_params[i].value = Lt_times_R_vec[i]
    
    try:
        prob.solve(solver="MOSEK", ignore_dpp=ignore_dpp)
    except Exception:
        print("Solver failed...")
        prob.solve(solver="MOSEK", verbose=True)
    
    assert prob.status == "optimal"

    return w.value


def max_log_likelihood_varying_L(prob, w, Lt_params, Lt_times_R_vec_params, Lt_hat, Lt_times_R_vec, prob_num, ignore_dpp=False):
    """
    param whiteners: array of shape (M, n, n) where M is the number of predictor candidates, and n is the number of assets
        whiteners[i] = Li^T, where Li*Li^T=Sigma_i, the precission matrix of the i-th predictor
    param R: numpy array of shape (T, n, 1) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days

    returns the weights that maximize the log likelihood of the data given M constant whiteners 
    """
    try: # TODO: just a quick fix for with_pooling=False case
        if prob_num[0] % int(prob_num[1] / 10) == 0:
            print(f"{prob_num[0]/prob_num[1]:.0%} done...", end=" ")
    except:
        pass

    n_predictors = len(Lt_params)
    for i in range(n_predictors):
        Lt_params[i].value = Lt_hat[i]
        Lt_times_R_vec_params[i].value = Lt_times_R_vec[i]
    
    try:
        prob.solve(solver="MOSEK", ignore_dpp=ignore_dpp)
    except:
        print("Solver failed...")
        prob.solve(solver="MOSEK", verbose=True)
    
    assert prob.status == "optimal"

    return w.value






    

def max_log_likelihood(prob, w, prec_params, prec_at_cov_params, precisions, prec_at_covs, prob_num):
    """
    param precisions: numpy array of shape (T, M, n, n) where T is the number of time steps, M is the number of predictor candidates, and n is the number of assets
    param R: numpy array of shape (T, n, 1) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days
    """
    if prob_num[0] % int(prob_num[1] / 10) == 0:
        print(f"{prob_num[0]/prob_num[1]:.0%} done...", end=" ")

    n_predictors = len(precisions[0])
    T = len(precisions)

    # Solve problem once for speedup later
    for t in range(T):
        for i in range(n_predictors):
            prec_params[t][i].value = precisions[t][i]
            prec_at_cov_params[t][i].value = prec_at_covs[t][i]

    try:
        print("Solver failed...")
        prob.solve(solver="MOSEK", verbose=False, ignore_dpp=True)
    except:
        w_temp = np.zeros(n_predictors)
        w_temp[0] = 1
        return w_temp
    return w.value


def max_log_likelihood_const_theta(prob, w, prec_params, prec_at_cov_params, precisions, prec_at_covs, prob_num): 
    """
    param precisions: array of shape (M, n, n) M is the number of predictor candidates, and n is the number of assets
    param R: numpy array of shape (T, n) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days

    returns the weights that maximize the log likelihood of the data given M constant precision matrices
        i.e., in comparison to max_log_likelihood, this function assumes that the precision matrices are constant over time t=1,2,...,T
    """
    if prob_num[0] % int(prob_num[1] / 10) == 0:
        print(f"{prob_num[0]/prob_num[1]:.0%} done...", end=" ")

    n_predictors = len(precisions)
    for i in range(n_predictors):
        Theta_temp = precisions[i]
        prec_params[i].value = Theta_temp
        prec_at_cov_params[i].value = prec_at_covs[i]

    prob.solve(solver="MOSEK", verbose=False)
    return w.value

def get_overtriangular(M):
    """
    returns the upper triangular part of a matrix M (including the diagonal)
    """
    return M[np.triu_indices(M.shape[0], k=0)].reshape(1,-1)

def get_overtriangular_stack(Ms):
    """
    returns the upper triangular part of a list of matrices Ms (including the diagonal)\
        upper diagonals are returned as rows in a matrix
    """
    return np.vstack([get_overtriangular(M) for M in Ms])

def get_vectorform(R):
    """
    returns the form of R that is used to compute L^T * R elementwise
    """
    n = R.shape[1]
    R_new = R.copy()
    for i in range(1, n):
        R_new = np.hstack((R_new, R[:,i:]))
    return R_new

def max_log_likelihood_varying_L2(prob, w, P_param, onest_A_param, P, onest_A, prob_num, ignore_dpp=False):
    """
    param whiteners: array of shape (M, n, n) where M is the number of predictor candidates, and n is the number of assets
        whiteners[i] = Li^T, where Li*Li^T=Sigma_i, the precission matrix of the i-th predictor
    param R: numpy array of shape (T, n, 1) where T is the number of time steps and n is the number of assets
        these are the realized returns over the T days

    returns the weights that maximize the log likelihood of the data given M constant whiteners 
    """
    P_param.value = P 
    onest_A_param.value = onest_A 

    try: # TODO: just a quick fix for with_pooling=False case
        if prob_num[0] % int(prob_num[1] / 10) == 0:
            print(f"{prob_num[0]/prob_num[1]:.0%} done...", end=" ")
    except:
        pass

    try:
        prob.solve(solver="ECOS", ignore_dpp=ignore_dpp)
    except:
        print("Solver failed...")
        prob.solve(solver="ECOS", verbose=True)
    
    assert prob.status == "optimal"

    return w.value

def get_opt_weights_new(model, likelihood_window=10, w_init=None, const_theta=None, const_L=None, turnover_cons=None):
    """
    param model: Covariance predictor model
    param likelihood_window: number of past days to use to compute the log likelihood
    param w_init: initial weights to use for the convex optimization problem
    param const_theta: whether the precision matrices are constant over time or not
    """
    # print("Turnover limit: ",turnover_cons)
    Theta_hats_for_all_t = []
    prec_at_covs_for_all_t = []
    R_for_all_t = []
    A_for_all_t = []
    P_for_all_t = []

    Lt_hats_for_all_t = []
    R_vec_for_all_t = []
    Lt_times_R_vec_for_all_t = []
    n_predictors = len(model.predictors)
    n = model.n
    T = model.T

    # Parametrize problem for weighted average of precision matrices
    if const_L is None:
        assert const_theta is not None
        if const_theta == "on":
            w = cp.Variable(n_predictors)
            precision_params = [cp.Parameter((n, n), PSD=True) for _ in range(n_predictors)]
            prec_at_cov_params = [cp.Parameter((n, n)) for _ in range(n_predictors)]

            Theta_hat = cp.sum([w[i]*precision_params[i] for i in range(n_predictors)])
            trace_arg = cp.sum([w[i]*prec_at_cov_params[i] for i in range(n_predictors)])
            obj = cp.Maximize(cp.log_det(Theta_hat) - cp.trace(trace_arg))

        elif const_theta == "off":

            w = cp.Variable(n_predictors)
            precision_params = [[cp.Parameter((n, n), PSD=True) for _ in range(n_predictors)] for _ in range(likelihood_window)]
            prec_at_cov_params = [[cp.Parameter((n, n)) for _ in range(n_predictors)] for _ in range(likelihood_window)]

            Theta_hat = [cp.sum([w[i]*precision_params[t][i] for i in range(n_predictors)]) for t in range(likelihood_window)]
            trace_arg = [cp.sum([w[i]*prec_at_cov_params[t][i] for i in range(n_predictors)]) for t in range(likelihood_window)]

            log_likelihood = cp.sum([cp.log_det(Theta_hat[t]) - cp.trace(trace_arg[t]) for t in range(likelihood_window)])
            obj = cp.Maximize(cp.sum(log_likelihood))

        cons = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, cons)
        # print(prob.is_dcp(dpp=True))

        # prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0])
    else:
        assert const_theta is None
        assert const_L is not None

        if const_L == "on":
            w = cp.Variable(n_predictors)

            # Each Lt has shape number of overtriangular elements in L^T
            n_overtriangular = int(n*(n+1)/2)
            Lt_params = [cp.Parameter((1,n_overtriangular)) for _ in range(n_predictors)]
            Lt_hat = cp.sum([w[i]*Lt_params[i] for i in range(n_predictors)])

            # Compute the norm part of the objective
            Lt_times_R_vec_params = [cp.Parameter((likelihood_window, n_overtriangular)) for _ in range(n_predictors)] # Lt_m*R_t elementwise for flattened Lt overtri.
            norm_arg = cp.sum([w[i]*Lt_times_R_vec_params[i] for i in range(n_predictors)])
            sums = cp.hstack([cp.sum(norm_arg[:,i*n- int(i*(i-1)/2):(i+1)*n-int(i*(i+1)/2)], axis=1) for i in range(n)]) # TODO: I think this is right...


            diag_indices = [i*n- int(i*(i-1)/2) for i in range(n)]

        elif const_L == "off":
            ########### NEW APPROACH
            w = cp.Variable(n_predictors)
            P_param = cp.Parameter((n_predictors,n_predictors), PSD=True)
            A_param = cp.Parameter((n*likelihood_window,n_predictors))

            obj = cp.Maximize(cp.sum(cp.log(A_param@w)) - 0.5 * cp.quad_form(w, P_param))



        cons = [cp.sum(w) == 1, w >= 0]

        if turnover_cons is not None:
            w_old = cp.Parameter(n_predictors)
            # w_old.value = np.zeros(n_predictors); w_old.value[-1] = 1 # initialize to last predictor
            w_old.value = np.ones(n_predictors) / n_predictors
            cons += [cp.norm(w-w_old, 2) <= turnover_cons] 
        else:
            w_old = None
        prob = cp.Problem(obj, cons)


    # Define parameter values for each time step
    # TODO: I changed from t-1 to t (this should still be causal)
    for t in range(likelihood_window, T):
        R_t = model.R.iloc[t-(likelihood_window):t, :].values
        R_for_all_t.append(R_t)
        if const_theta == "on":
            Theta_hats_t = np.array([[*predictor.Theta_hats.values()][t] for predictor in model.predictors.values()]) # get CURRENT day's precision matrices (at time t!)
            sample_covs_t = np.sum(R_t[i].reshape(-1,1)@R_t[i].reshape(1,-1) for i in range(likelihood_window)) / likelihood_window # Don't include t!
            prec_at_covs_t = np.array([Theta_hats_t[i] @ sample_covs_t for i in range(n_predictors)])
        elif const_theta == "off":
            set_of_Theta_hats_t = []
            set_of_prec_at_covs_t = []
            for predictor in model.predictors.values():
                # Precision matrices
                Theta_hats_temp = [*predictor.Theta_hats.values()][t-(likelihood_window+1):t-1]
                set_of_Theta_hats_t.append(Theta_hats_temp)

                # (Precision @ sample covariance matrices) (just one sample per time step...)
                prec_at_covs_temp = np.array([Theta_hats_temp[i] @ R_t[i].reshape(-1,1)@R_t[i].reshape(1,-1) for i in range(likelihood_window)])
                set_of_prec_at_covs_t.append(prec_at_covs_temp)
            Theta_hats_t = np.reshape(np.array(set_of_Theta_hats_t), (likelihood_window, n_predictors, n, n))
            prec_at_covs_t = np.reshape(np.array(set_of_prec_at_covs_t), (likelihood_window, n_predictors, n, n))
        elif const_L == "on":
            Lt_hats_t = np.array([get_overtriangular([*predictor.Lt_hats.values()][t]) for predictor in model.predictors.values()]) # get CURRENT day's Cholesky matrices (at time t!)
            R_vec_t = get_vectorform(R_t)
            Lt_times_R_vec_t = np.array([R_vec_t * Lt_hats_t[i] for i in range(n_predictors)])

        elif const_L == "off":
            # TODO: I changed from t-1 to t (this should still be causal)
            R_t = model.R.iloc[t-(likelihood_window):t, :].values

            Lt_hats_t = []
            R_vec_t = get_vectorform(R_t)

            A_t = np.zeros((n*likelihood_window, n_predictors))
            all_B_t = np.zeros((likelihood_window, n_predictors, n))
            for i, predictor in enumerate(model.predictors.values()):
                # Cholesky matrices
                Lt_hats_temp = get_overtriangular_stack([*predictor.Lt_hats.values()][t-(likelihood_window):t])
                Lt_hats_t.append(Lt_hats_temp)

                ### NEW APPROACH
                # Compute A_t and P_t
                Lt_hats_temp = np.array([*predictor.Lt_hats.values()][t-(likelihood_window):t])

                A_t[:,i] = np.array([np.diag(Lt_hats_temp[j]) for j in\
                     range(likelihood_window)]).reshape(-1,)


                for j in range(likelihood_window):
                    r_j = R_t[j].reshape(-1,1)
                    Lt_j = Lt_hats_temp[j]
                    all_B_t[j,i,:] = r_j.T @ Lt_j.T
                
            P_t = np.zeros((n_predictors, n_predictors))
            for j in range(likelihood_window):
                B_j = all_B_t[j]
                P_t += B_j @ B_j.T
            # ones = np.ones((n*likelihood_window,1))
            # onest_A_t = ones.T @ A_t


            Lt_times_R_vec_t = np.array([R_vec_t * Lt_hats_t[i] for i in range(n_predictors)])
        

        if const_L == "on" or const_L == "off":
            Lt_hats_for_all_t.append(Lt_hats_t)
            R_vec_for_all_t.append(R_vec_t)
            Lt_times_R_vec_for_all_t.append(Lt_times_R_vec_t)

            P_for_all_t.append(P_t)
            A_for_all_t.append(A_t)

        else:
            Theta_hats_for_all_t.append(Theta_hats_t)
            prec_at_covs_for_all_t.append(prec_at_covs_t)
    
    # Solve problem for each t in parallel
    if const_L=="on":
        print("Solving convex problem for constant L...")
        opt_model = max_log_likelihood_const_L
    elif const_L=="off":
        print("Solving convex problem for time-varying L...")
        opt_model = max_log_likelihood_varying_L
        opt_model = max_log_likelihood_varying_L2

    elif const_theta=="on":
        print("Solving convex problem for constant theta...")
        opt_model = max_log_likelihood_const_theta
    elif const_theta=="off":
        print("Solving convex problem for time-varying theta...")
        opt_model = max_log_likelihood
    

    if turnover_cons is None: # No turnover constraint, so we can problems in parallel
        with_pooling = False # TODO: FIX
    else:
        with_pooling = False
    if with_pooling:
        # Solve in parallel
        with ProcessPoolExecutor() as executor: # TODO: This can probably be removed... Using multiprocessing.Pool instead
        # Use tqdm to display a progress bar
            if const_L == "on" or const_L == "off":
                # ws = list(tqdm(executor.map(opt_model, Lt_hats_for_all_t, R_for_all_t), total=len(Lt_hats_for_all_t)))
                # Solve problem once for speedup later
                for i in range(n_predictors):
                    Lt_params[i].value = Lt_hats_for_all_t[0][i]
                    Lt_times_R_vec_params[i].value = Lt_times_R_vec_for_all_t[0][i]
                prob.solve(ignore_dpp=model.ignore_dpp)
                


                Lt_param_list = [Lt_params for _ in range(len(R_vec_for_all_t))]
                Lt_times_R_vec_param_list = [Lt_times_R_vec_params for _ in range(len(R_vec_for_all_t))]
                prob_list = [prob for _ in range(len(R_vec_for_all_t))]
                w_list = [w for _ in range(len(R_vec_for_all_t))]
                ignore_dpp_list = [model.ignore_dpp for _ in range(len(R_vec_for_all_t))]

                # Keep track on how many problems left to solve
                prob_nums = [(i, len(R_for_all_t)) for i in range(len(R_for_all_t))]

                # Solve all problems in parallel
                pool = mp.Pool()
                # ws = pool.starmap(opt_model, zip(prob_list, w_list, Lt_param_list, Lt_times_R_vec_param_list, \
                #             Lt_hats_for_all_t, Lt_times_R_vec_for_all_t, prob_nums, ignore_dpp_list))
                
                ws = pool.starmap(opt_model, zip(prob, w, P_param, A_param,\
                     P_for_all_t[t], A_for_all_t[t], t, model.ignore_dpp))
                pool.close()
                pool.join()




            elif const_theta == "on":
                # Solve problem once for speedup later
                for i in range(n_predictors):
                    Theta_temp = Theta_hats_for_all_t[0][i]
                    precision_params[i].value = Theta_temp
                    prec_at_cov_params[i].value = prec_at_covs_for_all_t[0][i]
                prob.solve()

                prec_param_list = [precision_params for _ in range(len(R_for_all_t))]
                prec_at_cov_param_list = [prec_at_cov_params for _ in range(len(R_for_all_t))]
                prob_list = [prob for _ in range(len(R_for_all_t))]
                w_list = [w for _ in range(len(R_for_all_t))]

                # Keep track on how many problems left to solve
                prob_nums = [(i, len(R_for_all_t)) for i in range(len(R_for_all_t))]

                pool = mp.Pool()
                ws = pool.starmap(opt_model, zip(prob_list, w_list, prec_param_list, prec_at_cov_param_list, \
                            Theta_hats_for_all_t, prec_at_covs_for_all_t, prob_nums))
                pool.close()
                pool.join()
            elif const_theta == "off":
                # Solve problem once for speedup later
                for t in range(likelihood_window):
                    for i in range(n_predictors):
                        precision_params[t][i].value = Theta_hats_for_all_t[0][t][i]
                        prec_at_cov_params[t][i].value = prec_at_covs_for_all_t[0][t][i]
                prob.solve(ignore_dpp=True)

                prec_param_list = [precision_params for _ in range(len(R_for_all_t))]
                prec_at_cov_param_list = [prec_at_cov_params for _ in range(len(R_for_all_t))]
                prob_list = [prob for _ in range(len(R_for_all_t))]
                w_list = [w for _ in range(len(R_for_all_t))]

                # Keep track on how many problems left to solve
                prob_nums = [(i, len(R_for_all_t)) for i in range(len(R_for_all_t))]

                pool = mp.Pool()
                ws = pool.starmap(opt_model, zip(prob_list, w_list, prec_param_list, prec_at_cov_param_list, \
                            Theta_hats_for_all_t, prec_at_covs_for_all_t, prob_nums))
                pool.close()
                pool.join()
    else:
        # Solve sequentially
        ws = []
        for t in trange(len(Lt_hats_for_all_t)):
            if const_L == "off":


                # w_t = opt_model(prob, w, Lt_params, Lt_times_R_vec_params,
                # Lt_hats_for_all_t[t], Lt_times_R_vec_for_all_t[t], t,
                # model.ignore_dpp)

                w_t = opt_model(prob, w, P_param, A_param,\
                     P_for_all_t[t], A_for_all_t[t], t, model.ignore_dpp)


                ws.append(w_t.copy())
                if w_old is not None:
                    w_old.value = w_t.copy()

            # if const_L == "on" or const_L == "off":
            #     w = opt_model(Lt_hats_for_all_t[t], R_for_all_t[t])
            # else:
            #     w = opt_model(Theta_hats_for_all_t[t], R_for_all_t[t])
        print("Finishing...")

    # TODO: I changed from likelihood_window+1 to likelihood_window
    ws = pd.DataFrame(ws, index = model.R.index[likelihood_window:])
    return ws


def get_opt_weights_old(model, t, prob, w, theta_params, likelihood_window=10, w_init=None, const_theta=True):
    """
    Choose the weights to combine the inverse covariance matrices by solving the following convex problem:  what (constant) weights would have maximized the log likelihood on the last 'likelihood_window' days? This is a convex problem, so we can solve it with cvxpy.

    param model: Covariance predictor model
    param t: time step (pandas date-time object) for which we want to compute the optimal weights
    param prob: cvxpy problem object
    param w: cvxpy variable object
    param theta_params: array of cvxpy parameters for the precision matrix
    param likelihood_window: number of past days to use to compute the log likelihood
    """
    n = model.n
    n_predictors = len(model.predictors)

    # t = np.where(model.R.index == t)[0][0] # get the index of the time step t (t is originally a pandas date-time object)
    R_temp = model.R.iloc[t-(likelihood_window+1):t-1, :].values # Don't include t; avoid look-ahead bias 
    
    Theta_hats_t = [[*predictor.Theta_hats.values()][t] for predictor in model.predictors.values()] # get CURRENT day's precision matrices (at time t!)
    if const_theta:
        for i in range(n_predictors):
            theta_params[i].value = Theta_hats_t[i] # set the value of the cvxpy parameters to the current precision matrices
        w_opt = max_log_likelihood_const_theta(prob, w, theta_params, R_temp)
    else:
        set_of_Theta_hats = []
        for predictor in model.predictors.values():
            Theta_hats_temp = [*predictor.Theta_hats.values()][t-(likelihood_window+1):t-1]
            set_of_Theta_hats.append(Theta_hats_temp)
        precisions = np.reshape(np.array(set_of_Theta_hats), (likelihood_window, n_predictors, n, n))
        w_opt = max_log_likelihood(precisions, R_temp)


    Theta_hat_opt = np.sum(w_opt[i]*Theta_hats_t[i] for i in range(len(w_opt)))

    return w_opt, Theta_hat_opt

    ##### Can probably remove this code below #####
    # for predictor in model.predictors.values():
    #     Theta_hats_temp = [*predictor.Theta_hats.values()][t-(likelihood_window+1):t-1] # Don't include t; avoid look-ahead bias
    #     set_of_Theta_hats.append(Theta_hats_temp) 

    # precisions = np.reshape(np.array(set_of_Theta_hats), (likelihood_window, n_predictors, n, n))

    # return max_log_likelihood(precisions, R_temp)

    # set_of_Theta_hats = np.array(set_of_Theta_hats).reshape(likelihood_window, n_predictors, n**2) # batch_size, n_predictors, n, n

    # weighted_Theta_hats = []
    # for i in range(likelihood_window):
    #     Theta_vec = cp.sum(cp.multiply(w, set_of_Theta_hats[i]), axis=0)
    #     weighted_Theta_hats.append(cp.reshape(Theta_vec, (n,n)))


    # # Compute the log likelihood (ignore the constant terms)
    # log_likelihood = cp.sum([cp_get_log_likelihood(R_temp[i], weighted_Theta_hats[i]) for i in range(likelihood_window)])

    # obj = cp.Maximize(log_likelihood)
    # cons = [cp.sum(w)==1, w>=0]
    # if w_init is not None:
    #     const += [cp.norm(w-w_init, 2) <= 0.1]
    # prob = cp.Problem(obj, cons)

    # prob.solve(solver="MOSEK")
    # w_opt = w.value

    # theta = np.array([w_opt[i]*set_of_Theta_hats[-1][i].reshape(n,n) for i in range(n_predictors)]).sum(axis=0)

    # return w_opt, theta




def get_realized_covs(R):
    """
    param R: numpy array where rows are vector of asset returns for t=0,1,...
        R has shape (T, n) where T is the number of days and n is the number of assets

    returns: (numpy array) list of r_t*r_t' (matrix multiplication) for all days, i.e,
        "daily realized covariances"
    """
    T = R.shape[0]
    n = R.shape[1]
    R = R.reshape(T,n,1)

    return R @ R.transpose(0,2,1)
    

def get_realized_vars(R):
    """
    param R: numpy array where rows are vector of asset returns for t=0,1,...
        R has shape (T, n) where T is the number of days and n is the number of assets

    returns: (numpy array) list of diag(r_t^2) for all days, i.e,\
        "daily realized variances"
    """


    variances = []
    
    for t in range(R.shape[0]):
        r_t = R[t, :].reshape(-1,)
        variances.append(np.diag(r_t**2))
    return np.array(variances)

def from_cov_to_corr(Sigma):
    """
    returns correlation matrix R corresponding to covariance matrix Sigma
    """
    V = np.sqrt(np.diag(Sigma))
    outer_V = np.outer(V, V)
    corr = Sigma / outer_V
    return corr

def get_realized_sizes(R):
    """
    param R: numpy array where rows are vector of asset returns for t=0,1,...
        R has shape (T, n) where T is the number of days and n is the number of assets

        returns: (numpy array) list of r_t^Tr_t for all days (t), i.e,
            "daily realized sizes"
    """
    sizes = []
    for t in range(R.shape[0]):
        r_t = R[t, :].reshape(-1,1)
        sizes.append(r_t.T @ r_t)
    return np.array(sizes)

def get_next_ewma(EWMA, y_last, t, beta):
    """
    param EWMA: EWMA at time t-1
    param y_last: observation at time t-1
    param t: current time step
    param beta: EWMA exponential forgetting parameter

    returns: EWMA estimate at time t (note that this does not depend on y_t)
    """

    old_weight = (beta-beta**t)/(1-beta**t)
    new_weight = (1-beta) / (1-beta**t)

    return old_weight*EWMA + new_weight*y_last

def get_ewmas(y, T_half):
    """
    y: array with measurements for times t=1,2,...,T=len(y)
    T_half: EWMA half life

    returns: list of EWMAs for times t=2,3,...,T+1 = len(y)


    Note: We define EWMA_t as a function of the 
    observations up to time t-1. This means that
    y = [y_1,y_2,...,y_T] (for some T), while
    EWMA = [EWMA_2, EWMA_3, ..., EWMA_{T+1}]
    This way we don't get a "look-ahead bias" in the EWMA
    """

    beta = np.exp(-np.log(2)/T_half)
    EWMA_t = 0
    EWMAs = []
    for t in range(1,y.shape[0]+1): # First EWMA is for t=2 
        y_last = y[t-1] # Note zero-indexing
        EWMA_t = get_next_ewma(EWMA_t, y_last, t, beta)
        EWMAs.append(EWMA_t)
    return np.array(EWMAs)

def get_r_tildes(R, Ws):
    """
    param Ws: array of whitening matrices for t=1,2,..., (i.e. L^t, where LL^T=Sigma^-1)
    param R: array of returns to whiten for t=1,2,..., 

    returns numpy array with r_tilde_hats as rows
    """
    T = R.shape[0]
    n = R.shape[1]
    R = R.reshape(T,n,1)
    return Ws @ R

def get_cholesky(Sigma_hats):
    """
    returns: Cholesky (transpose) decomposition of Sigma_hat^-1 for each Sigma_hat in Sigma_hats, i.e.,
        LL^T = Sigma_hat^-1;
        Note: returns L^T
    """
    Lts = []
    I = np.eye(Sigma_hats[0].shape[1])
    for Sigma_hat in Sigma_hats:
        Sigma_hat_inv = np.linalg.solve(Sigma_hat, I)
        L = np.linalg.cholesky(Sigma_hat_inv)
        Lts.append(L.T)
    return np.array(Lts)

####### I think the folloiwng functions are not used anymore, but let's keep them for now ########

def get_log_likelihood(x, Sigma, mu=None):
    """
    param x: numpy array of shape (n,1)
    param Sigma: numpy array of shape (n,n)
    param mu: numpy array of shape (n,1)
        if None, mu is set to zero vector

    returns: log-likelihood of x under multivariate normal distribution
    """
    x = x.reshape(-1,1)
    n = Sigma.shape[0]
    if mu is None:
        mu = np.zeros((n,1))
    if len(Sigma) == 1:
        det = Sigma
        Sigma_inv = 1 / Sigma
    else:
        det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)

    return float(-n/2*np.log(2*np.pi) - 1/2*np.log(det) -1/2*(x-mu).T @ Sigma_inv@(x-mu))



def get_avg_likelihood(R, Sigma_hats):
    """
    returns: (averaged) log-likelihood of observed returns R, under the Gaussian distribution
        with mean zero and covariances in Sigma_hats

    param R: rows are n-dimensinal return vectors for times t=1,...,T
    param Sigma_hats: array-like vector of corresponding covariance matrices
    """
    daily_likelihoods = get_daily_likelihood(R, Sigma_hats)
    
    return np.mean(daily_likelihoods)


def get_daily_likelihood(R, Sigma_hats, in_parallel=False):
    """
    returns: log-likelihood of observed returns R, under the Gaussian distribution
        with mean zero and covariances in Sigma_hats
    """
    # n assets
    n = R.shape[1]
    mu = np.zeros((n,1))

    log_likelihoods = []

    if in_parallel:
        # Compute log-likelihood for each day in parallel
        pool = mp.Pool()
        log_likelihoods = pool.starmap(get_log_likelihood, zip(R, Sigma_hats))
        pool.close()
        pool.join()

        return np.array(log_likelihoods)

    else:
        # Compute log-likelihood for each day sequentially
        for t in range(R.shape[0]):
            r_t = R[t,:].reshape(-1,1)
            Sigma_hat_t = Sigma_hats[t]
            log_likelihoods.append(get_log_likelihood(r_t, Sigma_hat_t, mu))

        return np.array(log_likelihoods)
        