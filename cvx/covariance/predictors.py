from cvx.covariance.utils import *
import numpy as np
from tqdm import trange

import pandas as pd
from sklearn.mixture import GaussianMixture


class CovariancePredictor:

    def whiten_returns(self):
        """
        """
        # Check if Lt_hats has been created
        if not hasattr(self, "Lt_hats"):
            self.get_Lt_hats()

        Lt_array = np.array([*self.Lt_hats.values()]).reshape(self.T, self.n, self.n)
        R_array = self.R.values.reshape(self.T, self.n, 1)

        if not hasattr(self, "R_whitened"):
            # Whiten returns
            R_whitened = Lt_array @ R_array
            self.R_whitened = pd.DataFrame(R_whitened.reshape(self.T, self.n), index=self.R.index, columns=self.R.columns)




    def get_Theta_hats(self, start_date=None, end_date=None):
        """
        creates dictionary of precision matrices, starting at time init_cutoff to avoid singularity problems
        alternates the other dictionaries, like R, Sigma_hats, etc. to match the time-index of Theta_hats
        """
        if end_date is not None:
            end = self.R.index.get_loc(end_date) + 1 # +1 because of python indexing
        else:
            end = self.R.shape[0]
        if start_date is not None:
            start = self.R.index.get_loc(start_date) # +1 because of python indexing
        else:
            start = 0
        self.R = self.R.iloc[start:end]
        self.Sigma_hats = {i: self.Sigma_hats[i] for i in [*self.Sigma_hats.keys()][start:end]}
        self.Theta_hats = {i: np.linalg.inv(self.Sigma_hats[i]) for i in [*self.Sigma_hats.keys()]}
        
        self.T = self.R.shape[0]

        assert list(self.R.index) == list(self.Theta_hats.keys())
        assert list(self.R.index) == list(self.Sigma_hats.keys())

    def get_Lt_hats(self, start_date=None, end_date=None):
        """
        creates dictionary of (transpose of) Cholesky factors of precision matrices, starting at time init_cutoff to avoid singularity problems
        alternates the other dictionaries, like R, Sigma_hats, etc. to match the time-index of Lt_hats
        "on the way" also creates the Theta_hats dictionary
        """
        self.get_Theta_hats(start_date=start_date, end_date=end_date)
        self.Lt_hats = {i: np.linalg.cholesky(self.Theta_hats[i]).T for i in self.Theta_hats.keys()}
        self.Lt_inv_hats = {i: np.linalg.inv(self.Lt_hats[i]) for i in self.Lt_hats.keys()}

        assert list(self.R.index) == list(self.Theta_hats.keys())
        assert list(self.R.index) == list(self.Sigma_hats.keys())
        assert list(self.R.index) == list(self.Lt_hats.keys())
        assert list(self.R.index) == list(self.Lt_inv_hats.keys())


    def get_log_likelihood_helper_(self, X, Sigmas, start, end, mus=None):
        """
        param X: numpy array of shape (batch_size, n, 1)
        param Sigmas: numpy array of shape (batch_size, n, n)
        param mus: numpy array of shape (batch_size, n, 1)
            if None, mu is set to zero matrix

        returns: log-likelihood for each day of X under multivariate normal distribution
        """
        # x = x.reshape(-1,1)
        n = Sigmas[-1].shape[0]
        if mus is None:
            mus = np.zeros(X.shape)
        dets = np.linalg.det(Sigmas).reshape(len(Sigmas), 1, 1)
        Sigma_invs = np.linalg.inv(Sigmas)

        log_likelihoods = -n/2*np.log(2*np.pi) - 1/2*np.log(dets) - 1/2 * np.transpose(X-mus, axes=(0, 2,1)) @ Sigma_invs @ (X-mus)

        self.log_likelihoods = pd.DataFrame(log_likelihoods.reshape(-1,1), index=self.R.iloc[start:end].index)

        return self.log_likelihoods.values

    def get_log_likelihoods(self, start=0, end=None, X = None, Sigmas=None):
        """
        param X: numpy array of shape (batch_size, n, 1)
        param Sigmas: numpy array of shape (batch_size, n, n)
        param start: start time (integer or pd.to_date_time object)

        returns: log-likelihood of observed returns R, under the Gaussian distribution
            starting at time start
        """

        if type(start)==int:
            pass
        else:
            start = self.R.index.get_loc(start)
            if end is None:
                end = -1
            else:
                end = self.R.index.get_loc(end)+1 # because loc includes end_date

        if X is None:
            R = self.R.values.reshape(self.T, self.n, 1)[start:end]
            Sigma_hats = np.array([*self.Sigma_hats.values()]).reshape(self.T, self.n, self.n)[start:end]
        else:
            assert Sigmas is not None
            R = X.reshape(X.shape[0], self.n, 1)[start:end]
            Sigma_hats = Sigmas.reshape(Sigmas.shape[0], self.n, self.n)[start:end]
        
        return self.get_log_likelihood_helper_(R, Sigma_hats, start, end).reshape(-1,)



class MgarchPredictor(CovariancePredictor):
    def __init__(self, R) -> None:
        pass



class GoldStandard(CovariancePredictor):
    def __init__(self, R, method="sample_full", T_half=50, init_cutoff=20) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        """
        self.n = R.shape[1]
        self.T_half = T_half

        if method == "sample_full":
            Sigma = np.cov(R, rowvar=False)

            R_sampled = np.random.multivariate_normal(np.zeros((Sigma.shape[0],)), Sigma, R.shape[0])

            self.Sigma_hats = {i: Sigma for i in R.index}

            self.R = pd.DataFrame(R_sampled, index=R.index, columns=R.columns)

        elif method == "two_sided_ewma":
            realized_covs_forward = get_realized_covs(R.values)
            Sigma_hat_forward = get_ewmas(realized_covs_forward, T_half=self.T_half)

            realized_covs_backward = get_realized_covs(R.values[::-1])
            Sigma_hat_backward = get_ewmas(realized_covs_backward, T_half=self.T_half)


            # Define two-sided EWMA covariance matrix as average of forward and backward
            # First remove begining and end to avoid singularities
            R = R.iloc[init_cutoff:-init_cutoff]
            Sigma_hat_forward = Sigma_hat_forward[init_cutoff:-init_cutoff]
            Sigma_hat_backward = Sigma_hat_backward[init_cutoff:-init_cutoff]

            self.Sigma_hats = {R.index[i]: (Sigma_hat_forward[i] + Sigma_hat_backward[i])/2 for i in range(len(R.index))}
            self.R = R

        self.T = R.shape[0]
        
        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())



class ConstantPredictor(CovariancePredictor):
    def __init__(self, R, train_frac) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets

        returns covariance matrix predictions using a constant covarince matrix,
            defined as the sample covariance matrix of the training data.
        """
        self.R = R
        self.n = R.shape[1]
        self.T = R.shape[0]
        train_len = int(train_frac*R.shape[0])
        self.train_frac = train_frac
        self.train_len = train_len

        # Define mixture model
        R_train = R.iloc[:train_len]
        self.R_train = R_train

        # Create weighted covaricance matrices
        self.Sigma_hats = {}
        for t in range(R.shape[0]):
            Sigma_hat = np.cov(R_train, rowvar=False)
            self.Sigma_hats[R.index[t]] = Sigma_hat


        assert list(self.R.index) == list(self.Sigma_hats.keys())
        

            


class WeightedGMMPredictor(CovariancePredictor):
    def __init__(self, R, n_gmm_comp, train_frac, l=10, ar=True, yearly_model=True) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets

        returns covariance matrix predictions using a weighted GMM approach.
            A GMM is fit to the training data, and an AR(l) model is trained on the
            GMM probabilities. The covariance matrix is then predicted as the weighted
            sum of the covariance matrices of the GMM components.
        """
        self.R = R
        self.n = R.shape[1]
        self.T = R.shape[0]
        train_len = int(train_frac*R.shape[0])
        self.train_frac = train_frac
        self.train_len = train_len
        self.n_gmm_comp = n_gmm_comp

        if yearly_model:
            # Define mixture model
            yearly_gmms = self.get_yearly_gmms(R, n_comps=n_gmm_comp)
            self.yearly_gmms = yearly_gmms
            years = R.index.year.unique()

            # Initialize weights (cheating first two year)
            R_0 = R[R.index.year <= years[1]].values
            gmm_0 = yearly_gmms[years[1]]
            gmm_probs_0 = gmm_0.predict_proba(R_0)
            all_weights = gmm_probs_0

            for i in trange(2,len(years)):
                year = years[i]
                year_prev = years[i-1]
                R_train = R[R.index.year < year].values
                R_test = R[R.index.year == year].values

                gmm = yearly_gmms[year_prev]
                gmm_probs_train = gmm.predict_proba(R_train)

                # Build gmm probs AR model
                if ar:
                    X_ar_train, y_ar_train = get_ar_format(gmm_probs_train, l)
                    A, _ = self.get_markov_ar(X_ar_train, y_ar_train, l)


                    gmm_probs_test = gmm.predict_proba(np.vstack([R_train[-l:], R_test]))
                    X_ar_test, _ = get_ar_format(gmm_probs_test, l)
                    weights = (A @ X_ar_test.T).T

                    all_weights = np.vstack([all_weights, weights])


                else:
                    # TODO
                    pass
                    weights = self.gmm_probs[:-1]
                    weights = np.vstack([self.gmm_probs[0], weights])
                
                self.weights = all_weights


        elif not yearly_model: # Only use one GMM model trained on train set
            # Define mixture model
            R_train = R.iloc[:train_len]
            self.R_train = R_train
            gmm = GaussianMixture(n_components=n_gmm_comp).fit(R_train.values)
            self.gmm = gmm

            # Get GMM probabilities
            self.gmm_probs = gmm.predict_proba(R.values)

            # Build gmm probs AR model
            if ar:
                X_ar_train, y_ar_train = get_ar_format(self.gmm_probs[:train_len], l)
                A, _ = self.get_markov_ar(X_ar_train, y_ar_train, l)

                # ar_model = LinearRegression().fit(X_ar_train, y_ar_train)

                X_ar, _ = get_ar_format(self.gmm_probs, l)
                weights = (A @ X_ar.T).T
                weights = np.vstack([self.gmm_probs[:l], weights])

            else:
                weights = self.gmm_probs[:-1]
                weights = np.vstack([self.gmm_probs[0], weights])
            
            self.weights = weights

        

        # Create weighted covaricance matrices
        self.Sigma_hats = {}
        self.r_hats = []

        if yearly_model:
            for t in range(R.shape[0]):
                year = R.index[t].year
                if year == years[0] or year == years[1]: # Cheating first two years
                    year_prev = years[1]
                else:
                    year_prev = year - 1

                gmm = yearly_gmms[year_prev]
                Sigma_hat = np.zeros((R.shape[1], R.shape[1]))
                r_hat = np.zeros(R.shape[1])
                for i in range(n_gmm_comp):
                    Sigma_hat += self.weights[t,i]*gmm.covariances_[i]
                    r_hat += self.weights[t,i]*gmm.means_[i]
                
                self.Sigma_hats[R.index[t]] = Sigma_hat
                self.r_hats.append(r_hat)
        else:
            for t in range(R.shape[0]):
                Sigma_hat = np.zeros((R.shape[1], R.shape[1]))
                r_hat = np.zeros(R.shape[1])
                for i in range(n_gmm_comp):
                    Sigma_hat += self.weights[t,i]*gmm.covariances_[i]
                    r_hat += self.weights[t,i]*gmm.means_[i]
                
                self.Sigma_hats[R.index[t]] = Sigma_hat
                self.r_hats.append(r_hat)
                
        self.r_hats = pd.DataFrame(self.r_hats, index=R.index)


        assert list(self.R.index) == list(self.Sigma_hats.keys())
        assert list(self.R.index) == list(self.r_hats.index)

    def get_yearly_gmms(self, R, n_comps=5):
        """
        returns a dict of "yearly" GMMs for each asset
        """
        years = R.index.year.unique()
        yearly_gmms = {}
        for year in years:
            R_temp = R[R.index.year <= year].values
            gmm = GaussianMixture(n_comps).fit(R_temp)
            yearly_gmms[year] = gmm

        return yearly_gmms


    @staticmethod
    def get_markov_ar(X, y, l):
        """
        Creates Markov AR(l) for X and y, where X is the AR(l) representation of the data and y is the next time step.
        Incoorporates probability constraints.
        """
        n = y.shape[1]
        A = cp.Variable((n, n*l))
        alpha = cp.Variable(l)

        obj = cp.sum_squares(A@X.T - y.T)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve()

        cons = [A >= 0]

        one_vec = np.ones((n,1))
        for i in range(l):
            cons += [A[:,i*n:(i+1)*n].T @ one_vec == alpha[i]*one_vec]

        cons += [cp.sum(alpha) == 1]

        prob = cp.Problem(cp.Minimize(obj), cons)   
        prob.solve()

        return A.value, alpha.value
        
    

class RollingWindowPredictor(CovariancePredictor):
    def __init__(self, R, memory, update_freq) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        """
        self.R = R
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.memory = memory
        self.update_freq = update_freq

        # Update covariance matrix every update_freq days using the last memory days
        Sigma_hats = {}
        i = 0
        if memory == "full":
            self.memory = 21
        for t in range(self.memory+1, self.T):
            if i % self.update_freq == 0:
                if memory == "full":
                    start = 0
                else:
                    start = t-(self.memory+1)
                end = t-1 # Don't include t; to avoid look-ahead bias
                Sigma_hat = np.cov(R.values[start:end], rowvar=False)
            Sigma_hats[R.index[t]] = Sigma_hat
            i += 1  

        self.Sigma_hats = Sigma_hats
        
        # Update variables to start at t=memory
        self.R = self.R.iloc[self.memory+1:, :]
        self.T = self.R.shape[0]

        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())
                


class EwmaPredictor(CovariancePredictor):
    def __init__(self, R, T_half, init_cutoff=0) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        """
        self.T_half = T_half

        realized_covs = get_realized_covs(R.values)
        Sigma_hats = get_ewmas(realized_covs, T_half=self.T_half)

        # Remove first to remove look-ahead bias
        R = R.iloc[1:, :]
        Sigma_hats = Sigma_hats[:-1, :, :]

        R = R.iloc[init_cutoff:, :]
        Sigma_hats = Sigma_hats[init_cutoff:, :, :]

        self.R = R
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.Sigma_hats = {self.R.index[t]: Sigma_hats[t] for t in range(self.T)}

        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())

class AlternatingPredictor(CovariancePredictor):
    """
    An alternating predictor is a predictor that alternates between several predictors
    at each time step the predictor is chosen based on the likelihood of the past likelihood_memory days
    """
    def __init__(self) -> None:
        pass

    def get_optimal_predictors(self, init_cutoff, likelihood_memory, R, inds):
        Sigma_hats = {}
        inds_opt = {}
        for t in trange(init_cutoff+likelihood_memory+1, R.shape[0]):
            # Find bets covariance predictor based on likelihood on past likelihood_memory days
            R_temp = R.iloc[t-(likelihood_memory+1):t-1, :] # Don't include t; avoid look-ahead bias
            log_like = -np.inf; Sigma_hat_best = None; ind_best = None; # Initialize
            for ind in inds:
                Sigma_hats_temp = np.array([self.predictors[ind].Sigma_hats[R.index[i]] for i in range(t-(likelihood_memory+1),t-1)]) # Don't include t
                log_like_temp = np.mean(self.get_log_likelihoods(X = R_temp.values, Sigmas=Sigma_hats_temp))
                if log_like_temp > log_like:
                    log_like = log_like_temp
                    Sigma_hat_best = self.predictors[ind].Sigma_hats[R.index[t]]
                    ind_best = ind
            Sigma_hats[R.index[t]] = Sigma_hat_best
            inds_opt[R.index[t]] = ind_best

        return Sigma_hats, inds_opt

class EnsemblePredictor(CovariancePredictor):
    def __init__(self, R, experts, pred_type="best", likelihood_memory=10) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        param experts: list of CovariancePredictor objects
        param pred_type: "best" or "average"
        param likelihood_memory: number of days to use to compute likelihood when deciding which expert to use\
            only used if pred_type="best"
        """
        self.experts = experts
        self.n = R.shape[1]

        Sigma_hats = {}
        if pred_type == "best":
            for t in trange(likelihood_memory+1, R.shape[0]):
                R_temp = R.iloc[t-(likelihood_memory+1):t-1, :] # Don't include t; avoid look-ahead bias
                log_like = -np.inf; Sigma_hat_best = None
                for expert in experts:
                    Sigma_hats_temp = np.array([expert.Sigma_hats[R.index[i]] for i in range(t-(likelihood_memory+1),t-1)]) # Don't include t
                    log_like_temp = np.mean(self.get_log_likelihoods(X = R_temp.values, Sigmas=Sigma_hats_temp))
                    if log_like_temp > log_like:
                        log_like = log_like_temp
                        Sigma_hat_best = expert.Sigma_hats[R.index[t]]

                Sigma_hats[R.index[t]] = Sigma_hat_best

            R = R.iloc[likelihood_memory+1:, :]
        elif pred_type == "average":
            for t in trange(R.shape[0]):
                Sigma_hat = np.zeros((self.n, self.n))
                for expert in experts:
                    Sigma_hat += expert.Sigma_hats[R.index[t]]
                Sigma_hat /= len(experts)
                Sigma_hats[R.index[t]] = Sigma_hat
        
        self.Sigma_hats = Sigma_hats
        self.R = R
        self.T = self.R.shape[0]
        

        assert list(self.R.index) == list(self.Sigma_hats.keys())




        




class AlternatingIteratedEwmaPredictor(AlternatingPredictor):
    def __init__(self, R, T_halfs, likelihood_memory, init_cutoff=20,\
        lamdas = None) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        T_halfs: list of tuples of T_half=(T_scale, T_full) to use for alternating iterated EWMA
        param likelihood_memory: number of days to use to compute likelihood when deciding which T_half to use
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.init_cutoff = init_cutoff
        self.n = R.shape[1]
        self.T_halfs = T_halfs
        if lamdas is None:
            lamdas = [0 for _ in range(len(T_halfs))]
        self.lamdas = lamdas

        iterated_ewma_predictors = {}
        for i, T_half in enumerate(T_halfs):
            iterated_ewma_predictors[T_half] = IteratedEwmaPredictor(R, T_half[0], T_half[1], init_cutoff, self.lamdas[i]) 
        
        
        self.T_halfs = T_halfs
        self.likelihood_memory = likelihood_memory

        # Remove first to remove look-ahead bias
        R = [*iterated_ewma_predictors.values()][0].R
        for T_half in T_halfs:
            assert list(R.index) == list(iterated_ewma_predictors[T_half].R.index)

        # Get optimal predictors
        self.predictors = iterated_ewma_predictors

        Sigma_hats, T_halfs_opt = self.get_optimal_predictors(init_cutoff, likelihood_memory, R, T_halfs)

        self.R = R.iloc[self.init_cutoff+likelihood_memory+1:, :]
        self.T = self.R.shape[0]
        self.Sigma_hats = Sigma_hats; self.T_halfs_opt = T_halfs_opt
        
        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())
        assert list(self.R.index) == list(self.T_halfs_opt.keys())


class AlternatingEwmaPredictor(AlternatingPredictor):
    def __init__(self, R, T_halfs, likelihood_memory, init_cutoff=0, compute_opt_Sigmas=True) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        T_halfs: list of T_halfs to use for alternating EWMA
        param likelihood_memory: number of days to use to compute likelihood when deciding which T_half to use
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.init_cutoff = init_cutoff
        self.n = R.shape[1]
        self.T_halfs = T_halfs

        ewma_predictors = {}
        for T_half in T_halfs:
            ewma_predictors[T_half] = EwmaPredictor(R, T_half)
        
        
        # Remove first to remove look-ahead bias
        # R = R.iloc[1:, :]
        R = [*ewma_predictors.values()][0].R
        for T_half in T_halfs:
            assert list(R.index) == list(ewma_predictors[T_half].R.index)

        # Get optimal predictors
        self.predictors = ewma_predictors

        

        self.R = R.iloc[self.init_cutoff+likelihood_memory+1:, :]
        self.T = self.R.shape[0]

        Sigma_hats, T_halfs_opt = self.get_optimal_predictors(init_cutoff, likelihood_memory, R, T_halfs)
        self.Sigma_hats = Sigma_hats; self.T_halfs_opt = T_halfs_opt
        
        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())
        assert list(self.R.index) == list(self.T_halfs_opt.keys())
        
        
class IteratedEwmaPredictor(CovariancePredictor):
    def __init__(self, R, T_half_scale, T_half_full, init_cutoff=20, lamda=0) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        param T_half_scale: half-life for scaling
        param T_half_full: half-life for full whitening
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.T_half_scale = T_half_scale
        self.T_half_full = T_half_full
        self.init_cutoff = init_cutoff

        ### Scaling  
        realized_vars = get_realized_vars(R.values)
        V_hats = get_ewmas(realized_vars, T_half=self.T_half_scale)
        V_inv_diags = 1/np.sqrt(np.diagonal(V_hats, axis1=1, axis2=2))
        Ws_V = np.stack([np.diag(V) for V in V_inv_diags]) # Gets L^T s.t. LL^T = V_hat^{-1}
        r_tilde_var = get_r_tildes(R.values, Ws_V)

        # winsorizing ?

        ### Full whitening, outer products in T x n x n tensor
        realized_covs = get_realized_covs(r_tilde_var)

        # moving across first dimenison of Tensor
        Sigma_tilde_hats = get_ewmas(realized_covs, T_half=self.T_half_full)

        # Create correlation matrix
        # refactor: separate function cov2corr
        V = np.sqrt(np.diagonal(Sigma_tilde_hats, axis1=1, axis2=2))
        outer_V = np.array([np.outer(v, v) for v in V])
        corr_tilde_hats = Sigma_tilde_hats / outer_V

        # Remove first 'init_cutoff' to avoid singularity
        corr_tilde_hats = corr_tilde_hats[init_cutoff:]
        Ws_V = Ws_V[init_cutoff:]
        R = R.iloc[init_cutoff:, :]
        V_hats = V_hats[init_cutoff:]


        # Engle 2002 formula
        sqrt_V_hats = np.sqrt(V_hats)
        Sigma_hats = sqrt_V_hats @ corr_tilde_hats @ sqrt_V_hats

        # Remove first asset to avoid look-ahead bias
        R = R.iloc[1:, :]
        Sigma_hats = Sigma_hats[:-1, :, :]

        self.R = R
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.Sigma_hats = {self.R.index[t]: Sigma_hats[t] for t in range(self.T)}
        # wrong place
        # Add regularization
        
        if lamda > 0:
            for key in self.Sigma_hats.keys():
                self.Sigma_hats[key] = self.Sigma_hats[key] + lamda * np.diag(np.diag(self.Sigma_hats[key]))

        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())


class IteratedGarchEwmaPredictor(CovariancePredictor):
    def __init__(self, R, T_half, init_cutoff=20, cheat=False) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        param T_half_scale: half-life for scaling
        param T_half_full: half-life for full whitening
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.T_half = T_half
        self.init_cutoff = init_cutoff

        ### Scaling  
        arch_models = {}
        vols = np.zeros(R.shape)

        if not cheat:
            all_arch_params = get_yearly_garch_params(R, update_freq=1)
            for i in range(R.shape[1]):
                vols_temp = []
                asset = R.columns[i]
                for year in R.index.year.unique():
                    am = arch_model(R.values[:,i], p=1, q=1)
                    fixed_res = am.fix(all_arch_params[asset][year]) 
                    vol_temp = pd.DataFrame(fixed_res.conditional_volatility, index=R.index)
                    vol_temp = vol_temp[vol_temp.index.year==year].values
                    vols_temp += vol_temp.tolist()
                vols[:,i] = np.array(vols_temp).reshape(-1,)

        else:
            for i in range(R.shape[1]):
                arch_models[i] = arch_model(R.iloc[:, i], vol='GARCH', p=1, q=1)
                res = arch_models[i].fit(disp=False)
                vols[:,i] = res.conditional_volatility

        V_hats = get_V_hats(vols)
        Ws_V = get_cholesky(V_hats) 
        r_tilde_var = get_r_tildes(R.values, Ws_V)
        

        ### Full whitening
        realized_covs = get_realized_covs(r_tilde_var)
        Sigma_tilde_hats = get_ewmas(realized_covs, T_half=self.T_half)

        # Remove first 'init_cutoff' to avoid singularity
        Sigma_tilde_hats = Sigma_tilde_hats[init_cutoff:]
        Ws_V = Ws_V[init_cutoff:]
        R = R.iloc[init_cutoff:, :]

        Ws_S = get_cholesky(Sigma_tilde_hats) 

        ### Define covariance matrix
        Ws = [Ws_S[t]@Ws_V[t] for t in range(len(Ws_S))]
        Sigma_inv_hats = [Ws[t].T@Ws[t] for t in range(len(Ws))]
        Sigma_hats = np.array([np.linalg.inv(Sigma_inv_hats[t]) for t in range(len(Sigma_inv_hats))])

        # Remove first asset to remove look-ahead bias
        R = R.iloc[1:, :]
        Sigma_hats = Sigma_hats[:-1, :, :]

        self.R = R
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.Sigma_hats = {self.R.index[t]: Sigma_hats[t] for t in range(self.T)}

        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())

class AlternatingIteratedGarchEwmaPredictor(AlternatingPredictor):
    def __init__(self, R, T_halfs, likelihood_memory, init_cutoff=20) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        T_halfs: list of tuples of T_half=(T_scale, T_full) to use for alternating iterated EWMA
        param likelihood_memory: number of days to use to compute likelihood when deciding which T_half to use
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.init_cutoff = init_cutoff
        self.n = R.shape[1]
        self.T_halfs = T_halfs
        self.likelihood_memory = likelihood_memory


        it_garch_ewma_predictors = {}
        for i in trange(len(T_halfs)):
            T_half = T_halfs[i]
            it_garch_ewma_predictors[T_half] = IteratedGarchEwmaPredictor(R, T_half, init_cutoff) 
        
        

        # Remove first to remove look-ahead bias
        R = [*it_garch_ewma_predictors.values()][0].R
        for T_half in T_halfs:
            assert list(R.index) == list(it_garch_ewma_predictors[T_half].R.index)

        # Get optimal predictors
        self.predictors = it_garch_ewma_predictors

        Sigma_hats, T_halfs_opt = self.get_optimal_predictors(init_cutoff, likelihood_memory, R, T_halfs)

        self.R = R.iloc[self.init_cutoff+likelihood_memory+1:, :]
        self.T = self.R.shape[0]
        self.Sigma_hats = Sigma_hats; self.T_halfs_opt = T_halfs_opt
        
        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())
        assert list(self.R.index) == list(self.T_halfs_opt.keys())
       

        
class TrippleIteratedEwmaPredictor(CovariancePredictor):
    def __init__(self, R, T_half_scale, T_half_std, T_half_full, init_cutoff=20) -> None:
        """
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        param T_half_scale: half-life for scaling
        param T_half_std: half-life for standerdizing
        param T_half_full: half-life for full whitening
        param init_cutoff: number of days to remove from beginning to avoid singularity
        """
        self.T_half_scale = T_half_scale
        self.T_half_std = T_half_std
        self.T_half_full = T_half_full
        self.init_cutoff = init_cutoff

        ### Scaling
        realized_sizes = get_realized_sizes(R.values) 
        size_hats = get_ewmas(realized_sizes, T_half_scale)
        Ws_size = (1/np.sqrt(size_hats)).reshape(-1,1)
        r_tildes_size = Ws_size * R.values

        ### Standerdizing  
        realized_vars = get_realized_vars(r_tildes_size)
        V_hats = get_ewmas(realized_vars, T_half=self.T_half_scale)
        Ws_V = get_cholesky(V_hats) 
        r_tilde_var = get_r_tildes(r_tildes_size, Ws_V)

        ### Full whitening
        realized_covs = get_realized_covs(r_tilde_var)
        Sigma_tilde_hats = get_ewmas(realized_covs, T_half=self.T_half_full)

        # Remove first 'init_cutoff' to avoid singularity
        Sigma_tilde_hats = Sigma_tilde_hats[init_cutoff:]
        Ws_V = Ws_V[init_cutoff:]
        R = R.iloc[init_cutoff:, :]

        Ws_S = get_cholesky(Sigma_tilde_hats) 

        ### Define covariance matrix
        Ws = [Ws_size[t]*Ws_S[t]@Ws_V[t] for t in range(len(Ws_S))]
        Sigma_inv_hats = [Ws[t].T@Ws[t] for t in range(len(Ws))]
        Sigma_hats = np.array([np.linalg.inv(Sigma_inv_hats[t]) for t in range(len(Sigma_inv_hats))])

        # Remove first asset to remove look-ahead bias
        R = R.iloc[1:, :]
        Sigma_hats = Sigma_hats[:-1, :, :]

        self.R = R
        self.T = R.shape[0]
        self.n = R.shape[1]
        self.Sigma_hats = {self.R.index[t]: Sigma_hats[t] for t in range(self.T)}

        # Make sure that times are consistent between R and Sigma_hats
        assert list(self.R.index) == list(self.Sigma_hats.keys())
       

class AlternatingWeightedPredictor(AlternatingPredictor):

    def __init__(self, alternating_type, R, T_halfs=None, likelihood_memory=10, init_cutoff=0, const_theta=None, const_L=None, experts=None, ignore_dpp=False,\
        turnover_cons=None, lamdas=None) -> None:
        """
        param alternating_type: "ewma", "iterated_ewma" or "garch_ewma"
        param R: pandas dataframe where rows are vector of asset returns for t=1,2,...
            R has shape (T, n) where T is the number of days and n is the number of assets
        param T_halfs: list of half-lives for ewma or iterated ewma predictors
        param likelihood_memory: number of days to look back when computing likelihood
        param init_cutoff: number of days to remove from beginning to avoid singularity
        param const_theta: constant theta in lookback period for log-likelihood
        param const_L: constant L in lookback period for log-likelihood
        param experts: list of experts to use in alternating predictor
        param ignore_dpp: if True, ignore the DPP when solving convex problem
        param turnover_cons: weight to put on turnover constraint, if None, no turnover constraint
        param lamdas: regularization weight. Currently only applicable to alternating_type="iterated_ewma"
        """
        self.likelihood_memory = likelihood_memory
        self.init_cutoff = init_cutoff
        self.n = R.shape[1]
        self.experts = experts
        self.alternating_type = alternating_type
        self.ignore_dpp = ignore_dpp
        self.turnover_cons = turnover_cons
        self.lamdas = lamdas

        if alternating_type == "ewma":
            self.T_halfs = T_halfs

            ewma_predictors = {}
            for T_half in T_halfs:
                ewma_predictors[T_half] = EwmaPredictor(R, T_half, init_cutoff=init_cutoff)
            
            R = ewma_predictors[T_halfs[0]].R
            self.R = R
            for T_half in T_halfs:
                assert list(self.R.index) == list(ewma_predictors[T_half].R.index)

            self.predictors = ewma_predictors
            self.T = self.R.shape[0]

        elif alternating_type == "iterated_ewma":
            self.T_halfs = T_halfs

            iterated_ewma_predictors = {}
            for i, T_half in enumerate(T_halfs):
                iterated_ewma_predictors[T_half] = IteratedEwmaPredictor(R, T_half[0], T_half[1], init_cutoff=init_cutoff, lamda=lamdas[i])
            
            R = iterated_ewma_predictors[T_halfs[0]].R
            self.R = R
            for T_half in T_halfs:
                assert list(self.R.index) == list(iterated_ewma_predictors[T_half].R.index)

            self.predictors = iterated_ewma_predictors
            self.T = self.R.shape[0]

        elif alternating_type == "iterated_garch_ewma":
            self.T_halfs = T_halfs

            it_garch_ewma_predictors = {}
            for i in trange(len(T_halfs)):
                T_half = T_halfs[i]
                it_garch_ewma_predictors[T_half] = IteratedGarchEwmaPredictor(R, T_half, init_cutoff=init_cutoff) 

            R = it_garch_ewma_predictors[T_halfs[0]].R
            self.R = R
            for T_half in T_halfs:
                assert list(self.R.index) == list(it_garch_ewma_predictors[T_half].R.index)

            self.predictors = it_garch_ewma_predictors
            self.T = self.R.shape[0]

        if alternating_type is None:
            self.R = R
            self.T = R.shape[0]
            self.predictors = dict()

        if experts is not None:
            for i, expert in enumerate(experts):
                # Add expert to predictors
                expert.R = self.R
                expert.T = self.T
                expert.Sigma_hats = {t: expert.Sigma_hats[t] for t in self.R.index}
                assert list(expert.R.index) == list(expert.Sigma_hats.keys())
                assert list(expert.R.index) == list(self.R.index)

                self.predictors[f"expert_{i}"] = expert

        
        # Get precison matrices and Cholesky factors
        for predictor in self.predictors.values():
            predictor.get_Lt_hats() # This also gets Theta_hats

        # Update model's parameters as well
        if alternating_type is not None:
            R = self.predictors[T_halfs[0]].R
            self.R = R
            for T_half in T_halfs:
                assert list(self.R.index) == list(self.predictors[T_half].R.index)
            self.T = self.R.shape[0]

        self.get_Sigma_hats(const_theta, const_L)



    def get_Sigma_hats(self, const_theta=None, const_L=None):
        Sigma_hats = {}
        M = len(self.predictors)

        ws = get_opt_weights_new(self, likelihood_window=self.likelihood_memory, w_init=None, const_theta=const_theta, const_L=const_L,\
             turnover_cons=self.turnover_cons)

        Theta_hats_for_all_t = {}
        Lt_hats_for_all_t = {}
        Sigma_hats_for_all_t = {}
        for i in range(self.likelihood_memory + 1, self.T):
            t = self.R.index[i]

            Theta_hats_t = np.array([predictor.Theta_hats[t] for predictor in self.predictors.values()]) # available "experts" at time t
            Theta_hats_for_all_t[t] = Theta_hats_t

            Lt_hats_t = np.array([predictor.Lt_hats[t] for predictor in self.predictors.values()]) # available "experts" at time t
            Lt_hats_for_all_t[t] = Lt_hats_t


            Sigma_hat_t = np.array([predictor.Sigma_hats[t] for predictor in self.predictors.values()])
            Sigma_hats_for_all_t[t] = Sigma_hat_t


        # Store in Sigma_hats and weights dictionaries
        for i in range(self.likelihood_memory + 1, self.T):

            t = self.R.index[i]
            wt = ws.loc[t].values

            if const_L is None:
                Theta_hats_t = Theta_hats_for_all_t[t]
                Theta_hat_t = np.zeros((self.n, self.n))
                for j in range(M):
                    Theta_hat_t += wt[j]*Theta_hats_t[j]
                
            elif const_theta is None:
                Lt_hats_t = Lt_hats_for_all_t[t]

                Lt_hat_t = np.zeros((self.n, self.n))
                for j in range(M):
                    Lt_hat_t += wt[j]*Lt_hats_t[j]
                Theta_hat_t = Lt_hat_t.T@Lt_hat_t

            Sigma_hats[t] = np.linalg.solve(Theta_hat_t, np.eye(self.n))

        
        self.R = self.R.iloc[self.likelihood_memory+1:, :]
        self.T = self.R.shape[0]
        self.Sigma_hats = Sigma_hats
        self.weights = ws


