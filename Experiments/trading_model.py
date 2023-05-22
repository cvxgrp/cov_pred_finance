import cvxpy as cp
import numpy as np
import multiprocessing as mp
import pandas as pd
from collections import namedtuple

def _get_L_inv(covariance):
    Theta = np.linalg.inv(covariance)
    return np.linalg.inv(np.linalg.cholesky(Theta))

metrics = namedtuple('metrics', ['mean_return', 'risk', 'sharpe', 'drawdown'])

class Trader():
    def __init__(self, R, Sigma_hats, r_hats=None):
        """
        param R: nxT pandas dataframe of T returns from n assets; index is date.
        param Sigma_hats: dictionary of causal covariance matrices; keys are
        dates; values are n x n pandas dataframes. They are causal in the sense that
        Sigma_hat[time] does NOT depend on returns[time]
        param r_hats: dictionary of causal expected returns, i.e., r_hat[time]
        does NOT depend on returns[time]


        """
        self.R = R
        self.Sigma_hats = Sigma_hats
        self.r_hats = r_hats
        self.diluted = False

        assert list(self.R.index) == list(self.Sigma_hats.keys())
        sigma_hats = []
        for t in self.R.index:
            Sigma_hat_t = self.Sigma_hats[t].values
            sigma_hat_t = np.sqrt(np.diag(Sigma_hat_t))
            sigma_hats.append(sigma_hat_t)

        if r_hats is not None:
            assert list(self.R.index) == list(self.r_hats.index)
            assert list(self.R.columns) == list(self.r_hats.columns)

        # Generate dataframe of standard deviations
        sigma_hats = np.array(sigma_hats)
        self.sigma_hats = pd.DataFrame(sigma_hats, index=self.R.index,\
                columns=self.R.columns)
        assert list(self.R.index) == list(self.sigma_hats.index)
        assert list(self.R.columns) == list(self.sigma_hats.columns)

        # Generate Choleksy factors
        self.L_inv_hats = {}
        for time in R.index:
            self.L_inv_hats[time] = _get_L_inv(pd.DataFrame(self.Sigma_hats[time].values, index = self.assets, columns = self.assets))

    @property
    def assets(self):
        return self.R.columns

    @property
    def n(self):
        return self.R.shape[1]
    
    @property
    def T(self):
        return self.R.shape[0]


    def solve_min_risk(self, prob, w, L_inv_param, L_inv,\
        sigma_param=None, sigma=None):
        """
        Solves the minimum risk problem for a given whiteners. 

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Lt_inv_param: cvxpy parameter parameter
        param Lt: whitener
        """
        L_inv_param.value = L_inv
        if sigma_param is not None and sigma is not None:
            sigma_param.value = sigma
        if self.C_speedup:
            prob.register_solve('cpg', cpg_solve) # TODO: remove?
            prob.solve(method='cpg', updated_params=["L_inv_param"])
        else:
            prob.solve(verbose=False)

        return w.value, prob.objective.value

    def solve_risk_parity(self, prob, w, L_inv_param, L_inv):
        """
        Solves the risk parity problem for a given covariance matrix. 

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Sigma_t_param: cvxpy parameter parameter
        param Sigma_t: covariance matrix
        """
        L_inv_param.value = L_inv
        prob.solve()
        w_normalized = w.value / np.sum(w.value)
     
        return w_normalized

    def solve_mean_variance(self, prob, w, L_inv_param, r_hat_param, L_inv, r_hat):
        """
        Solves the mean variance problem. 

        param prob: cvxpy problem object
        param w: cvxpy variable object
        param Sigma_t_param: cvxpy parameter parameter
        param Sigma_t: covariance matrix
        """

        L_inv_param.value = L_inv
        r_hat_param.value = np.vstack([r_hat.reshape(-1,1), 0])
        r_hat_param = np.zeros(r_hat_param.shape)
        prob.solve()
     
        return w.value, prob.objective.value

    def get_vol_cont_w(self, ws, obj, sigma_des):
        """
        Computes the weights of the volatility controlled portfolio. 

        param ws: list of weights of the minimum risk portfolios
        param obj: list of objective values of the minimum risk portfolios
        param sigma_des: desired volatility of the portfolio
        """
        sigma_des = sigma_des * np.sqrt(self.adjust_factor)
        sigma_des = sigma_des / np.sqrt(252)
        # sigma_r = np.sqrt(obj)

        sigma_r = obj
        w_cash = (1 - sigma_des / sigma_r).reshape(-1,1)
        ws = (1-w_cash) * ws

        return np.hstack([ws, w_cash])

    def solve_max_diverse(self, prob, z, L_inv_param, sigma_param,\
         L_inv, sigma):
        """
        Solves the maximum diversification optimization problem.

        param prob: cvxpy problem object
        param z: cvxpy variable object 
        param L_inv_param: cvxpy parameter parameter
        param L_inv: inverse whitener
        param sigma_param: cvxpy parameter parameter
        param sigma: diagonal of the covariance matrix

        """
        L_inv_param.value = L_inv
        sigma_param.value = sigma
        prob.solve(solver="ECOS")

        w = z.value / np.sum(z.value)
        return w

    def solve_max_sharpe(self, prob, z, L_inv_param, r_hat_param,\
         L_inv, r_hat):
        """
        Solves the maximum diversification optimization problem.

        param prob: cvxpy problem object
        param z: cvxpy variable object 
        param L_inv_param: cvxpy parameter parameter
        param L_inv: inverse whitener
        param r_hat_param: cvxpy parameter parameter
        param r_hat: return

        """
        L_inv_param.value = L_inv.T
        r_hat_param.value = r_hat
        prob.solve(solver="ECOS")


        # call assertion error if fails
        assert prob.status == "optimal", r_hat

        # print(prob.status)

        w = z.value / np.sum(z.value)
        return w

     

    def dilute_with_cash(self, sigma_des=0.1):
        """
        Dilutes the portfolio with cash to achieve a desired volatility.
        """
        if self.diluted:
            print("Already trades cash...")
            return None
        ws_new = []
        for i, t in enumerate(self.R.index):
            Sigma_hat_t = self.Sigma_hats[t].values
            w_t = self.ws[i].reshape(-1,1)
            sigma_hat = np.sqrt(w_t[:self.n].T @ Sigma_hat_t @ w_t[:self.n])
            
            theta = np.sqrt(1/252) * (sigma_des / sigma_hat)
            w_t_new = np.vstack([theta * w_t, 1 - theta])
            ws_new.append(w_t_new.flatten())
        
        self.ws_diluted = np.array(ws_new)
        self.diluted = True

    def get_risk_adj_returns(self, sigma_des=0.1):
        """
        Scales returns to adjusted risk level sigma_des. 
        """
        a = sigma_des / (np.sqrt(252) * np.std(self.rets))

        self.rets_adjusted = a * self.rets

    def get_risk_adj_portfolio_growth(self, sigma_des=0.1):
        """
        Computes the risk adjusted portfolio growth from the daily (total)  returns.
        """
        self.get_risk_adj_returns(sigma_des=sigma_des)
        Vs = [np.array([1])]
        for t, r_t in enumerate(self.rets_adjusted):
            Vs.append(Vs[t]*(1+r_t/self.adjust_factor))
        self.Vs_adjusted = np.array(Vs[1:])

            


    def backtest(self, portfolio_type="min_risk", cons=[], sigma_des=None, adjust_factor=1,\
         additonal_cons={"short_lim":1.6, "upper_bound":0.15,\
             "lower_bound":-0.1}, C_speedup=False, kappa=None):
        """
        param portfolio_type: type of portfolio to backtest. Options are "min_risk", "vol_cont", "risk_parity", "mean_variance".
        param cons: list of constraints to impose on the optimization problem. 
        """
        self.portfolio_type = portfolio_type
        self.adjust_factor = adjust_factor
        self.additonal_cons = additonal_cons
        self.C_speedup = C_speedup

        if portfolio_type == "eq_weighted":
            ws = np.ones((self.T, self.n)) / self.n
            

        if portfolio_type == "min_risk" or portfolio_type == "vol_cont"\
             or portfolio_type == "robust_min_risk":
            # get the minimum risk portfolio
            w = cp.Variable((self.n,1))
            L_inv_param = cp.Parameter((self.n, self.n))
            # risk = cp.norm(L_inv_param @ w, 2)

            if portfolio_type == "robust_min_risk":
                assert kappa != None
                sigma_hat_param = cp.Parameter((self.n, 1), nonneg=True)
                risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))\
                    + kappa*cp.square(sigma_hat_param.T@cp.abs(w))
            else:
                risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))

            if self.C_speedup and portfolio_type != "robust_min_risk":
                print("Using prespecified CVXPYgen formulation... Ignoring new constraints")
                n = 49
                w = cp.Variable((n,1), name="w")
                L_inv_param = cp.Parameter((n, n), name="L_inv_param")
                # risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))
                risk = cp.norm(L_inv_param @ w, 2)
                ones = np.ones((n,1))
                cons=[ones.T@w<=1, ones.T@w>=1]
                additonal_cons={"short_lim":1.6, "upper_bound":0.15,\
                            "lower_bound":-0.1}
                if [*additonal_cons.keys()]:
                    if "short_lim" in additonal_cons.keys():
                        cons += [cp.norm(w, 1) <= additonal_cons["short_lim"]]
                    if "upper_bound" in additonal_cons.keys():
                        cons += [w <= additonal_cons["upper_bound"]]
                    if "lower_bound" in additonal_cons.keys():
                        cons += [w >= additonal_cons["lower_bound"]]

                prob = cp.Problem(cp.Minimize(risk), cons)
                prob.register_solve('cpg', cpg_solve)
            else:
                # add constraints
                cons += [cp.sum(w)==1]
                if [*additonal_cons.keys()]:
                    # print("Adding additional constraints")
                    if "short_lim" in additonal_cons.keys():
                        cons += [cp.norm(w, 1) <= additonal_cons["short_lim"]]
                    if "upper_bound" in additonal_cons.keys():
                        cons += [w <= additonal_cons["upper_bound"]]
                    if "lower_bound" in additonal_cons.keys():
                        cons += [w >= additonal_cons["lower_bound"]]
                    # cons += [w >= -0.15]
                    # cons += [w <= 0.3]
                    # cons += [w >= 0]


                prob = cp.Problem(cp.Minimize(risk), cons)

                # if self.portfolio_type == "robust_min_risk":
                # TODO: always define this? Not used for min_risk!
                all_sigma_hats = [self.sigma_hats.values[t].reshape(-1,1)\
                for t in range(self.T)]

                # Solve problem with random inputs once to speed up later solves
                L_inv_param.value = [*self.L_inv_hats.values()][0]
                if portfolio_type == "robust_min_risk":
                    sigma_hat_param.value = all_sigma_hats[0]
                prob.solve()

            # solve the problem for each date
            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            if self.portfolio_type == "robust_min_risk":
                all_sigma_hat_param = [sigma_hat_param for _ in range(self.T)]
            else:
                all_sigma_hat_param = [None for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

  
            pool = mp.Pool()
            ws_and_obj = pool.starmap(self.solve_min_risk, zip(all_prob, all_w,\
                 all_L_inv_param, [*self.L_inv_hats.values(),\
                    all_sigma_hat_param, all_sigma_hats]))
            pool.close()
            pool.join()

            # get the weights and objective values
            ws, obj = zip(*ws_and_obj)
            ws = np.array(ws)
            obj = np.array(obj)
            self.obj = np.array(obj)

            if portfolio_type == "vol_cont":
                assert sigma_des is not None
                self.diluted = True # Trades cash
                ws = self.get_vol_cont_w(ws, obj, sigma_des)

        elif portfolio_type == "mean_variance":
            assert self.r_hats is not None
            assert sigma_des is not None
            sigma_des = sigma_des * self.adjust_factor
 
            w = cp.Variable((self.n+1,1)) # Last asset is cash
            L_inv_param = cp.Parameter((self.n, self.n))
            r_hat_param = cp.Parameter((self.n+1,1)) # Last asset is cash

            # risk = cp.norm(L_inv_param @ w[:-1], 2)
            risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w[:-1].T), axis=1))
            ret = r_hat_param.T @ w

            # add constraints
            cons += [cp.sum(w)==1]
            cons += [risk <= sigma_des / np.sqrt(252)]

            if [*additonal_cons.keys()]:
                if "short_lim" in additonal_cons.keys():
                    cons += [cp.norm(w[:-1], 1) <= additonal_cons["short_lim"]]
                if "upper_bound" in additonal_cons.keys():
                    cons += [w[:-1] <= additonal_cons["upper_bound"]]
                if "lower_bound" in additonal_cons.keys():
                    cons += [w[:-1] >= additonal_cons["lower_bound"]]
                # cons += [w >= 0]

            prob = cp.Problem(cp.Maximize(ret), cons)

            # Solve problem with random inputs once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            r_hat_param.value = np.vstack([self.r_hats.values[0].reshape(-1,1), 0])
            prob.solve()

            # solve the problem for each date
            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_r_hat_param = [r_hat_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

  
            pool = mp.Pool()
            ws_and_obj = pool.starmap(self.solve_mean_variance, zip(all_prob, all_w,\
                 all_L_inv_param, all_r_hat_param, [*self.L_inv_hats.values()], self.r_hats.values))
            pool.close()
            pool.join()

            # get the weights and objective values
            ws, obj = zip(*ws_and_obj)
            ws = np.array(ws)
            obj = np.array(obj)
            self.obj = np.array(obj)

        elif portfolio_type=="max_diverse":
            L_inv_param = cp.Parameter((self.n, self.n))
            sigma_param = cp.Parameter((self.n, 1))

            z = cp.Variable((self.n,1)) # Change of variables solution
            obj = cp.norm2(cp.sum(cp.multiply(L_inv_param, z.T), axis=1))

            M = additonal_cons["upper_bound"]
            cons = [z >= 0, z <= M*cp.sum(z), sigma_param.T@z == 1]
            prob = cp.Problem(cp.Minimize(obj), cons)

            # Solve problem once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            sigma_param.value = np.ones((self.n,1))
            prob.solve(solver="ECOS")  

            all_z = [z for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_sigma_param = [sigma_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            pool = mp.Pool()
            all_sigma_hats = [self.sigma_hats.values[t].reshape(-1,1)\
                 for t in range(self.T)]
            ws = pool.starmap(self.solve_max_diverse, zip(all_prob, all_z, all_L_inv_param, all_sigma_param, \
                [*self.L_inv_hats.values()], all_sigma_hats))
            pool.close()
            pool.join() 

            ws = np.array(ws)

        elif portfolio_type=="risk_parity":
            w = cp.Variable((self.n,1))
            L_inv_param = cp.Parameter((self.n, self.n))

            # risk = cp.norm(L_inv_param @ w, 2)
            risk = cp.norm2(cp.sum(cp.multiply(L_inv_param, w.T), axis=1))
            prob = cp.Problem(cp.Minimize(1/2*risk - cp.log(cp.geo_mean(w))))

        
            # Solve once for speedup later
            # Solve problem with random inputs once to speed up later solves
            L_inv_param.value = [*self.L_inv_hats.values()][0]
            prob.solve()


            all_w = [w for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]

            pool = mp.Pool()
            ws = pool.starmap(self.solve_risk_parity, zip(all_prob, all_w, all_L_inv_param, [*self.L_inv_hats.values()]))
            pool.close()
            pool.join()
            ws = np.array(ws)

        elif portfolio_type=="max_sharpe":
            assert self.r_hats is not None

            L_inv_param = cp.Parameter((self.n, self.n))
            r_hat_param = cp.Parameter((self.n, 1))

            z = cp.Variable((self.n,1)) # Change of variables solution
            obj = cp.norm2(cp.sum(cp.multiply(L_inv_param, z.T), axis=1))

            M = additonal_cons["upper_bound"]
            # cons = [z >= 0, z <= M*cp.sum(z), r_hat_param.T@z == 1]
            cons = [z >= 0, r_hat_param.T@z == 1]

            prob = cp.Problem(cp.Minimize(obj), cons)

            # Solve problem once to speed up later solves
            L_inv_param.value = [*self.Lt_inv_hats.values()][0].T
            r_hat_param.value = np.ones((self.n,1))
            prob.solve(solver="ECOS")  


            all_z = [z for _ in range(self.T)]
            all_L_inv_param = [L_inv_param for _ in range(self.T)]
            all_r_hat_param = [r_hat_param for _ in range(self.T)]
            all_prob = [prob for _ in range(self.T)]
            all_r_hats = [self.r_hats.values[t].reshape(-1,1)\
                 for t in range(self.T)]



            pool = mp.Pool()
            
            ws = pool.starmap(self.solve_max_sharpe, zip(all_prob, all_z, all_L_inv_param, all_r_hat_param, \
                [*self.Lt_inv_hats.values()], all_r_hats))
            pool.close()
            pool.join() 

            ws = np.array(ws)


        self.ws = ws.reshape(self.T, -1)

    def get_total_returns(self, diluted_with_cash=False, sigma_des=None, rf=None):
        """
        Computes total daily returns of the portfolio from the weights and indivudal asset returns.
        """
        rets = []
        for t in range(self.T):
            w_t = self.ws[t]
            r_t = self.R.iloc[t].values.reshape(-1,1)
            if self.portfolio_type == "vol_cont" or self.portfolio_type == "mean_variance" or self.portfolio_type == "mean_variance2" or \
                diluted_with_cash:
                if diluted_with_cash:
                    if not self.diluted == True:
                        assert sigma_des is not None
                        self.dilute_with_cash(sigma_des)
                    w_t = self.ws_diluted[t]
                if rf is None:
                    r_t = np.vstack([r_t, 0]) # Add cash return (last)
                else:
                    rf_t = rf.iloc[t]
                    r_t = np.vstack([r_t, rf_t])
            rets.append(np.dot(w_t, r_t))
        self.rets = np.array(rets)

    def get_portfolio_growth(self):
        """
        Computes the portfolio growth from the daily (total) returns.
        Note: run get_total_returns() first.
        """
        Vs = [np.array([1])]
        for t, r_t in enumerate(self.rets):
            Vs.append(Vs[t]*(1+r_t/self.adjust_factor))
        self.Vs = np.array(Vs[1:])

    def compute_max_drawdown(self):
        """
        Computes the maximum drawdown of the portfolio.
        """
        max_dd = 0
        peak = self.Vs[0]
        for V in self.Vs:
            if V > peak:
                peak = V
            dd = (peak - V) / peak
            if dd > max_dd:
                max_dd = dd
        return float(max_dd)

    def get_metrics(self, diluted_with_cash=False, sigma_des=0.1, rf=None,\
         excess=False):
        """
        param adjust_factor: factor to results by, \
            e.g., if we have a model for 100*r_daily, the volatility\
                should be adjusted by 1/sqrt(100)


        Computes the avg return, stdev, sharpe ratio, and max drawdown of the portfolio.
        Note: Run backtest first.
        """
        self.get_total_returns(diluted_with_cash=diluted_with_cash, sigma_des=sigma_des, rf=rf)
        self.get_portfolio_growth()
        if rf is not None:
            rf = rf.values.reshape(-1,1)
        if excess:
            assert rf is not None
            excess_returns = self.rets/self.adjust_factor - rf
            mean_return = np.mean(excess_returns) * 252
            stdev = np.std(self.rets) * np.sqrt(252)
            # mean_return = (np.mean(self.rets/self.adjust_factor) - np.mean(rf))* 252 
            # stdev = np.std(self.rets / self.adjust_factor-rf) * np.sqrt(252) 
        else:
            mean_return = np.mean(self.rets/self.adjust_factor) * 252 
            stdev = np.std(self.rets / self.adjust_factor) * np.sqrt(252) 
        sharpe_ratio = mean_return / stdev

        return metrics(mean_return, stdev, sharpe_ratio, self.compute_max_drawdown())

        # print(f"Mean annual return: {mean_return:.2%}")
        # print(f"Annual risk: {stdev:.2%}")
        # print(f"Sharpe ratio: {sharpe_ratio:.3}")
        # print(f"Maximum drawdown: {self.compute_max_drawdown():.2%}")









