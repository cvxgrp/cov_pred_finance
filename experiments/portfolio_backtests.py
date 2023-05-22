from abc import ABC, abstractmethod
from tqdm import trange
import warnings
import pandas as pd

# Mute specific warning
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")

from experiments.trading_model import *



def _get_causal(returns, Sigmas, start_date, end_date, mus = None):
    """
    Returns returns and Sigmas with keys in the interval [start_date, end_date]
    """
    returns_temp = returns.loc[start_date:end_date].shift(-1).dropna()
    Sigmas_temp = {time: Sigmas[time] for time in returns_temp.index}

    if mus is not None:
        mus_temp = pd.DataFrame({time: mus.loc[time] for time in returns_temp.index}).T
        return returns_temp, Sigmas_temp, mus_temp

    return returns_temp, Sigmas_temp


def _create_table_helper(metrics):
    print("\\begin{tabular}{lcccc}")
    print("   \\toprule")
    print("   {Predictor} & {Return} & {Risk} & {Sharpe} & {Drawdown} \\\\")
    print("   \\midrule")

    for name, metric in metrics.items():
        if name != "PRESCIENT":
            print("   {} & {:.1f}\% & {:.1f}\% & {:.1f} & {:.0f}\% \\\\".format(
                name, metric.mean_return * 100, metric.risk * 100,
                metric.sharpe, metric.drawdown * 100))
    print("   \\hline")
    metric = metrics["PRESCIENT"]
    print("   {} & {:.1f}\% & {:.1f}\% & {:.1f} & {:.0f}\% \\\\".format(
                name, metric.mean_return * 100, metric.risk * 100,
                metric.sharpe, metric.drawdown * 100))
    print("   \\bottomrule")
    print("\\end{tabular}")
    
def create_table(traders, sigma_tar, rf, excess):
    """
    param traders: dict of Trader Class objects
    param sigma_tar: target volatility
    param rf: risk free rate
    param excess: True if excess returns is used in computation of Sharpe ratio etc., False otherwise
    """
    metrics = {}
    for name, trader in traders.items():
        if sigma_tar:
            metrics[name] = trader.get_metrics(diluted_with_cash=True, sigma_des=sigma_tar, rf=rf,\
                excess=excess)
        else: # TODO: for now this means already trades cash
            metrics[name] = trader.get_metrics(diluted_with_cash=False, rf=rf,\
            excess=excess)
    _create_table_helper(metrics)



class PortfolioBacktest(ABC):
    def __init__(self, returns, cov_predictors, names, mean_predictors=None, start_date=None, end_date=None):
        """
        param returns: pd.DataFrame with returns
        param cov_predictors: list of covariance predictors
        param names: list of names of the predictors
        param mean_predictors: list of mean predictors
        param start_date: start date of the backtest
        param end_date: end date of the backtest

        Note for cov_predictor in cov_predictors: cov_predictor[time] is the
        covariance predictor for time+1, i.e., cov_predictor[time] used
        information up to and including returns[time] (the Portfolio backtest
        Class takes care of the shifting) 
        """
        for cov_predictor in cov_predictors:
            assert set(cov_predictor.keys()).issubset(set(returns.index))
        if mean_predictors is not None:
            for mean_predictor in mean_predictors:
                assert set(cov_predictor.keys()).issubset(set(mean_predictor.index))
        assert start_date is None or start_date in list(cov_predictor.keys())
        assert end_date is None or end_date in list(cov_predictor.keys())


        self.returns = returns
        self.cov_predictors = cov_predictors
        self.mean_predictors = mean_predictors
        self.names = names
        self.start_date = start_date or list(cov_predictor.keys())[0]
        self.end_date = end_date or list(cov_predictor.keys())[-1]
       
    @abstractmethod
    def backtest(self):
        pass

class EqWeighted(PortfolioBacktest):
    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(returns, cov_predictors, names, start_date=start_date, end_date=end_date)

    def backtest(self):
        adjust_factor = 1
        traders_eq_w = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(self.returns, Sigma_hats, self.start_date, self.end_date)

            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(portfolio_type="eq_weighted", adjust_factor=adjust_factor) 

            traders_eq_w[self.names[i]] = trader

        return traders_eq_w

class MinRisk(PortfolioBacktest):

    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(returns, cov_predictors, names, start_date=start_date, end_date=end_date)

    def backtest(self, additonal_cons):
        adjust_factor = 1
        traders_min_risk = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(self.returns, Sigma_hats, self.start_date, self.end_date)
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(portfolio_type="min_risk", adjust_factor=adjust_factor, additonal_cons=additonal_cons)
            traders_min_risk[self.names[i]] = trader

        return traders_min_risk

class MaxDiverse(PortfolioBacktest):

    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(returns, cov_predictors, names, start_date=start_date, end_date=end_date)

    def backtest(self, additonal_cons):
        adjust_factor = 1
        traders_max_diverse = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(self.returns, Sigma_hats, self.start_date, self.end_date)
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(portfolio_type="max_diverse", adjust_factor=adjust_factor, additonal_cons=additonal_cons)
            traders_max_diverse[self.names[i]] = trader

        return traders_max_diverse

class RiskParity(PortfolioBacktest):

    def __init__(self, returns, cov_predictors, names, start_date=None, end_date=None):
        super().__init__(returns, cov_predictors, names, start_date=start_date, end_date=end_date)

    def backtest(self):
        adjust_factor = 1
        traders_risk_par = {}
        for i in trange(len(self.cov_predictors)):
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp = _get_causal(self.returns, Sigma_hats, self.start_date, self.end_date)
            trader = Trader(returns_temp, Sigmas_temp)
            trader.backtest(portfolio_type="risk_parity", adjust_factor=adjust_factor)
            traders_risk_par[self.names[i]] = trader

        return traders_risk_par


class MeanVariance(PortfolioBacktest):

    def __init__(self, returns, cov_predictors, names, mean_predictors, start_date=None, end_date=None):
        super().__init__(returns, cov_predictors, names, mean_predictors=mean_predictors, start_date=start_date, end_date=end_date)

    def backtest(self, additonal_cons, sigma_tar):
        adjust_factor = 1
        traders_mean_var = {}
        for i in trange(len(self.cov_predictors)):
            mu_hats = self.mean_predictors[i]
            Sigma_hats = self.cov_predictors[i]
            returns_temp, Sigmas_temp, mus_temp = _get_causal(self.returns, Sigma_hats, self.start_date, self.end_date, mus=mu_hats)
            trader = Trader(returns_temp, Sigmas_temp, r_hats=mus_temp)
            trader.backtest(portfolio_type="mean_variance", adjust_factor=adjust_factor, additonal_cons=additonal_cons, sigma_des=sigma_tar)
            traders_mean_var[self.names[i]] = trader

        return traders_mean_var




