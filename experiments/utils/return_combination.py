import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import trange


def feature_engineer(return_predictions):
    """
    param return_predictions: pandas DataFrame of return predictions

    Splits each row into positive and negative parts
    """
    return_predictions_pos = return_predictions.copy()
    return_predictions_neg = return_predictions.copy()
    return_predictions_pos[return_predictions_pos < 0] = 0
    return_predictions_neg[return_predictions_neg > 0] = 0
    return pd.concat([return_predictions_pos, return_predictions_neg], axis=1)


def tune_alpha(returns, return_predictors, alphas, memory=1000):
    correlations = []
    rmses = []
    all_weights_list = []
    for alpha in alphas:
        combinator = ReturnCombinator()
        # combinator.build(250, 7, alpha)
        # returns_hats, all_weights = combinator.fit(returns.iloc[:,0], returns_predictions)

        returns_hats, all_weights = combinator.fit(
            pd.DataFrame(returns), return_predictors, memory, alpha
        )

        correlations.append(
            np.corrcoef(
                returns_hats.values.flatten(),
                returns.loc[returns_hats.index].values.flatten(),
            )[0, 1]
        )
        rmses.append(
            np.sqrt(
                np.mean(
                    (
                        returns_hats.values.flatten()
                        - returns.loc[returns_hats.index].values.flatten()
                    )
                    ** 2
                )
            )
        )
        all_weights_list.append(all_weights)

    return alphas, correlations, rmses, all_weights_list


# def get_corr(returns, alpha):
#     combinator = ReturnCombinator()
#     combinator.build(250, 7, alpha)

#     returns_predictions = ewma_predictors(returns, [1, 5, 10, 21, 63, 125, 250])


#     returns_hats, all_weights = combinator.fit(returns, returns_predictions)
#     return np.corrcoef(returns_hats.values.flatten(), returns.loc[returns_hats.index].values.flatten())[0,1]


def _combinator_problem(memory, K, alpha):
    """
    param memory: number of days to look back
    param K: number of predictors
    param alpha: regularization parameter in combination problem
    """

    weights = cp.Variable(K)
    return_param = cp.Parameter(memory)
    return_prediction_param = cp.Parameter((memory, K))

    # halflife = 250
    # beta = np.exp(np.log(0.5) / halflife)
    # decay_weights = np.array([beta ** i for i in range(memory)])
    # decay_weights = decay_weights / np.sum(decay_weights)
    # decay_weights = decay_weights[::-1]
    # decay_weights = np.ones_like(decay_weights)

    objective = cp.Minimize(
        cp.sum_squares(return_param - return_prediction_param @ weights)
        + alpha * cp.norm1(weights)
    )
    # constraints = [cp.sum(weights)==1]
    # constraints = [weights<=1, weights>=-1]

    # problem = cp.Problem(objective, constraints)

    problem = cp.Problem(objective)

    return problem, weights, return_param, return_prediction_param


# def _combinator_problem2(memory, n, K, alpha):
#     """
#     param memory: number of days to look back
#     param K: number of predictors
#     param alpha: regularization parameter in combination problem
#     """

#     weights = cp.Variable((K,n))
#     returns_param = cp.Parameter((memory, n))
#     returns_predictions_param = [cp.Parameter((memory, K)) for _ in range(n)]

#     # halflife = 250
#     # beta = np.exp(np.log(0.5) / halflife)
#     # decay_weights = np.array([beta ** i for i in range(memory)])
#     # decay_weights = decay_weights / np.sum(decay_weights)
#     # decay_weights = decay_weights[::-1]
#     # decay_weights = np.ones_like(decay_weights)

#     objective = 0
#     for i in range(n):
#         objective += cp.sum_squares(
#             returns_param[:,i] - returns_predictions_param[i] @ weights[:,i]
#         )
#     objective += alpha * cp.norm1(weights)

#     problem = cp.Problem(cp.Minimize(objective))

#     return problem, weights, returns_param, returns_predictions_param


class ReturnCombinator:
    def __init__(self) -> None:
        pass

    def _get_weights(self, returns, returns_predictions, scale, alpha, weighted_hf=500):
        """
        param returns: pandas DataFrame of observed returns
        param returns_predictions: pandas DataFrame of various return predictors
        param scale: pandas series of scales for weighted least squares
        param alpha: regularization parameter in combination problem
        param weighted_hf: halflife for weighted least squares

        return: pandas series of weights
        """

        beta = np.exp(np.log(0.5) / weighted_hf)
        decay_weights = np.array([beta**i for i in range(len(returns_predictions))])
        decay_weights = decay_weights / np.sum(decay_weights) * len(returns_predictions)
        decay_weights = decay_weights[::-1]
        b = decay_weights.reshape(-1, 1)
        # b = np.ones_like(b)

        D = np.diag(scale.values)

        weights = np.linalg.solve(
            (returns_predictions.values.T * b.T) @ returns_predictions.values
            + alpha * D,
            (returns_predictions.values.T * b.T) @ returns.values,
        )

        return pd.Series(weights, index=returns_predictions.columns)

        # return np.linalg.inv((returns_predictions.T * b.T) @ returns_predictions + alpha * D) @ (returns_predictions.T * b.T) @ returns

    def _get_scales(self, returns_predictions, halflife=250):
        """
        param returns_predictions: pandas DataFrame of various return predictors

        return: pandas series of scales
        """
        squared_returns_predictions = [
            prediction**2 for prediction in returns_predictions
        ]

        variances = [
            squared_prediction.ewm(halflife).mean().shift(1)
            for squared_prediction in squared_returns_predictions
        ]

        # scales = [1 / variance for variance in variances]
        scales = [variance for variance in variances]

        return scales

    def fit(self, returns, returns_predictions, memory, alpha):
        """
        param returns: pandas DataFrame of observed returns
        param return_predictions: list of pandas DataFrames of various
            return predictors
        """

        n = len(returns_predictions)
        scales = self._get_scales(returns_predictions)

        times = returns_predictions[0].index
        for i in range(n):
            assert (times == returns_predictions[i].index).all()
            "predictors must have same indices"

        assert set(times).issubset(returns.index)
        "returns must have all predictor indices"

        # Create empty pandas dataframe to store weights
        all_weights = [
            np.nan * returns_predictions[i].iloc[memory:].copy() for i in range(n)
        ]

        # for t in trange(memory, len(times)):
        # for t in range(2, len(times)):
        for t in trange(memory, len(times)):  # Note: scales are None for day one
            t_start = times[max(0, t - memory)]
            t_end = times[t - 1]

            returns_temp = returns.loc[t_start:t_end]

            t_next = times[t]
            for i, asset in enumerate(returns.columns):
                scale = scales[i].loc[t_next]
                all_weights[i].loc[t_next] = self._get_weights(
                    returns_temp[asset],
                    returns_predictions[i].loc[t_start:t_end],
                    scale,
                    alpha,
                )

        returns_hats = {}
        for i in range(n):
            returns_hat = returns_predictions[i].iloc[memory:] * all_weights[i]
            returns_hats[i] = returns_hat.sum(axis=1)

        returns_hats = pd.DataFrame(returns_hats)
        returns_hats.columns = returns.columns

        return returns_hats, all_weights


# def get_r_hat(returns):
#     return_predictions = ewma_predictors(returns, [1, 5, 10, 21, 63, 125, 250])
#     returns_hat, all_weights = ewma_return_combination(returns.iloc[1:], return_predictions, alpha=0.0001, memory=250)

#     return returns_hat, all_weights


class ReturnCombinatorOld:
    def __init__(self) -> None:
        pass

    def build(self, memory, K, alpha):
        """
        param memory: number of days to look back
        param n: number of assets
        param K: number of predictors
        param alpha: regularization parameter in combination problem
        """
        self.K = K
        self.memory = memory

        (
            self.problem,
            self.weights,
            self.returns_param,
            self.returns_predictions_param,
        ) = _combinator_problem(memory, K, alpha)

    def fit(self, returns, returns_predictions):
        """
        param returns: pandas DataFrame of observed returns
        param return_predictions: list of pandas DataFrames of various
            return predictors
        """
        assert self.K is not None
        "build the model first"

        assert returns_predictions.shape[1] == self.K
        "number of predictors must match built model"

        times = returns_predictions.index

        assert set(times).issubset(returns.index)
        "returns must have all predictor indices"

        # Create empty pandas dataframe to store weights
        all_weights = np.nan * returns_predictions.iloc[self.memory :].copy()

        for t in trange(self.memory, len(times)):
            t_start = times[t - self.memory]
            t_end = times[t - 1]
            self.returns_param.value = returns.loc[t_start:t_end].values
            self.returns_predictions_param.value = returns_predictions.loc[
                t_start:t_end
            ].values
            self.problem.solve(ignore_dpp=True)

            t_next = times[t]
            all_weights.loc[t_next] = self.weights.value

        returns_hat = (returns_predictions.iloc[self.memory :] * all_weights).sum(
            axis=1
        )

        return returns_hat, all_weights

    # def fit(self, returns, returns_predictions):
    #     """
    #     param returns: pandas DataFrame of observed returns
    #     param return_predictions: list of pandas DataFrames of various
    #         return predictors
    #     """
    #     assert self.K is not None; "build the model first"

    #     for i in range(self.n):
    #         assert returns_predictions[i].shape[1] == self.K; "number of predictors must match built model"
    #     assert returns.shape[1] == self.n; "number of assets must match built model"

    #     times = returns_predictions[0].index
    #     for i in range(self.n):
    #         assert returns_predictions[i].index.equals(times); "predictor indices must match"

    #     assert set(times).issubset(returns.index); "returns must have all predictor indices"

    #     # Create empty pandas dataframe to store weights
    #     all_weights = [np.nan * predictor.iloc[self.memory:].copy() for predictor in returns_predictions]

    #     for t in trange(self.memory, len(times)):
    #         t_start = times[t-self.memory]
    #         t_end = times[t-1]
    #         self.returns_param.value = returns.loc[t_start:t_end].values
    #         for i in range(self.n):
    #             self.returns_predictions_param[i].value = returns_predictions[i].loc[t_start:t_end].values
    #         self.problem.solve(ignore_dpp=True)

    #         t_next = times[t]
    #         for i in range(self.n):
    #             all_weights[i].loc[t_next] = self.weights.value[:,i]

    #     returns_hats = {}
    #     for i in range(self.n):
    #         returns_hat = returns_predictions[i].iloc[self.memory:] * all_weights[i]
    #         returns_hats[i] = returns_hat.sum(axis=1)

    #     returns_hats = pd.DataFrame(returns_hats)

    #     return returns_hats, all_weights


def ewma_return_combination(returns, return_predictions, alpha, memory):
    """
    param returns: pandas Series of observed returns
    param return_predictions: pandas DataFrame of various return predictions
    param alpha: regularization parameter in combination problem
    param memory: number of days to look back in combination problem
    """
    times = returns.index

    problem, weights, return_param, return_prediction_param = _combinator_problem(
        memory, return_predictions.shape[1], alpha
    )

    # Create empty pandas datafrema to store weights of shape (T, K)
    all_weights = np.nan * return_predictions.iloc[memory:].copy()

    for t in trange(memory, returns.shape[0]):
        t_start = times[t - memory]
        t_end = times[t - 1]
        return_param.value = returns.loc[t_start:t_end].values
        return_prediction_param.value = return_predictions.loc[t_start:t_end].values
        problem.solve(ignore_dpp=True)

        t_next = times[t]
        all_weights.loc[t_next] = weights.value.T

    returns_hat = ((all_weights * return_predictions).dropna()).sum(axis=1)

    return returns_hat, all_weights


def daily_predictors(returns, lags):
    """
    param returns: pandas Series of observed returns
    param lags: list of lags to use in daily predictors
    """

    return_predictions = {}

    for lag in lags:
        return_predictions[lag] = returns.shift(lag).dropna()
    return_predictions = pd.DataFrame(return_predictions, index=returns.index)

    return return_predictions


def ewma_predictors(returns, halflives):
    """
    param returns: pandas Series of observed returns
    param halflives: list of halflives to use in EWMA
    """

    return_predictions = {}

    for halflife in halflives:
        return_predictions[halflife] = (
            returns.ewm(halflife=halflife).mean().shift(1).dropna()
        )
    return_predictions = pd.DataFrame(return_predictions, index=returns.index[1:])

    return return_predictions


# def combinator_problem(returns, K, alpha):
#     T, n = returns.shape

#     weights = cp.Variable(K)
#     return_prediction_params = {i: cp.Parameter((T, n)) for i in range(K)}

#     print(returns.shape)
#     print(cp.sum([weights[i] * return_prediction_params[i] for i in range(K)]).shape)

#     print(returns.shape)
#     print(return_prediction_params[0].shape)
#     print((returns - return_prediction_params[0]).shape)


#     objective = cp.Minimize(
#         cp.sum_squares(returns - cp.sum([weights[i] * return_prediction_params[i] for i in range(K)])) + alpha * cp.norm1(weights)
#     )
#     problem = cp.Problem(objective)

#     return problem, weights, return_prediction_params


# def return_combination(returns, return_predictions, alpha):
#     """
#     param returns: pandas DataFrame of observed returns
#     param return_predictions: dictionary of various return predictions
#         each prediction is a pandas DataFrame
#     param alpha: regularization parameter in combination problem

#     return: weight to put on each predictior
#     """
#     K = len(return_predictions)

#     problem, weights, return_prediction_params = combinator_problem(returns, K, alpha)
