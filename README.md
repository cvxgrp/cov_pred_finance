# [cvxcovariance](http://www.cvxgrp.org/cov_pred_finance)

Consider a vector time series of $n$ financial returns, denoted by the $n$-dimensional return vectors $r_t$, $t=1,2,\ldots$. (We take $r_t$ to be the return from $t-1$ to $t$.)

The $\texttt{cvxsimulator}$ package
provides simple tools for creating an estimate $\hat\Sigma_t$ of the covariance $\Sigma_t$
at each time step.

In the simplest case the user provides an $n\times T$ pandas DataFrame
of returns $r_1,\ldots,r_T$ and gets back covariance predictors for the $T$ time
steps. Note: at time $t$ the user is provided with $\Sigma_{t+1}$,
$\textit{i.e.}$, the covariance matrix for the next time step.

## Installation
To install the package, run the following command in the terminal:

```bash
pip install cvxcovariance
```

## Usage
There are two main ways to use the package. The first is to use the
$\texttt{covariance$\_$combination}$
 function to create a CM-IEWMA predictor. The
second is to define your own covariance predictors, via dictionaries, and pass
them to the $\texttt{CovarianceCombination}$ class.

### CM-IEWMA
The $\texttt{covariance_combination}$ function takes in a pandas DataFrame of
returns and the IEWMA half-life pairs and returns an iterator object that
iterates over the CM-IEWMA covariance predictors defined via a namedtuple:
    
```python
prices = pd.read_csv(
        "resources/stock_prices.csv", index_col=0, header=0, parse_dates=True
    ).ffill()
returns = prices.pct_change().dropna()

halflife_pairs = [(10, 21), (21, 63), (63, 125)]

covariance_predictors = {}
iewma_weights = {}
for predictor in covariance_combination(returns, half_life_pairs):
    covariance_predictors[predictor.time] = predictor.covariance
```

### CovarianceCombination
The $\texttt{CovarianceCombination}$ class takes in a pandas DataFrame of
returns and a dictionary of covariance predictors $\texttt{\{key: \{time:
sigma\}\}}$. For example, here we combine two EWMA covariance predictors from pandas:

```python
import panda as pd
from cvxcovariance import CovarianceCombination

prices = pd.read_csv(
        "resources/stock_prices.csv", index_col=0, header=0, parse_dates=True
    ).ffill()
returns = prices.pct_change().dropna()

# Define 21 and 63 day EWMAs as dictionaries
ewma21 = returns.ewm(halflife=21, min_periods=63).cov().dropna()
ewma21 = {time: ewma21.loc[time] for time in ewma21.index.get_level_values(0).unique()}

ewma63 = returns.ewm(halflife=63, min_periods=63).cov().dropna()
ewma63 = {time: ewma63.loc[time] for time in ewma63.index.get_level_values(0).unique()}

ewmas = {21: ewma21, 63: ewma63}

# Define the combinator and solve combination problems
combinator = CovarianceCombination(sigmas=ewmas, returns=returns)

covariance_predictors = {}
for predictor in combinator.solve(window=10):
    covariance_predictors[predictor.time] = predictor.covariance
```

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
poetry install
```

to replicate the virtual environment we have defined in pyproject.toml.

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual
environment. Executing

```bash
./create_kernel.sh
```

constructs a dedicated
[Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) for the
project.


