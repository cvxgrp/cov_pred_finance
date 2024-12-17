import marimo

__generated_with = "0.10.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""# Covariance estimation""")
    return


@app.cell
def _(__file__):
    from pathlib import Path

    path = Path(__file__).parent

    import pandas as pd

    pd.options.plotting.backend = "plotly"
    return Path, path, pd


@app.cell
def _(path, pd):
    # Load historic price data of 20 stocks
    prices = pd.read_csv(
        path / "data" / "stock_prices.csv", header=0, index_col=0, parse_dates=True
    )
    return (prices,)


@app.cell
def _(prices):
    # Compute the historic returns
    returns = prices.pct_change().dropna(axis=0, how="all")[["GOOG", "AAPL", "FB"]]
    return (returns,)


@app.cell
def _():
    from cvx.covariance.combination import from_ewmas

    return (from_ewmas,)


@app.cell
def _(from_ewmas, pd, returns):
    # Pairs of halflife volatility vs halflife covariances
    pairs = [(10, 10), (21, 21), (21, 63)]

    combinations = from_ewmas(returns, pairs, clip_at=4.2, mean=True)

    weights = pd.DataFrame(
        {result.time: result.weights for result in combinations.solve(window=10)}
    ).transpose()
    weights
    return combinations, pairs, weights


@app.cell
def _(weights):
    weights.plot()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
