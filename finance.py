
"""Price loading, return calculations, metrics and simple backtesting."""

import numpy as np
import pandas as pd


def load_demo_prices(days=900, seed=7):
    """Create repeatable simulated stock prices without internet."""

    rng = np.random.default_rng(seed)

    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]

    drift = np.array([0.00045, 0.00038, 0.00042, 0.00030, 0.00025])
    volatility = np.array([0.018, 0.014, 0.017, 0.013, 0.011])

    shocks = (
        rng.normal(size=(days, len(tickers))) * volatility + drift
    )

    prices = 100 * np.exp(np.cumsum(shocks, axis=0))

    # Create exactly 'days' dates.
    dates = pd.date_range(
        start="2020-01-01",
        periods=days,
        freq="D"
    )

    # Safety check: prices and dates must have equal lengths.
    assert len(dates) == len(prices), "Date and price lengths do not match!"

    return pd.DataFrame(
        prices,
        index=dates,
        columns=tickers
    )


def load_market_prices(tickers, start, end):
    """Download adjusted closing prices using Yahoo Finance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install yfinance using: pip install yfinance"
        ) from exc

    if not tickers or len(tickers) < 2:
        raise ValueError("Enter at least two ticker symbols.")

    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if data.empty:
        raise ValueError(
            "No prices were returned. Check the ticker symbols and dates."
        )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise ValueError("Could not find closing prices in downloaded data.")
    else:
        if "Close" not in data.columns:
            raise ValueError("Downloaded data has no Close column.")

        prices = data[["Close"]].rename(
            columns={"Close": tickers[0]}
        )

    prices = prices.dropna(how="all").ffill().dropna()

    # Remove assets with too little valid price history.
    prices = prices.loc[:, prices.notna().sum() > 30]

    if prices.shape[1] < 2:
        raise ValueError(
            "Fewer than two stocks have sufficient valid price history."
        )

    return prices


def calculate_returns(prices):
    """Convert prices into daily percentage returns."""

    returns = prices.pct_change()

    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    return returns


def portfolio_metrics(daily_returns, risk_free_rate=0.06):
    """Calculate annual return, volatility, Sharpe ratio and max drawdown."""

    daily_returns = pd.Series(daily_returns).dropna()

    if daily_returns.empty:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0
        }

    # Growth of an initial investment of 1 unit.
    growth = (1 + daily_returns).cumprod()

    # Annualized compounded return.
    annual_return = (
        growth.iloc[-1] ** (252 / len(daily_returns))
    ) - 1

    # Annualized standard deviation of returns.
    annual_volatility = (
        daily_returns.std(ddof=1) * np.sqrt(252)
    )

    # Return earned per unit of risk above the risk-free rate.
    if annual_volatility > 1e-12:
        sharpe = (
            daily_returns.mean() * 252 - risk_free_rate
        ) / annual_volatility
    else:
        sharpe = 0.0

    # Drop from the highest growth value reached so far.
    drawdown = growth / growth.cummax() - 1

    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min())
    }


def backtest_portfolios(
    test_returns,
    optimized_weights,
    risk_free_rate=0.06
):
    """Compare GWO weights with equal-weight and market-proxy baselines."""

    columns = list(test_returns.columns)
    optimized_weights = np.asarray(optimized_weights, dtype=float)

    if len(columns) != len(optimized_weights):
        raise ValueError(
            "The number of weights must match the number of stocks."
        )

    if len(columns) == 0:
        raise ValueError("No stocks are available for backtesting.")

    # Strategy 1: optimized portfolio.
    gwo_daily = pd.Series(
        test_returns.to_numpy() @ optimized_weights,
        index=test_returns.index,
        name="GWO"
    )

    # Strategy 2: give every stock an equal allocation.
    equal_weights = np.repeat(1 / len(columns), len(columns))

    equal_daily = pd.Series(
        test_returns.to_numpy() @ equal_weights,
        index=test_returns.index,
        name="Equal Weight"
    )

    # Strategy 3: average return of the selected stocks.
    market_daily = test_returns.mean(axis=1).rename("Market Proxy")

    daily_returns = {
        "GWO": gwo_daily,
        "Equal Weight": equal_daily,
        "Market Proxy": market_daily
    }

    daily_df = pd.DataFrame(daily_returns)

    # Convert daily returns into cumulative growth curves.
    curves = (1 + daily_df).cumprod()

    return {
        "daily_returns": daily_returns,
        "curves": curves
    }
