# EvoPortfolio — Grey Wolf Optimization Portfolio Lab

A student-friendly, research-style Python project inspired by the uploaded experiential-learning proposal. It connects financial data processing, Grey Wolf Optimization (GWO), and out-of-sample backtesting in a Streamlit dashboard.

> Educational prototype only. It is not financial advice and does not promise future returns.

## Features
- Interactive Streamlit dashboard
- Built-in simulated demo data (works without internet)
- Optional Yahoo Finance adjusted prices
- Daily returns and portfolio risk/return calculations
- Grey Wolf Optimization with alpha, beta and delta leaders
- Long-only weights, sum-to-one normalization, optional per-stock cap
- Out-of-sample test period
- Comparison against equal-weight and average-asset market-proxy baselines
- Allocation chart, convergence chart, cumulative-growth chart and metrics table

## Run it
1. Install Python 3.10 or newer.
2. Open a terminal inside this folder.
3. Create an environment (optional but recommended):
   - Windows: `py -m venv .venv`
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
4. Install packages: `pip install -r requirements.txt`
5. Start dashboard: `streamlit run app.py`
6. Choose **Built-in demo data** first, then click **Run portfolio experiment**.

For real data, select **Yahoo Finance**, enter symbols such as `RELIANCE.NS, TCS.NS, INFY.NS`, choose dates, and run. Real data needs internet access and the Yahoo Finance service may change.

## Project structure
- `app.py` — dashboard, controls, charts and result display
- `finance.py` — price data, returns, performance metrics and backtest
- `optimizer.py` — weight constraints, Sharpe fitness and GWO search
- `requirements.txt` — Python packages

## Function-by-function guide

### `finance.py`
- `load_demo_prices(days=900, seed=7)`: makes repeatable *simulated* price series. These are not actual market prices.
- `load_market_prices(tickers, start, end)`: downloads adjusted prices through yfinance, cleans missing values and checks the result.
- `calculate_returns(prices)`: computes daily percentage change: `(today price / yesterday price) - 1`.
- `portfolio_metrics(daily_returns, risk_free_rate=0.06)`: computes annualized return, annualized volatility, Sharpe ratio and maximum drawdown.
- `backtest_portfolios(test_returns, optimized_weights, risk_free_rate=0.06)`: applies fixed optimized weights to unseen returns and creates comparison curves. The risk-free-rate argument is kept for a consistent interface; metrics are calculated separately in the dashboard.

### `optimizer.py`
- `project_weights(vector, max_weight=1.0)`: clips negative values, normalizes weights to sum to one and redistributes excess when a per-stock cap is active.
- `portfolio_sharpe(weights, returns, risk_free_rate=0.0)`: computes daily portfolio returns, annualizes return/volatility and calculates Sharpe ratio.
- `grey_wolf_optimizer(returns, population_size=20, iterations=40, max_weight=0.60, risk_free_rate=0.06, seed=42)`: creates candidate portfolios (“wolves”), ranks them, uses alpha/beta/delta leaders to update candidates, projects every candidate back into valid weights, and returns the best weights plus convergence history.

### Dashboard flow in `app.py`
- `get_prices(...)`: cached wrapper selecting demo or online data.
- Sidebar controls: data source, tickers, dates, population size, iteration count, weight cap and risk-free rate.
- `st.session_state`: keeps the loaded prices and latest experiment available between dashboard interactions.
- Tabs show allocation, GWO convergence, test-period performance and a written pipeline explanation.

## Core GWO idea in plain English
1. Each wolf is a possible stock allocation.
2. The best three wolves are called alpha, beta and delta.
3. The rest move toward those three leaders, with some randomness.
4. The parameter `a` decreases over time: early iterations explore more; later iterations refine.
5. Each new candidate is converted into valid portfolio weights.
6. The best candidate is selected using training-period Sharpe ratio.

## Important assumptions and limitations
- The demo dataset is synthetic; do not present it as actual stock history.
- A single fixed allocation is used during the test window.
- No transaction fees, taxes, slippage or liquidity constraints are modeled.
- The “Market Proxy” is the average of the selected assets, not an official market index.
- This implementation is for learning and reproducible experiments, not investment decisions.
- The simple Sharpe objective is single-objective. Multi-objective GWO and rolling rebalancing can be added later.
