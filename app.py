
import streamlit as st
import pandas as pd
import plotly.express as px

from finance import (
    load_demo_prices,
    load_market_prices,
    calculate_returns,
    portfolio_metrics,
    backtest_portfolios,
)
from optimizer import grey_wolf_optimizer

st.set_page_config(
    page_title="EvoPortfolio | GWO Lab",
    page_icon="🐺",
    layout="wide",
)

st.title("🐺 EvoPortfolio")
st.caption(
    "Grey Wolf Optimization for data-driven stock portfolio selection & backtesting"
)
st.info(
    "Educational research prototype only — historical results do not predict future performance."
)

with st.sidebar:
    st.header("Experiment controls")

    data_mode = st.radio(
        "Data source",
        ["Built-in demo data", "Yahoo Finance (internet required)"],
    )

    ticker_text = st.text_input(
        "Tickers (comma-separated)",
        "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS",
    )

    start_date = st.date_input(
        "Start date", pd.Timestamp("2021-01-01")
    )
    end_date = st.date_input(
        "End date", pd.Timestamp("2025-12-31")
    )

    st.divider()

    population = st.slider("Wolf population", 10, 50, 20, 5)
    iterations = st.slider("Optimization iterations", 10, 100, 40, 10)
    max_weight = st.slider(
        "Maximum weight per stock", 0.25, 1.0, 0.60, 0.05
    )
    risk_free_rate = st.number_input(
        "Annual risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
    ) / 100

    run = st.button(
        "Run portfolio experiment",
        type="primary",
        use_container_width=True,
    )


@st.cache_data(show_spinner=False)
def get_prices(mode, tickers, start, end):
    if mode == "Built-in demo data":
        return load_demo_prices()

    return load_market_prices(tickers, str(start), str(end))


if run or "prices" not in st.session_state:
    try:
        tickers = [
            ticker.strip().upper()
            for ticker in ticker_text.split(",")
            if ticker.strip()
        ]

        if end_date <= start_date:
            st.error("End date must be later than start date.")
            st.stop()

        prices = get_prices(
            data_mode, tuple(tickers), start_date, end_date
        )

        if prices.empty or prices.shape[1] < 2:
            st.error(
                "Please use at least two valid assets and a date range "
                "with enough observations."
            )
            st.stop()

        st.session_state["prices"] = prices
        st.session_state["run_config"] = (
            population,
            iterations,
            max_weight,
            risk_free_rate,
        )

        # A new data load requires a new optimization and backtest.
        st.session_state.pop("experiment", None)

    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()


prices = st.session_state["prices"]

population, iterations, max_weight, risk_free_rate = (
    st.session_state.get(
        "run_config",
        (population, iterations, max_weight, risk_free_rate),
    )
)

returns = calculate_returns(prices)

split = max(30, int(len(returns) * 0.70))
split = min(split, len(returns))

train_returns = returns.iloc[:split]
test_returns = returns.iloc[split:]

if len(test_returns) < 5:
    st.warning(
        "Not enough rows for a meaningful test period. "
        "Use a longer date range."
    )
    st.stop()


if run or "experiment" not in st.session_state:
    try:
        with st.spinner("Optimizing portfolio and running backtest..."):
            weights, history = grey_wolf_optimizer(
                train_returns,
                population_size=population,
                iterations=iterations,
                max_weight=max_weight,
                risk_free_rate=risk_free_rate,
            )

            result = backtest_portfolios(
                test_returns, weights, risk_free_rate
            )

            st.session_state["experiment"] = (
                weights,
                history,
                result,
                train_returns,
                test_returns,
            )

    except Exception as exc:
        st.error(f"Could not run the portfolio experiment: {exc}")
        st.stop()


weights, history, result, train_returns, test_returns = (
    st.session_state["experiment"]
)

names = list(prices.columns)

weights_df = pd.DataFrame(
    {"Stock": names, "Weight": weights}
).sort_values("Weight", ascending=False)

# All strategy return series are stored inside daily_returns.
daily_returns = result["daily_returns"]

gwo = daily_returns["GWO"]
equal = daily_returns["Equal Weight"]
market = daily_returns["Market Proxy"]


c1, c2, c3, c4 = st.columns(4)

c1.metric("Assets", len(names))
c2.metric("Training days", len(train_returns))
c3.metric("Testing days", len(test_returns))
c4.metric(
    "Top allocation",
    f"{weights_df.iloc[0]['Weight']:.1%}",
    weights_df.iloc[0]["Stock"],
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Portfolio overview",
        "Wolf optimizer",
        "Backtest",
        "How it works",
    ]
)


with tab1:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Optimized allocation")

        fig = px.pie(
            weights_df,
            names="Stock",
            values="Weight",
            hole=0.45,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Weights table")

        st.dataframe(
            weights_df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Weights are non-negative, sum to 100%, and respect "
            "the selected per-stock cap when feasible."
        )

    st.subheader("Historical adjusted prices")

    st.plotly_chart(
        px.line(
            prices,
            x=prices.index,
            y=names,
            labels={"value": "Price", "index": "Date"},
        ),
        use_container_width=True,
    )


with tab2:
    st.subheader("Grey Wolf Optimization convergence")

    hist_df = pd.DataFrame(
        {
            "Iteration": range(1, len(history) + 1),
            "Best training Sharpe": history,
        }
    )

    st.plotly_chart(
        px.line(
            hist_df,
            x="Iteration",
            y="Best training Sharpe",
            markers=True,
        ),
        use_container_width=True,
    )

    st.markdown("""
    **Wolf mapping:** Each wolf represents one possible set of stock
    weights. Alpha, beta and delta are the three best candidates.
    The other wolves update their positions using those leaders.
    The control value `a` gradually decreases, shifting the search
    from broad exploration toward smaller refinements.
    """)

    st.code(
        "fitness = annualized_return / annualized_volatility",
        language="python",
    )


with tab3:
    st.subheader("Out-of-sample cumulative growth")

    curve = result["curves"].reset_index(names="Date").melt(
        id_vars="Date",
        var_name="Strategy",
        value_name="Growth of ₹1",
    )

    st.plotly_chart(
        px.line(
            curve,
            x="Date",
            y="Growth of ₹1",
            color="Strategy",
        ),
        use_container_width=True,
    )

    metric_rows = []

    for strategy, daily in daily_returns.items():
        metrics = portfolio_metrics(daily, risk_free_rate)

        metric_rows.append(
            {
                "Strategy": strategy,
                "Annual return": metrics["annual_return"],
                "Annual volatility": metrics["annual_volatility"],
                "Sharpe ratio": metrics["sharpe"],
                "Max drawdown": metrics["max_drawdown"],
            }
        )

    metric_df = pd.DataFrame(metric_rows)

    st.dataframe(
        metric_df.style.format(
            {
                "Annual return": "{:.2%}",
                "Annual volatility": "{:.2%}",
                "Sharpe ratio": "{:.2f}",
                "Max drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The test period is held out from optimization. The market proxy "
        "is the average return of the selected assets, not an official index."
    )


with tab4:
    st.subheader("Pipeline")

    st.markdown("""
    1. **Load prices** — use bundled demo data or fetch adjusted prices.
    2. **Calculate returns** — convert price changes into daily percentage returns.
    3. **Split chronologically** — first 70% for optimization, final 30% for testing.
    4. **Optimize** — GWO searches for weights with a strong training-period Sharpe ratio.
    5. **Backtest** — freeze weights and evaluate on unseen test returns.
    6. **Compare** — evaluate GWO against equal weights and a simple market proxy.

    **Important limitation:** This teaching implementation uses a single
    fixed allocation during the test window and does not model transaction
    costs, taxes, slippage or survivorship bias.
    """)

    st.write("Data preview")
    st.dataframe(prices.tail(10), use_container_width=True)
