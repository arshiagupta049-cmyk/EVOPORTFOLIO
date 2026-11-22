"""Grey Wolf Optimization (GWO) for constrained portfolio weights."""
import numpy as np

TRADING_DAYS = 252


def project_weights(vector, max_weight=1.0):
    """Turn any candidate vector into valid non-negative weights summing to 1."""
    vector = np.asarray(vector, dtype=float)
    vector = np.clip(vector, 0.0, None)
    if vector.sum() == 0:
        vector = np.ones_like(vector)
    weights = vector / vector.sum()

    # Enforce a per-asset cap by repeatedly redistributing excess to uncapped assets.
    if max_weight < 1.0:
        weights = np.minimum(weights, max_weight)
        for _ in range(100):
            remaining = 1.0 - weights.sum()
            if remaining <= 1e-10:
                break
            free = weights < max_weight - 1e-10
            if not free.any():
                break
            weights[free] += remaining / free.sum()
            weights = np.minimum(weights, max_weight)
        weights = weights / weights.sum()
    return weights


def portfolio_sharpe(weights, returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio for a candidate portfolio."""
    daily = returns.to_numpy() @ weights
    mean_daily = np.mean(daily)
    volatility_daily = np.std(daily, ddof=1)
    annual_return = mean_daily * TRADING_DAYS
    annual_volatility = volatility_daily * np.sqrt(TRADING_DAYS)
    if annual_volatility < 1e-12:
        return -1e6
    return float((annual_return - risk_free_rate) / annual_volatility)


def grey_wolf_optimizer(returns, population_size=20, iterations=40,
                        max_weight=0.60, risk_free_rate=0.06, seed=42):
    """
    Search for portfolio weights using Grey Wolf Optimization.
    Returns (best_weights, best_sharpe_history).
    """
    rng = np.random.default_rng(seed)
    asset_count = returns.shape[1]
    if asset_count < 2:
        raise ValueError("GWO needs at least two assets.")
    if max_weight * asset_count < 1.0:
        raise ValueError("Maximum weight is too small for weights to sum to 1.")

    wolves = np.array([project_weights(rng.random(asset_count), max_weight)
                       for _ in range(population_size)])
    history = []

    def score(wolf):
        return portfolio_sharpe(wolf, returns, risk_free_rate)

    for step in range(iterations):
        scores = np.array([score(w) for w in wolves])
        order = np.argsort(scores)[::-1]
        alpha, beta, delta = wolves[order[0]], wolves[order[1]], wolves[order[2]]
        history.append(float(scores[order[0]]))
        a = 2.0 - 2.0 * step / max(iterations - 1, 1)

        updated = []
        for wolf in wolves:
            leader_moves = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(asset_count), rng.random(asset_count)
                A = 2 * a * r1 - a
                C = 2 * r2
                distance = np.abs(C * leader - wolf)
                leader_moves.append(leader - A * distance)
            candidate = np.mean(leader_moves, axis=0)
            updated.append(project_weights(candidate, max_weight))
        wolves = np.asarray(updated)

    final_scores = np.array([score(w) for w in wolves])
    best = wolves[np.argmax(final_scores)]
    history.append(float(np.max(final_scores)))
    return best, history
