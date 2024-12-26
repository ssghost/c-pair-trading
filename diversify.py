import pandas as pd
import yfinance as yf

from pypfopt import expected_returns, risk_models, black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier

import warnings
warnings.filterwarnings("ignore")

mag_7 = [
    "AAPL",
    "AMZN",
    "NVDA",
    "TLSA",
    "GOOGL",
    "META",
    "MSFT",
]

prices = yf.download(mag_7, start="2020-01-01")["Adj Close"]

views = {
    "AAPL": 0.05,
    "AMZN": 0.15,
    "NVDA": 0.25,
    "TLSA": -0.05,
    "GOOGL": -0.15,
    "META": 0.07,
    "MSFT": 0.12
}

mcaps = {
    "AAPL": 2.5e12,
    "AMZN": 1.3e12,
    "NVDA": 1.0e12,
    "TLSA": 0.9e12,
    "GOOGL": 1.4e12,
    "META": 0.7e12,
    "MSFT": 2.0e12,
}

expected_returns_mean = expected_returns.mean_historical_return(prices)
cov_matrix = risk_models.sample_cov(prices)

delta = black_litterman.market_implied_risk_aversion(prices)
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)

bl = BlackLittermanModel(
    cov_matrix,
    absolute_views=views,
    pi=market_prior
)

bl_returns = bl.bl_returns()

ef = EfficientFrontier(bl_returns, cov_matrix)

weights = ef.max_sharpe()

bl_weights = pd.DataFrame(
    list(weights.items()), 
    columns=["symbol", "weight"]
).set_index("symbol")

performance = ef.portfolio_performance(verbose=True)
print(f"Sharpe ratio: {performance[0]:.2%}")