import yfinance as yf
import statsmodels.api as sm


tickers = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOG', 'META', 'TSLA', 'QQQ']
data = yf.download(tickers, start='2022-01-01', end='2023-12-31')['Adj Close']

benchmark_returns = (
    data
    .pop("QQQ")
    .pct_change()
    .dropna()
)

portfolio_returns = (
    data
    .pct_change()
    .dropna()
    .sum(axis=1)
)

portfolio_returns.plot()
benchmark_returns.plot()

def linreg(x, y):    
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model

X = benchmark_returns.values
Y = portfolio_returns.values

model = linreg(X, Y)
alpha, beta = model.params[0], model.params[1]

print(model.summary())
print(f"Alpha: {alpha}")
print(f"Beta: {beta}")

hedged_portfolio_returns = -beta * benchmark_returns + portfolio_returns

P = hedged_portfolio_returns.values
model = linreg(X, P)
alpha, beta = model.params[0], model.params[1]

print(f"Alpha: {alpha}")
print(f"Beta: {round(beta, 6)}")

hedged_portfolio_returns.plot()
benchmark_returns.plot()