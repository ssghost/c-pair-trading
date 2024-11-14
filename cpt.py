from vectorbtpro import *
import pandas as pd
import scipy.stats as st
import statsmodels.tsa.stattools as ts  
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

COINT_FILE = "coint_pvalues.pickle"
POOL_FILE = "data_pool.h5"
START = "2015-01-01"
END = "2023-12-31"

if not vbt.file_exists(POOL_FILE):
    with vbt.ProgressBar(total=len(sp500_tickers)) as pbar:
        collected = 0
        for symbol in sp500_tickers:
            try:
                data = vbt.YFData.pull(
                    symbol,
                    start=START,
                    end=END,
                    silence_warnings=True,
                )
                data.to_hdf(POOL_FILE)
                collected += 1
            except:
                pass
            pbar.set_prefix(f"{symbol} ({collected})")
            pbar.update()

data = vbt.HDFData.pull(
    POOL_FILE,
    start=START,
    end=END,
    silence_warnings=True
)

data = data.select_symbols([
    k
    for k, v in data.data.items()
    if not v.isnull().any().any()
])

@vbt.parameterized(
    merge_func="concat",
    engine="pathos",
    distribute="chunks",
    n_chunks="auto"
)
def coint_pvalue(close, s1, s2):
    return ts.coint(np.log(close[s1]), np.log(close[s2]))[1]

if not vbt.file_exists(COINT_FILE):
    coint_pvalues = coint_pvalue(
        data.close,
        vbt.Param(data.symbols, condition="s1 != s2"),
        vbt.Param(data.symbols)
    )
    vbt.save(coint_pvalues, COINT_FILE)
else:
    coint_pvalues = vbt.load(COINT_FILE)

coint_pvalues = coint_pvalues.sort_values()
coint_pvalues.head(20)

S1, S2 = "WYNN", "DVN"

data = vbt.YFData.pull(
    [S1, S2],
    start=START,
    end=END,
    silence_warnings=True,
)

UPPER = st.norm.ppf(1 - 0.05 / 2)
LOWER = -st.norm.ppf(1 - 0.05 / 2)

S1_close = data.get("Close", S1)
S2_close = data.get("Close", S2)
ols = vbt.OLS.run(S1_close, S2_close, window=vbt.Default(21))
spread = ols.error.rename("Spread")
zscore = ols.zscore.rename("Z-score")

upper_crossed = zscore.vbt.crossed_above(UPPER)
lower_crossed = zscore.vbt.crossed_below(LOWER)

long_entries = data.symbol_wrapper.fill(False)
short_entries = data.symbol_wrapper.fill(False)

short_entries.loc[upper_crossed, S1] = True
long_entries.loc[upper_crossed, S2] = True
long_entries.loc[lower_crossed, S1] = True
short_entries.loc[lower_crossed, S2] = True

pf = vbt.Portfolio.from_signals(
    data,
    entries=long_entries,
    short_entries=short_entries,
    size=10,
    size_type="valuepercent100",
    group_by=True,
    cash_sharing=True,
    call_seq="auto"
)

pf.stats()

