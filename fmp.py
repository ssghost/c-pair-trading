import io
import os
import time
import requests
import pandas as pd
import arcticdb as adb


arctic = adb.Arctic("lmdb://fundamantals")
lib = arctic.get_library("financial_ratios", create_if_missing=True)

def build_fmp_url(request, period, year):
    apikey = os.environ.get("FMP_API_KEY")
    return f"https://financialmodelingprep.com/api/v4/{request}?year={year}&period={period}&apikey={apikey}"

def get_fmp_data(request, period, year):
    url = build_fmp_url(request, period, year)
    response = requests.get(url)
    csv = response.content.decode("utf-8")
    return pd.read_csv(io.StringIO(csv), parse_dates=True)

ratios = get_fmp_data("ratios-bulk", "quarter", "2020")

for year in [2020, 2021, 2022]:
    ratios = get_fmp_data("ratios-bulk", "quarter", year)
    adb_sym = f"financial_ratios/{year}"
    adb_fcn = lib.update if lib.has_symbol(adb_sym) else lib.write
    adb_fcn(adb_sym, ratios)
    time.sleep(3)

def filter_by_year(year):
    cols = [
        "symbol",
        "period",
        "date",
        "debtEquityRatio", 
        "currentRatio", 
        "priceToBookRatio", 
        "returnOnEquity", 
        "returnOnAssets", 
        "interestCoverage"
    ]
    
    q = adb.QueryBuilder()
    filter = (
        (q["debtEquityRatio"] < 0.5)
        & (
            (q["currentRatio"] > 1.5) & (q["currentRatio"] < 2.5)
        )
        & (q["priceToBookRatio"] < 1.5)
        & (q["returnOnEquity"] > 0.08)
        & (q["returnOnAssets"] > 0.06)
        & (q["interestCoverage"] > 5)
    )
    q = q[filter]
    
    return lib.read(
        f"financial_ratios/{year}", 
        query_builder=q
    ).data[cols].set_index("symbol")

if __name__ == "__main__":
    filter_by_year(year)