# %%
import pandas as pd

# https://eresearch.fidelity.com/eresearch/markets_sectors/sectors/sectors_in_market.jhtml
market_cap_by_sector = pd.Series(
    {
        "Communication Services": 5.37,
        "Consumer Discretionary": 7.5,
        "Consumer Staples": 4.12,
        "Energy": 1.76,
        "Financials": 5.95,
        "Health Care": 6.89,
        "Industrials": 4.46,
        "IT": 12.31,
        "Materials": 2.1,
        "Real Estate": 1.35,
        "Utilities": 1.49,
    }
)

num_stock_by_sector = (30 * (market_cap_by_sector / market_cap_by_sector.sum())).round()
print(num_stock_by_sector)
# %%

import yfinance as yf

stock_list = [
    "T",
    "EA",
    "VZ",
    "BBY",
    "KMX",
    "CMG",
    "F",
    "KO",
    "WMT",
    "CVX",
    "JPM",
    "MS",
    "C",
    "JNJ",
    "MRK",
    "PFE",
    "CVS",
    "AAL",
    "FDX",
    "UPS",
    "AAPL",
    "AMD",
    "MSFT",
    "NVDA",
    "ORCL",
    "QCOM",
    "INTC",
    "EMN",
    "AVB",
    "SO",
]

stock_close = yf.download(stock_list, start="2010-01-01", end="2020-01-01")["Close"]
# print(stock_data.index[stock_data["Close"].isna().any(axis=1)]) # market closed on 2012-10-29
stock_close = stock_close.dropna(axis=0, how="all")

stock_ret = (stock_close - stock_close.shift()) / stock_close.shift()
stock_ret = stock_ret.dropna()
stock_cov = stock_ret.cov()
stock_corr = stock_ret.corr()
# %%
import os

newpath = os.getcwd() + "\\output"
if not os.path.exists(newpath):
    os.makedirs(newpath)

filename = newpath + "\\data.xlsx"
with pd.ExcelWriter(filename) as writer:
    stock_close.to_excel(writer, sheet_name="stock_close")
    stock_ret.to_excel(writer, sheet_name="stock_ret")
    stock_cov.to_excel(writer, sheet_name="stock_cov")
    stock_corr.to_excel(writer, sheet_name="stock_corr")

# %%
