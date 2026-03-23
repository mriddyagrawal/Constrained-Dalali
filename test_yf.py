import yfinance as yf
import pandas as pd

TICKERS = ['AAPL', 'MSFT']
yf_df = yf.download(
    TICKERS,
    start='2020-01-01',
    end='2020-01-10',
    auto_adjust=True,
    progress=False
)
print(yf_df.columns)
print(yf_df.head())
