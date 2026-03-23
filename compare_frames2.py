import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START = '2017-01-01'
END = '2025-12-31'
USE_ADJ_CLOSE = True
keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# Single
frames_single = []
for ticker in TICKERS:
    yf_df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=USE_ADJ_CLOSE,
        progress=False
    )
    if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
        yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
    if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
        yf_df['Close'] = yf_df['Adj Close']
    for c in keep_cols:
        if c not in yf_df.columns:
            yf_df[c] = np.nan
    df = yf_df[keep_cols].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df['Ticker'] = ticker
    df = df.reset_index().set_index(['Date', 'Ticker'])
    frames_single.append(df)
prices_single = pd.concat(frames_single).sort_index()

# Batch
frames_batch = []
yf_batch = yf.download(
    TICKERS,
    start=START,
    end=END,
    auto_adjust=USE_ADJ_CLOSE,
    progress=False
)

for ticker in TICKERS:
    if isinstance(yf_batch.columns, pd.MultiIndex):
        yf_df = yf_batch.xs(ticker, axis=1, level=1).copy()
    else:
        yf_df = yf_batch.copy()
    yf_df = yf_df.dropna(how='all')

    if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
        yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
    if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
        yf_df['Close'] = yf_df['Adj Close']
    for c in keep_cols:
        if c not in yf_df.columns:
            yf_df[c] = np.nan
    df = yf_df[keep_cols].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df['Ticker'] = ticker
    df = df.reset_index().set_index(['Date', 'Ticker'])
    frames_batch.append(df)
prices_batch = pd.concat(frames_batch).sort_index()

print("Equals?", prices_single.equals(prices_batch))
if not prices_single.equals(prices_batch):
    print("Different items:")
    diff = prices_single.compare(prices_batch)
    print(diff.head())
    print("Shape single:", prices_single.shape)
    print("Shape batch:", prices_batch.shape)
