import pandas as pd
import yfinance as yf
import numpy as np

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START = '2017-01-01'
END = '2025-12-31'
USE_ADJ_CLOSE = True

yf_batch = yf.download(
    TICKERS,
    start=START,
    end=END,
    auto_adjust=USE_ADJ_CLOSE,
    progress=False
)

frames_list = []
for ticker in TICKERS:
    yf_df = yf_batch.xs(ticker, axis=1, level=1).copy()
    yf_df = yf_df.dropna(how='all')
    if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
        yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
    if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
        yf_df['Close'] = yf_df['Adj Close']
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in yf_df.columns:
            yf_df[c] = np.nan
    df = yf_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df['Ticker'] = ticker
    df = df.reset_index().set_index(['Date', 'Ticker'])
    frames_list.append(df)

prices_batch = pd.concat(frames_list)
prices_batch = prices_batch[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
prices_batch = prices_batch.sort_index()

# Single iteratively
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
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in yf_df.columns:
            yf_df[c] = np.nan
    df = yf_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    df['Ticker'] = ticker
    df = df.reset_index().set_index(['Date', 'Ticker'])
    frames_single.append(df)

prices_single = pd.concat(frames_single)
prices_single = prices_single[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
prices_single = prices_single.sort_index()

print("Equals?", prices_single.equals(prices_batch))
if not prices_single.equals(prices_batch):
    print("Different items:")
    diff = prices_single.compare(prices_batch)
    print(diff.head())
