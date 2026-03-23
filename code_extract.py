import yfinance as yf
import pandas as pd
import numpy as np
import os
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START = '2017-01-01'
END = '2025-12-31'
USE_ADJ_CLOSE = True
keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

def get_baseline():
    frames = []
    for ticker in TICKERS:
        yf_df = yf.download(
            ticker,
            start=START,
            end=END,
            auto_adjust=USE_ADJ_CLOSE,
            progress=False
        )
        if yf_df.empty:
            continue
        if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
            yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
        if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
            yf_df['Close'] = yf_df['Adj Close']
        for c in keep_cols:
            if c not in yf_df.columns:
                yf_df[c] = np.nan
        df = yf_df[keep_cols].copy()
        df.index = pd.to_datetime(df.index)

        if df.empty:
            continue
        df.index.name = 'Date'
        df['Ticker'] = ticker
        df = df.reset_index().set_index(['Date', 'Ticker'])
        frames.append(df)
    prices = pd.concat(frames) if frames else pd.DataFrame(columns=keep_cols)
    prices = prices[keep_cols].copy()
    prices = prices.sort_index()
    return prices

def get_modified():
    frames = []
    tickers_to_download = TICKERS.copy()
    yf_batch = yf.download(
        tickers_to_download,
        start=START,
        end=END,
        auto_adjust=USE_ADJ_CLOSE,
        progress=False
    )
    for ticker in tickers_to_download:
        if yf_batch.empty:
            continue
        if isinstance(yf_batch.columns, pd.MultiIndex):
            try:
                yf_df = yf_batch.xs(ticker, axis=1, level=1).copy()
            except KeyError:
                continue
        else:
            yf_df = yf_batch.copy()
        yf_df = yf_df.dropna(how='all')
        if yf_df.empty:
            continue
        if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
            yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]
        if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
            yf_df['Close'] = yf_df['Adj Close']
        for c in keep_cols:
            if c not in yf_df.columns:
                yf_df[c] = np.nan
        df = yf_df[keep_cols].copy()
        df.index = pd.to_datetime(df.index)
        if df.empty:
            continue
        df.index.name = 'Date'
        df['Ticker'] = ticker
        df = df.reset_index().set_index(['Date', 'Ticker'])
        frames.append(df)
    prices = pd.concat(frames) if frames else pd.DataFrame(columns=keep_cols)
    prices = prices[keep_cols].copy()
    prices = prices.sort_index()
    return prices

b = get_baseline()
m = get_modified()

print("Equals?", b.equals(m))
if not b.equals(m):
    diff = b.compare(m)
    print("Diff:")
    print(diff.head())
    print("Shape b:", b.shape, "Shape m:", m.shape)
