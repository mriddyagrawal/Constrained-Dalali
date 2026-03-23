import yfinance as yf
import pandas as pd
import time

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START = '2017-01-01'
END = '2025-12-31'
USE_ADJ_CLOSE = True

# Method 1: Iterative
start_time = time.time()
for ticker in TICKERS:
    yf_df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=USE_ADJ_CLOSE,
        progress=False
    )
iterative_time = time.time() - start_time
print(f"Iterative download time: {iterative_time:.4f} seconds")

# Method 2: Batch
start_time = time.time()
yf_df_batch = yf.download(
    TICKERS,
    start=START,
    end=END,
    auto_adjust=USE_ADJ_CLOSE,
    progress=False
)
batch_time = time.time() - start_time
print(f"Batch download time: {batch_time:.4f} seconds")
