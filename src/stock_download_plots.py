#!/usr/bin/env python
# coding: utf-8

# # Stock Download and Matplotlib Plots
#
# This notebook downloads daily adjusted close data from Yahoo Finance for the 10 hackathon stocks and visualizes them with matplotlib.

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'widget')
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Universe and date range
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'V', 'JNJ', 'BRK-B']
START = '2017-01-01'
END = '2025-12-31'

# Download adjusted OHLCV (auto_adjust=True)
raw = yf.download(
    tickers=TICKERS,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
    group_by='ticker'
)

# Build close-price table (index=date, columns=tickers)
close_df = pd.DataFrame(index=raw.index)
for t in TICKERS:
    if (t, 'Close') in raw.columns:
        close_df[t] = raw[(t, 'Close')]

close_df = close_df.dropna(how='all')

print(f'Stocks downloaded: {close_df.shape[1]} / {len(TICKERS)}')
print(f'Date range: {close_df.index.min().date()} to {close_df.index.max().date()}')
close_df.tail()


# In[ ]:


# Plot 1: one subplot per stock
n = len(close_df.columns)
fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)

if n == 1:
    axes = [axes]

for ax, ticker in zip(axes, close_df.columns):
    ax.plot(close_df.index, close_df[ticker], linewidth=1.3)
    ax.set_title(f'{ticker} Adjusted Close', fontsize=10)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()

# Plot 2: normalized performance comparison (start = 100)
norm_df = close_df.div(close_df.iloc[0]).mul(100)

plt.figure(figsize=(14, 6))
for ticker in norm_df.columns:
    plt.plot(norm_df.index, norm_df[ticker], label=ticker, linewidth=1.5)

plt.title('Normalized Price Comparison (Start = 100)')
plt.xlabel('Date')
plt.ylabel('Normalized Price')
plt.grid(alpha=0.3)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
