import yfinance as yf
import pandas as pd

# Multiple
batch_df = yf.download(['AAPL', 'MSFT'], start='2020-01-01', end='2020-01-10', progress=False)
if isinstance(batch_df.columns, pd.MultiIndex):
    aapl = batch_df.xs('AAPL', axis=1, level=1)
    print("Columns:", aapl.columns)
