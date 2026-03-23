import json

with open("momentum_strategy.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "loaded_yf = 0" in source and "yfinance fallback" in source:

            new_source = """import os

print('Loading daily OHLCV data (local data folder first, then yfinance fallback)...')
DATA_DIR = 'data'
keep_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

frames = []
loaded_local = 0
loaded_yf = 0
missing_all = []

# First pass: Check local files and build a list of missing tickers for yfinance
tickers_to_download = []
for ticker in TICKERS:
    local_path = os.path.join(DATA_DIR, f'{ticker}.csv')

    try:
        if os.path.exists(local_path):
            print(f'  Processing {ticker}...', end=' ', flush=True)
            df = pd.read_csv(local_path)

            # Normalize common CSV formats
            if 'Date' not in df.columns and len(df.columns) > 0:
                first_col = df.columns[0]
                if str(first_col).startswith('Unnamed') or first_col == 'index':
                    df = df.rename(columns={first_col: 'Date'})

            if 'Date' not in df.columns:
                raise ValueError(f'No Date column in {local_path}')

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()

            # Prefer adjusted close to avoid split-driven return spikes
            if USE_ADJ_CLOSE and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']

            for c in keep_cols:
                if c not in df.columns:
                    df[c] = np.nan

            df = df[keep_cols].copy()
            df = df.loc[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]

            if df.empty:
                missing_all.append(ticker)
                print(f'Empty (LOCAL)')
                continue

            df.index.name = 'Date'
            df['Ticker'] = ticker
            df = df.reset_index().set_index(['Date', 'Ticker'])
            frames.append(df)
            loaded_local += 1
            print(f'OK ({len(df)} rows, LOCAL)')
        else:
            tickers_to_download.append(ticker)

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError) as e:
        missing_all.append(ticker)
        err_type = type(e).__name__
        print(f'FAILED {ticker}: {err_type}')

# Second pass: Download missing tickers from yfinance in batch
if tickers_to_download:
    print(f'  Downloading batch from yfinance: {tickers_to_download}...')
    try:
        yf_batch = yf.download(
            tickers_to_download,
            start=START,
            end=END,
            auto_adjust=USE_ADJ_CLOSE,
            progress=False
        )

        for ticker in tickers_to_download:
            print(f'  Processing {ticker}...', end=' ', flush=True)
            if yf_batch.empty:
                missing_all.append(ticker)
                print('Missing (local+yfinance)')
                continue

            # Extract single ticker's data
            if isinstance(yf_batch.columns, pd.MultiIndex):
                # We have multiple tickers in batch
                try:
                    yf_df = yf_batch.xs(ticker, axis=1, level=1).copy()
                except KeyError:
                    missing_all.append(ticker)
                    print('Missing (local+yfinance)')
                    continue
            else:
                # Only 1 ticker in batch, single level index
                yf_df = yf_batch.copy()

            # yfinance returns an empty dataframe if the ticker data is fully missing,
            # or a dataframe full of NaNs. Let's drop completely NaN rows to check emptiness
            yf_df = yf_df.dropna(how='all')

            if yf_df.empty:
                missing_all.append(ticker)
                print('Missing (local+yfinance)')
                continue

            if hasattr(yf_df.columns, 'nlevels') and yf_df.columns.nlevels > 1:
                yf_df.columns = [c[0] if isinstance(c, tuple) else c for c in yf_df.columns]

            # If not auto-adjusting, still allow adjusted close override if available
            if USE_ADJ_CLOSE and 'Adj Close' in yf_df.columns:
                yf_df['Close'] = yf_df['Adj Close']

            for c in keep_cols:
                if c not in yf_df.columns:
                    yf_df[c] = np.nan

            df = yf_df[keep_cols].copy()
            df.index = pd.to_datetime(df.index)

            if df.empty:
                missing_all.append(ticker)
                print(f'Empty (YF)')
                continue

            df.index.name = 'Date'
            df['Ticker'] = ticker
            df = df.reset_index().set_index(['Date', 'Ticker'])
            frames.append(df)
            loaded_yf += 1
            print(f'OK ({len(df)} rows, YF)')

    except Exception as e:
        for ticker in tickers_to_download:
            missing_all.append(ticker)
            print(f'FAILED {ticker}: {type(e).__name__}')

prices = pd.concat(frames) if frames else pd.DataFrame(columns=keep_cols)
prices = prices[keep_cols].copy()
prices = prices.sort_index()

if len(prices) == 0:
    raise ValueError('No data loaded from local data folder or yfinance.')

print(f'\\nLoaded from local CSVs: {loaded_local}')
print(f'Loaded from yfinance : {loaded_yf}')
if missing_all:
    print(f'Missing tickers      : {sorted(set(missing_all))}')

print(f'\\nShape: {prices.shape}')
print(f'Date range: {prices.index.get_level_values("Date").min()} → {prices.index.get_level_values("Date").max()}')
print(f'Tickers present: {sorted(prices.index.get_level_values("Ticker").unique().tolist())}')"""

            # Update the cell source ensuring line breaks
            cell["source"] = [line + ("\n" if i < len(new_source.split('\n')) - 1 else "") for i, line in enumerate(new_source.split('\n'))]

with open("momentum_strategy.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
