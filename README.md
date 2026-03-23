# ML Momentum Strategy Project

This project implements a machine-learning based long-only momentum trading strategy in Python.

## Overview
- Universe: 10 major US stocks
- Models: Logistic Regression, Random Forest, XGBoost (Soft Voting Ensemble)
- Data: Daily OHLCV data fetched via yfinance
- Backtest Period: 2017-2025

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the strategy script:
   ```bash
   python src/momentum_strategy.py
   ```

## Project Structure
- `src/momentum_strategy.py`: The main script that runs the data ingestion, feature engineering, model training, and backtesting.
- `notebooks/momentum_strategy.ipynb`: The Jupyter Notebook version.
- `src/stock_download_plots.py`: Script to download and visualize stock price data.
- `requirements.txt`: Python package dependencies.
