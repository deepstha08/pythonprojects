# Create a ready-to-run Python script that fetches BTC-USD data from the internet (via yfinance),
# engineers technical features, trains a time-series-aware classifier,
# and prints a "RISE" / "FALL" prediction with probabilities and basic evaluation.
#
# The file will be saved at /mnt/data/btc_signal.py for download.

script = r'''#!/usr/bin/env python3
"""
BTC Direction Predictor (Rise/Fall)

What it does
------------
1) Downloads BTC-USD historical data from Yahoo Finance (via yfinance).
2) Builds common technical indicators (SMA/EMA/RSI/MACD/Volatility/Returns).
3) Trains a simple, time-series-aware classifier (RandomForest by default).
4) Outputs today's (last bar's) prediction: "RISE" or "FALL" with probability.
5) Prints recent model accuracy on a holdout (last 20% of samples).

Quick start
-----------
pip install -U yfinance pandas numpy scikit-learn
python btc_signal.py --interval 1h --lookback_days 180
python btc_signal.py --interval 1d --lookback_days 720

Notes
-----
- This is an educational baseline. It is **not** financial advice.
- Do not use this in isolation for live trading.
- Deterministic CV is used (no shuffling) to respect time order.

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

# yfinance is a lightweight data source without keys
try:
    import yfinance as yf
except Exception as e:
    print("yfinance is required. Install with: pip install yfinance", file=sys.stderr)
    raise

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ------------------------
# Technical Indicators
# ------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    # Annualized volatility approximation for daily/1h; we keep it windowed std
    return returns.rolling(window).std()

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-9)

# ------------------------
# Feature Engineering
# ------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    high = df['High']
    low = df['Low']

    # Returns
    df['ret_1'] = close.pct_change()
    df['ret_3'] = close.pct_change(3)
    df['ret_5'] = close.pct_change(5)

    # Trend/MA
    df['sma_10'] = sma(close, 10)
    df['sma_20'] = sma(close, 20)
    df['ema_21'] = ema(close, 21)
    df['ema_50'] = ema(close, 50)
    df['ma_cross'] = df['ema_21'] - df['ema_50']

    # RSI / MACD
    df['rsi_14'] = rsi(close, 14)
    macd_line, signal_line = macd(close, 12, 26, 9)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Volatility & range
    df['hl_range'] = (high - low) / close.shift(1)
    df['vol_20'] = realized_vol(df['ret_1'], 20)
    df['z_close_20'] = zscore(close, 20)

    # Target: next bar up/down
    df['future_ret_1'] = close.pct_change().shift(-1)
    df['target_up'] = (df['future_ret_1'] > 0).astype(int)

    df = df.dropna().copy()
    return df

# ------------------------
# Data Download
# ------------------------
def load_data(interval: str, lookback_days: int) -> pd.DataFrame:
    symbol = "BTC-USD"
    period = f"{lookback_days}d"
    # yfinance supports intervals: 1m, 2m, 5m, 15m, 30m, 60m/1h, 90m, 1d, 5d, 1wk, 1mo
    yf_interval = interval
    print(f"Downloading {symbol} from Yahoo Finance: period={period}, interval={yf_interval}")
    df = yf.download(symbol, period=period, interval=yf_interval, progress=False)
    if df.empty:
        raise RuntimeError("Downloaded dataframe is empty. Try increasing lookback_days or using a higher interval (e.g., 1h or 1d).")
    df = df.rename(columns=str.title)
    return df

# ------------------------
# Modeling
# ------------------------
@dataclass
class ModelResult:
    last_time: pd.Timestamp
    last_close: float
    proba_up: float
    prediction: str
    test_accuracy: float
    test_auc: float

def fit_and_predict(df: pd.DataFrame) -> ModelResult:
    features = [
        'ret_1','ret_3','ret_5',
        'sma_10','sma_20','ema_21','ema_50','ma_cross',
        'rsi_14','macd','macd_signal','macd_hist',
        'hl_range','vol_20','z_close_20'
    ]

    X = df[features]
    y = df['target_up']

    # Time-series split: last 20% test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = float('nan')

    # Last bar prediction
    last_row = X.iloc[[-1]]
    last_proba_up = float(clf.predict_proba(last_row)[:,1][0])
    prediction = "RISE" if last_proba_up >= 0.5 else "FALL"

    return ModelResult(
        last_time=df.index[-1],
        last_close=float(df['Close'].iloc[-1]),
        proba_up=last_proba_up,
        prediction=prediction,
        test_accuracy=acc,
        test_auc=auc
    )

# ------------------------
# Support / Resistance (simple)
# ------------------------
def support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
    # Very simple: recent swing high/low over a window
    recent = df.tail(lookback)
    support = float(recent['Low'].min())
    resistance = float(recent['High'].max())
    return support, resistance

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, default="1h",
                        help="Data interval (e.g., 1h, 1d, 30m, 15m).")
    parser.add_argument("--lookback_days", type=int, default=180,
                        help="How many days of history to download.")
    parser.add_argument("--sr_lookback", type=int, default=120,
                        help="Bars to use for support/resistance scan.")
    args = parser.parse_args()

    raw = load_data(args.interval, args.lookback_days)
    df = make_features(raw)

    if len(df) < 200:
        print(f"Warning: Only {len(df)} rows after feature engineering. Consider increasing lookback_days.", file=sys.stderr)

    result = fit_and_predict(df)
    support, resistance = support_resistance(raw, lookback=min(args.sr_lookback, len(raw)))

    print("\n=== BTC Direction Prediction ===")
    print(f"Last Bar Time         : {result.last_time}")
    print(f"Last Close (BTC-USD)  : {result.last_close:,.2f}")
    print(f"Prediction            : {result.prediction}")
    print(f"Probability (RISE)    : {result.proba_up:.3f}")
    print("\n=== Recent Out-of-Sample Performance (holdout) ===")
    print(f"Accuracy              : {result.test_accuracy:.3f}")
    print(f"AUC                   : {result.test_auc:.3f}")
    print("\n=== Context (not signals) ===")
    print(f"Recent Support (min)  : {support:,.2f}")
    print(f"Recent Resistance(max): {resistance:,.2f}")
    print("\nDISCLAIMER: Educational example. Not financial advice. "
          "Past performance is not indicative of future results.")

if __name__ == "__main__":
    main()
'''

# Write the script to the sandbox
path = "btc_signal.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(script)

print(f"Saved: {path}")
