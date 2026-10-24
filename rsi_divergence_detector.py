import argparse
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


def fetch_ohlc(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False, actions=False)
    if df.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level="Ticker", drop_level=False)

    if "Close" not in df.columns:
        raise ValueError(f"Yahoo Finance did not return a Close column for {symbol}")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _to_scalar(value) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def is_pivot_low(series: pd.Series, idx: int, lookback: int = 2) -> bool:
    if idx < lookback or idx >= len(series) - lookback:
        return False

    value = series.iloc[idx]
    if pd.isna(value):
        return False

    window = series.iloc[idx - lookback : idx + lookback + 1]
    value_scalar = _to_scalar(value)
    prev_scalar = _to_scalar(series.iloc[idx - 1])
    next_scalar = _to_scalar(series.iloc[idx + 1])
    window_min = float(window.min())

    return value_scalar == window_min and value_scalar < prev_scalar and value_scalar <= next_scalar


def is_pivot_high(series: pd.Series, idx: int, lookback: int = 2) -> bool:
    if idx < lookback or idx >= len(series) - lookback:
        return False

    value = series.iloc[idx]
    if pd.isna(value):
        return False

    window = series.iloc[idx - lookback : idx + lookback + 1]
    value_scalar = _to_scalar(value)
    prev_scalar = _to_scalar(series.iloc[idx - 1])
    next_scalar = _to_scalar(series.iloc[idx + 1])
    window_max = float(window.max())

    return value_scalar == window_max and value_scalar > prev_scalar and value_scalar >= next_scalar


def detect_divergences(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 2,
    min_separation: int = 3,
) -> Tuple[pd.DataFrame, List[dict], List[dict]]:
    close = df["Close"].astype(float)
    rsi = calculate_rsi(close, period=rsi_period)
    df = df.copy()
    df["RSI"] = rsi

    bullish = []
    bearish = []

    for idx in range(lookback, len(close) - lookback):
        if not is_pivot_low(close, idx, lookback=lookback):
            continue
        if not is_pivot_low(rsi, idx, lookback=lookback):
            continue

        prev_idx = idx - min_separation
        if prev_idx < lookback:
            continue
        if not is_pivot_low(close, prev_idx, lookback=lookback):
            continue
        if not is_pivot_low(rsi, prev_idx, lookback=lookback):
            continue

        current_price = close.iloc[idx]
        prev_price = close.iloc[prev_idx]
        current_rsi = rsi.iloc[idx]
        prev_rsi = rsi.iloc[prev_idx]

        if current_price < prev_price and current_rsi > prev_rsi:
            bullish.append(
                {
                    "type": "bullish",
                    "start_idx": prev_idx,
                    "end_idx": idx,
                    "start_date": df.index[prev_idx],
                    "end_date": df.index[idx],
                    "start_price": prev_price,
                    "end_price": current_price,
                    "start_rsi": prev_rsi,
                    "end_rsi": current_rsi,
                    "prev_price": prev_price,
                    "prev_rsi": prev_rsi,
                }
            )
        elif current_price > prev_price and current_rsi < prev_rsi:
            bearish.append(
                {
                    "type": "bearish",
                    "start_idx": prev_idx,
                    "end_idx": idx,
                    "start_date": df.index[prev_idx],
                    "end_date": df.index[idx],
                    "start_price": prev_price,
                    "end_price": current_price,
                    "start_rsi": prev_rsi,
                    "end_rsi": current_rsi,
                    "prev_price": prev_price,
                    "prev_rsi": prev_rsi,
                }
            )

    return df, bullish, bearish


def plot_divergences(
    df: pd.DataFrame,
    bullish: List[dict],
    bearish: List[dict],
    symbol: str,
    output_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    close = df["Close"]
    rsi = df["RSI"]

    axes[0].plot(df.index, close, color="black", linewidth=1.2, label="Close")
    if bullish:
        for signal in bullish:
            axes[0].plot(
                [signal["start_date"], signal["end_date"]],
                [signal["start_price"], signal["end_price"]],
                color="green",
                linewidth=1.8,
                alpha=0.8,
            )
            axes[0].scatter([signal["start_date"], signal["end_date"]], [signal["start_price"], signal["end_price"]], marker="^", color="green", s=110, zorder=5)
        axes[0].scatter([], [], marker="^", color="green", s=110, label="Bullish Divergence")
    if bearish:
        for signal in bearish:
            axes[0].plot(
                [signal["start_date"], signal["end_date"]],
                [signal["start_price"], signal["end_price"]],
                color="red",
                linewidth=1.8,
                alpha=0.8,
            )
            axes[0].scatter([signal["start_date"], signal["end_date"]], [signal["start_price"], signal["end_price"]], marker="v", color="red", s=110, zorder=5)
        axes[0].scatter([], [], marker="v", color="red", s=110, label="Bearish Divergence")
    axes[0].set_title(f"{symbol} Price with RSI Divergences")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(df.index, rsi, color="tab:blue", linewidth=1.2, label="RSI (14)")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=1, alpha=0.7)
    if bullish:
        for signal in bullish:
            axes[1].plot(
                [signal["start_date"], signal["end_date"]],
                [signal["start_rsi"], signal["end_rsi"]],
                color="green",
                linewidth=2.0,
                alpha=0.85,
            )
            axes[1].scatter([signal["start_date"], signal["end_date"]], [signal["start_rsi"], signal["end_rsi"]], marker="^", color="green", s=110, zorder=5)
    if bearish:
        for signal in bearish:
            axes[1].plot(
                [signal["start_date"], signal["end_date"]],
                [signal["start_rsi"], signal["end_rsi"]],
                color="red",
                linewidth=2.0,
                alpha=0.85,
            )
            axes[1].scatter([signal["start_date"], signal["end_date"]], [signal["start_rsi"], signal["end_rsi"]], marker="v", color="red", s=110, zorder=5)
    axes[1].set_ylabel("RSI")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect RSI bullish and bearish divergences for a ticker")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="2y", help="Yahoo Finance period, e.g. 2y or 1y")
    parser.add_argument("--interval", default="1d", help="Yahoo Finance interval, e.g. 1d or 1wk")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=2)
    parser.add_argument("--output", default=None, help="Optional output image path")
    args = parser.parse_args()

    df, bullish, bearish = detect_divergences(
        fetch_ohlc(args.symbol, period=args.period, interval=args.interval),
        rsi_period=args.rsi_period,
        lookback=args.lookback,
    )

    print(f"Detected {len(bullish)} bullish divergence(s) and {len(bearish)} bearish divergence(s) for {args.symbol}.")
    for item in bullish:
        print(f"Bullish: {item['start_date'].strftime('%Y-%m-%d')} -> {item['end_date'].strftime('%Y-%m-%d')} | price={item['start_price']:.2f}->{item['end_price']:.2f}, rsi={item['start_rsi']:.2f}->{item['end_rsi']:.2f}")
    for item in bearish:
        print(f"Bearish: {item['start_date'].strftime('%Y-%m-%d')} -> {item['end_date'].strftime('%Y-%m-%d')} | price={item['start_price']:.2f}->{item['end_price']:.2f}, rsi={item['start_rsi']:.2f}->{item['end_rsi']:.2f}")

    plot_divergences(df, bullish, bearish, args.symbol, output_path=args.output)


if __name__ == "__main__":
    main()
