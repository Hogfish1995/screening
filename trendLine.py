import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


def fetch_stock_data(
    symbol: str,
    timeframe: str = "Day",
    duration: str = "1 Year"
) -> pd.DataFrame:

    """
    Fetch OHLCV data from Yahoo Finance with extra historical warmup buffer.
    Supports intraday (1 Hour, 4 Hours), daily (Day), weekly (Week), and monthly (Month).
    """

    fetch_period_map = {
        "1 Hour": {
            "1 Month": "2mo",
            "3 Months": "6mo",
            "6 Months": "1y",
            "1 Year": "2y",
            "2 Years": "2y",
            "Max": "2y",
        },
        "4 Hours": {
            "1 Month": "3mo",
            "3 Months": "6mo",
            "6 Months": "1y",
            "1 Year": "2y",
            "2 Years": "2y",
            "Max": "2y",
        },
        "Day": {
            "1 Month": "6mo",
            "3 Months": "1y",
            "6 Months": "2y",
            "1 Year": "2y",
            "2 Years": "4y",
            "5 Years": "7y",
            "Max": "max",
        },
        "Week": {
            "1 Month": "2y",
            "3 Months": "2y",
            "6 Months": "5y",
            "1 Year": "5y",
            "2 Years": "7y",
            "5 Years": "10y",
            "Max": "max",
        },
        "Month": {
            "1 Month": "max",
            "3 Months": "max",
            "6 Months": "max",
            "1 Year": "max",
            "2 Years": "max",
            "5 Years": "max",
            "Max": "max",
        },
    }

    interval_map = {
        "1 Hour": "1h",
        "4 Hours": "1h",
        "Day": "1d",
        "Week": "1wk",
        "Month": "1mo",
    }

    tf_dict = fetch_period_map.get(
        timeframe,
        fetch_period_map["Day"]
    )

    period_str = tf_dict.get(
        duration,
        "2y" if timeframe in ["1 Hour", "4 Hours"] else "2y"
    )

    interval_str = interval_map.get(
        timeframe,
        "1d"
    )

    try:
        import requests
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        ticker = yf.Ticker(symbol, session=session)
    except Exception:
        ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period_str,
        interval=interval_str,
        auto_adjust=False,
        actions=False
    )

    if df.empty:
        raise ValueError(
            f"No data returned for symbol '{symbol}' "
            f"with duration '{duration}' "
            f"and timeframe '{timeframe}'."
        )

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(
                symbol,
                axis=1,
                level="Ticker",
                drop_level=False
            )
        except Exception:
            pass

    df = df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna().copy()

    df.index = pd.to_datetime(df.index)

    if timeframe == "4 Hours":
        df = (
            df.resample("4h", origin="start")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna()
            .copy()
        )

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    df["SMA20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["SMA200"] = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    # ---------------------------------------------------------
    # TD Sequential
    # ---------------------------------------------------------

    df = calculate_td_sequential(df)

    # ---------------------------------------------------------
    # Trim to requested display duration
    # ---------------------------------------------------------

    if duration != "Max" and len(df) > 0:

        offset_map = {
            "1 Month": pd.DateOffset(months=1),
            "3 Months": pd.DateOffset(months=3),
            "6 Months": pd.DateOffset(months=6),
            "1 Year": pd.DateOffset(years=1),
            "2 Years": pd.DateOffset(years=2),
            "5 Years": pd.DateOffset(years=5),
        }

        if duration in offset_map:

            cutoff_date = (
                df.index[-1]
                - offset_map[duration]
            )

            display_df = df[
                df.index >= cutoff_date
            ].copy()

            if len(display_df) >= 10:
                return display_df

    return df


def calculate_td_sequential(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Computes Tom DeMark Sequential TD Setup 9.

    Buy Setup:
        Close < Close 4 bars ago
        for 9 consecutive bars

    Sell Setup:
        Close > Close 4 bars ago
        for 9 consecutive bars
    """

    closes = df["Close"].values

    n = len(closes)

    td_buy = np.zeros(
        n,
        dtype=int
    )

    td_sell = np.zeros(
        n,
        dtype=int
    )

    b_count = 0
    s_count = 0

    for i in range(n):

        if i >= 4:

            if closes[i] < closes[i - 4]:

                b_count += 1
                s_count = 0

            elif closes[i] > closes[i - 4]:

                s_count += 1
                b_count = 0

            else:

                b_count = 0
                s_count = 0

        td_buy[i] = b_count
        td_sell[i] = s_count

    df["TD_Buy"] = td_buy
    df["TD_Sell"] = td_sell

    return df


def find_pivots(
    df: pd.DataFrame,
    window: int = 5
):

    """
    Identify local maxima and local minima.
    """

    highs = df["High"].values
    lows = df["Low"].values

    pivot_high_idx = scipy.signal.argrelextrema(
        highs,
        np.greater_equal,
        order=window
    )[0]

    pivot_low_idx = scipy.signal.argrelextrema(
        lows,
        np.less_equal,
        order=window
    )[0]

    pivot_highs = [
        (i, highs[i])
        for i in pivot_high_idx
    ]

    pivot_lows = [
        (i, lows[i])
        for i in pivot_low_idx
    ]

    return pivot_highs, pivot_lows


def fit_best_trendline(
    df: pd.DataFrame,
    pivots: list,
    is_support: bool = True,
    max_lines: int = 2,
    max_breach_pct: float = 0.015
):

    """
    Fit independent automatic trendlines
    connecting pivot points.
    """

    lines = []

    n = len(pivots)

    if n < 2:
        return lines

    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    candidate_lines = []

    for i in range(n):

        for j in range(i + 1, n):

            x1, y1 = pivots[i]
            x2, y2 = pivots[j]

            min_dist = max(
                3,
                int(len(df) * 0.03)
            )

            if (x2 - x1) < min_dist:
                continue

            slope = (
                y2 - y1
            ) / (
                x2 - x1
            )

            intercept = (
                y1 - slope * x1
            )

            x_range = np.arange(
                x1,
                len(df)
            )

            line_vals = (
                slope * x_range
                + intercept
            )

            if is_support:

                actual_lows = lows[x1:]

                breaches = (
                    actual_lows
                    <
                    (
                        line_vals
                        * (1 - max_breach_pct)
                    )
                )

            else:

                actual_highs = highs[x1:]

                breaches = (
                    actual_highs
                    >
                    (
                        line_vals
                        * (1 + max_breach_pct)
                    )
                )

            num_breaches = np.sum(
                breaches
            )

            span = x2 - x1

            if is_support:

                diffs = np.abs(
                    lows[x1:j + 1]
                    -
                    (
                        slope
                        * np.arange(x1, j + 1)
                        + intercept
                    )
                )

            else:

                diffs = np.abs(
                    highs[x1:j + 1]
                    -
                    (
                        slope
                        * np.arange(x1, j + 1)
                        + intercept
                    )
                )

            tolerance = (
                (y1 + y2) / 2
                * 0.012
            )

            touches = np.sum(
                diffs <= tolerance
            )

            score = (
                span * 1.5
                + touches * 10
                - num_breaches * 50
                + x2 * 0.5
            )

            if num_breaches <= max(
                2,
                int(
                    (len(df) - x1)
                    * 0.05
                )
            ):

                candidate_lines.append(
                    {
                        "score": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "slope": slope,
                        "intercept": intercept,
                        "date1": dates[x1],
                        "date2": dates[-1],
                        "y_end": (
                            slope
                            * (len(df) - 1)
                            + intercept
                        ),
                    }
                )

    candidate_lines.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []

    for cand in candidate_lines:

        overlap = False

        for sel in selected:

            if (
                abs(
                    cand["slope"]
                    - sel["slope"]
                ) < 1e-4
                or
                abs(
                    cand["y_end"]
                    - sel["y_end"]
                )
                /
                max(
                    1,
                    cand["y_end"]
                )
                < 0.02
            ):

                overlap = True
                break

        if not overlap:

            selected.append(cand)

            if len(selected) >= max_lines:
                break

    return selected


def fit_finviz_parallel_channel(
    df: pd.DataFrame,
    pivot_lows: list,
    pivot_highs: list
):

    """
    Fits a Finviz-style parallel trend channel.
    """

    if (
        len(pivot_lows) < 2
        and len(pivot_highs) < 2
    ):
        return None, None, None

    x = np.arange(
        len(df)
    )

    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    best_slope = None
    best_score = -1e9
    best_pair = None

    all_pivots = (
        pivot_lows
        if len(pivot_lows) >= len(pivot_highs)
        else pivot_highs
    )

    for i in range(
        len(all_pivots)
    ):

        for j in range(
            i + 1,
            len(all_pivots)
        ):

            x1, y1 = all_pivots[i]
            x2, y2 = all_pivots[j]

            if (
                x2 - x1
                <
                max(
                    5,
                    int(len(df) * 0.1)
                )
            ):
                continue

            slope = (
                y2 - y1
            ) / (
                x2 - x1
            )

            channel_low = (
                lows - slope * x
            )

            channel_high = (
                highs - slope * x
            )

            c_supp = np.percentile(
                channel_low,
                5
            )

            c_res = np.percentile(
                channel_high,
                95
            )

            width = (
                c_res - c_supp
            )

            if width <= 0:
                continue

            in_channel = (
                (channel_low >= c_supp * 0.98)
                &
                (channel_high <= c_res * 1.02)
            )

            coverage = np.mean(
                in_channel
            )

            score = (
                coverage * 100
                -
                (
                    width
                    /
                    np.mean(df["Close"])
                    * 20
                )
                +
                (
                    (x2 - x1)
                    * 0.1
                )
            )

            if score > best_score:

                best_score = score
                best_slope = slope
                best_pair = (
                    x1,
                    x2,
                    c_supp,
                    c_res
                )

    if best_slope is None:

        best_slope, _ = np.polyfit(
            x,
            df["Close"].values,
            1
        )

        channel_low = (
            lows - best_slope * x
        )

        channel_high = (
            highs - best_slope * x
        )

        c_supp = np.percentile(
            channel_low,
            5
        )

        c_res = np.percentile(
            channel_high,
            95
        )

        x1 = 0
        x2 = len(df) - 1

    else:

        x1, x2, c_supp, c_res = best_pair

    c_mid = (
        c_supp + c_res
    ) / 2.0

    support_line = {
        "date1": dates[x1],
        "y1": (
            best_slope * x1
            + c_supp
        ),
        "date2": dates[-1],
        "y_end": (
            best_slope
            * (len(df) - 1)
            + c_supp
        ),
    }

    resistance_line = {
        "date1": dates[x1],
        "y1": (
            best_slope * x1
            + c_res
        ),
        "date2": dates[-1],
        "y_end": (
            best_slope
            * (len(df) - 1)
            + c_res
        ),
    }

    midline = {
        "date1": dates[x1],
        "y1": (
            best_slope * x1
            + c_mid
        ),
        "date2": dates[-1],
        "y_end": (
            best_slope
            * (len(df) - 1)
            + c_mid
        ),
    }

    return (
        support_line,
        resistance_line,
        midline
    )


def create_candlestick_chart(
    symbol: str,
    df: pd.DataFrame,
    support_lines: list,
    resistance_lines: list,
    parallel_support: dict,
    parallel_resistance: dict,
    parallel_midline: dict,
    timeframe: str,
    duration: str,
    show_independent_pivots: bool = True,
    show_finviz_channel: bool = True,
    show_channel_midline: bool = True,
    show_smas: bool = True,
    show_td_9: bool = True,
    show_td_all_counts: bool = False,
    hide_non_trading: bool = True,
    use_log_scale: bool = False,
) -> go.Figure:

    """
    Create interactive Plotly candlestick chart.
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[
            0.83,
            0.17
        ],
        subplot_titles=(
            "",
            "Volume"
        ),
    )

    # ---------------------------------------------------------
    # Candlestick
    # ---------------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#00c805",
            decreasing_line_color="#ff5000",
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    if show_smas:

        if (
            "SMA20" in df.columns
            and not df["SMA20"].isna().all()
        ):

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA20"],
                    name="SMA 20",
                    line=dict(
                        color="#42A5F5",
                        width=1.5
                    ),
                ),
                row=1,
                col=1,
            )

        if (
            "SMA50" in df.columns
            and not df["SMA50"].isna().all()
        ):

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA50"],
                    name="SMA 50",
                    line=dict(
                        color="#EF5350",
                        width=1.5
                    ),
                ),
                row=1,
                col=1,
            )

        if (
            "SMA200" in df.columns
            and not df["SMA200"].isna().all()
        ):

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA200"],
                    name="SMA 200",
                    line=dict(
                        color="#FFCA28",
                        width=1.5
                    ),
                ),
                row=1,
                col=1,
            )

    # ---------------------------------------------------------
    # Independent pivot trendlines
    # ---------------------------------------------------------

    if show_independent_pivots:

        for idx, line in enumerate(
            support_lines
        ):

            fig.add_trace(
                go.Scatter(
                    x=[
                        line["date1"],
                        line["date2"]
                    ],
                    y=[
                        line["y1"],
                        line["y_end"]
                    ],
                    mode="lines+markers",
                    name=f"Pivot Support {idx + 1}",
                    line=dict(
                        color="#00E676",
                        width=2,
                        dash="dash"
                    ),
                    marker=dict(
                        size=6,
                        color="#00E676"
                    ),
                ),
                row=1,
                col=1,
            )

        for idx, line in enumerate(
            resistance_lines
        ):

            fig.add_trace(
                go.Scatter(
                    x=[
                        line["date1"],
                        line["date2"]
                    ],
                    y=[
                        line["y1"],
                        line["y_end"]
                    ],
                    mode="lines+markers",
                    name=f"Pivot Resistance {idx + 1}",
                    line=dict(
                        color="#FF5252",
                        width=2,
                        dash="dash"
                    ),
                    marker=dict(
                        size=6,
                        color="#FF5252"
                    ),
                ),
                row=1,
                col=1,
            )

    # ---------------------------------------------------------
    # Finviz parallel channel
    # ---------------------------------------------------------

    if show_finviz_channel:

        if parallel_support:

            d1_supp = parallel_support["date1"]
            d2_supp = parallel_support["date2"]
            idx1_supp = df.index.get_loc(d1_supp) if d1_supp in df.index else 0
            idx2_supp = df.index.get_loc(d2_supp) if d2_supp in df.index else len(df) - 1
            if idx1_supp > idx2_supp:
                idx1_supp, idx2_supp = idx2_supp, idx1_supp
            dates_supp = df.index[idx1_supp:idx2_supp + 1]
            y_supp_vals = np.linspace(
                parallel_support["y1"],
                parallel_support["y_end"],
                len(dates_supp)
            )

            fig.add_trace(
                go.Scatter(
                    x=dates_supp,
                    y=y_supp_vals,
                    mode="lines",
                    name="Finviz Support",
                    line=dict(
                        color="#00B0FF",
                        width=2.5
                    ),
                ),
                row=1,
                col=1,
            )

        if (
            parallel_midline
            and show_channel_midline
        ):

            d1_mid = parallel_midline["date1"]
            d2_mid = parallel_midline["date2"]
            idx1_mid = df.index.get_loc(d1_mid) if d1_mid in df.index else 0
            idx2_mid = df.index.get_loc(d2_mid) if d2_mid in df.index else len(df) - 1
            if idx1_mid > idx2_mid:
                idx1_mid, idx2_mid = idx2_mid, idx1_mid
            dates_mid = df.index[idx1_mid:idx2_mid + 1]
            y_mid_vals = np.linspace(
                parallel_midline["y1"],
                parallel_midline["y_end"],
                len(dates_mid)
            )

            fig.add_trace(
                go.Scatter(
                    x=dates_mid,
                    y=y_mid_vals,
                    mode="lines",
                    name="Finviz Midline (50%)",
                    line=dict(
                        color="#B388FF",
                        width=2,
                        dash="dot"
                    ),
                ),
                row=1,
                col=1,
            )

        if parallel_resistance:

            d1_res = parallel_resistance["date1"]
            d2_res = parallel_resistance["date2"]
            idx1_res = df.index.get_loc(d1_res) if d1_res in df.index else 0
            idx2_res = df.index.get_loc(d2_res) if d2_res in df.index else len(df) - 1
            if idx1_res > idx2_res:
                idx1_res, idx2_res = idx2_res, idx1_res
            dates_res = df.index[idx1_res:idx2_res + 1]
            y_res_vals = np.linspace(
                parallel_resistance["y1"],
                parallel_resistance["y_end"],
                len(dates_res)
            )

            fig.add_trace(
                go.Scatter(
                    x=dates_res,
                    y=y_res_vals,
                    mode="lines",
                    name="Finviz Resistance",
                    line=dict(
                        color="#E040FB",
                        width=2.5
                    ),
                ),
                row=1,
                col=1,
            )

    # ---------------------------------------------------------
    # TD Sequential
    # ---------------------------------------------------------

    if show_td_9:

        td_buy_9 = df[
            df["TD_Buy"] == 9
        ]

        if not td_buy_9.empty:

            fig.add_trace(
                go.Scatter(
                    x=td_buy_9.index,
                    y=td_buy_9["Low"] * 0.982,
                    mode="text",
                    text=[
                        "<b>9</b>"
                    ] * len(td_buy_9),
                    textposition="bottom center",
                    textfont=dict(
                        color="#00E676",
                        size=16
                    ),
                    name="TD Buy 9 (Bullish)",
                    hoverinfo="none",
                ),
                row=1,
                col=1,
            )

        td_sell_9 = df[
            df["TD_Sell"] == 9
        ]

        if not td_sell_9.empty:

            fig.add_trace(
                go.Scatter(
                    x=td_sell_9.index,
                    y=td_sell_9["High"] * 1.018,
                    mode="text",
                    text=[
                        "<b>9</b>"
                    ] * len(td_sell_9),
                    textposition="top center",
                    textfont=dict(
                        color="#FF5252",
                        size=16
                    ),
                    name="TD Sell 9 (Bearish)",
                    hoverinfo="none",
                ),
                row=1,
                col=1,
            )

        if show_td_all_counts:

            td_buy_all = df[
                (df["TD_Buy"] >= 1)
                &
                (df["TD_Buy"] < 9)
            ]

            if not td_buy_all.empty:

                fig.add_trace(
                    go.Scatter(
                        x=td_buy_all.index,
                        y=td_buy_all["Low"] * 0.988,
                        mode="text",
                        text=[
                            str(v)
                            for v
                            in td_buy_all["TD_Buy"]
                        ],
                        textposition="bottom center",
                        textfont=dict(
                            color="#81C784",
                            size=10
                        ),
                        name="TD Buy Setup (1-8)",
                        showlegend=False,
                        hoverinfo="none",
                    ),
                    row=1,
                    col=1,
                )

            td_sell_all = df[
                (df["TD_Sell"] >= 1)
                &
                (df["TD_Sell"] < 9)
            ]

            if not td_sell_all.empty:

                fig.add_trace(
                    go.Scatter(
                        x=td_sell_all.index,
                        y=td_sell_all["High"] * 1.012,
                        mode="text",
                        text=[
                            str(v)
                            for v
                            in td_sell_all["TD_Sell"]
                        ],
                        textposition="top center",
                        textfont=dict(
                            color="#E57373",
                            size=10
                        ),
                        name="TD Sell Setup (1-8)",
                        showlegend=False,
                        hoverinfo="none",
                    ),
                    row=1,
                    col=1,
                )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    colors = [
        "#00c805"
        if c >= o
        else "#ff5000"
        for c, o
        in zip(
            df["Close"],
            df["Open"]
        )
    ]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    # ---------------------------------------------------------
    # Hide non-trading days
    # ---------------------------------------------------------

    if (
        hide_non_trading
        and len(df) > 1
    ):

        if timeframe in ["1 Hour", "4 Hours"]:

            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour")
                ]
            )

        elif timeframe == "Day":

            all_dates = pd.date_range(
                start=df.index[0],
                end=df.index[-1],
                freq="D"
            )

            trading_dates_set = set(
                df.index.strftime(
                    "%Y-%m-%d"
                )
            )

            missing_dates = [
                d.strftime("%Y-%m-%d")
                for d in all_dates
                if d.strftime("%Y-%m-%d")
                not in trading_dates_set
            ]

            fig.update_xaxes(
                rangebreaks=[
                    dict(
                        values=missing_dates
                    )
                ]
            )

        elif timeframe in [
            "Week",
            "Month"
        ]:

            fig.update_xaxes(
                rangebreaks=[
                    dict(
                        bounds=[
                            "sat",
                            "mon"
                        ]
                    )
                ]
            )

    # ---------------------------------------------------------
    # Crosshair
    # ---------------------------------------------------------

    fig.update_xaxes(
        showspikes=True,
        spikecolor="rgba(255, 255, 255, 0.35)",
        spikethickness=1,
        spikedash="dash",
        spikemode="across",
        spikesnap="cursor",
        gridcolor="#2A2E39",
        tickfont=dict(
            color="#E0E0E0",
            size=12
        ),
    )

    fig.update_yaxes(
        showspikes=True,
        spikecolor="rgba(255, 255, 255, 0.35)",
        spikethickness=1,
        spikedash="dash",
        spikemode="across",
        spikesnap="cursor",
        gridcolor="#2A2E39",
        tickfont=dict(
            color="#E0E0E0",
            size=12
        ),
    )

    # ---------------------------------------------------------
    # Y axis
    # ---------------------------------------------------------

    y_type = (
        "log"
        if use_log_scale
        else "linear"
    )

    fig.update_yaxes(
        title_text="Price ($)",
        type=y_type,
        row=1,
        col=1,
        title_font=dict(
            color="#FFFFFF",
            size=13
        ),
    )

    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        title_font=dict(
            color="#FFFFFF",
            size=13
        ),
    )

    # ---------------------------------------------------------
    # Layout
    # ---------------------------------------------------------

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
        font=dict(
            color="#FFFFFF",
            size=13
        ),
        xaxis_rangeslider_visible=False,
        height=870,
        margin=dict(
            l=65,
            r=40,
            t=55,
            b=25
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color="#FFFFFF",
                size=12
            ),
            bgcolor="rgba(19,23,34,0.85)",
            bordercolor="#363C4E",
            borderwidth=1,
        ),
        hovermode="x",
        hoverlabel=dict(
            bgcolor="rgba(19, 23, 34, 0.95)",
            font_size=12,
            font_color="#FFFFFF",
            bordercolor="#363C4E",
            align="left",
        ),
    )

    fig.update_yaxes(
        title_text="Price ($)",
        type=y_type,
        row=1,
        col=1,
        gridcolor="#2A2E39",
        title_font=dict(
            color="#FFFFFF",
            size=13
        ),
        tickfont=dict(
            color="#E0E0E0",
            size=12
        ),
    )

    fig.update_yaxes(
        title_text="Volume",
        row=2,
        col=1,
        gridcolor="#2A2E39",
        title_font=dict(
            color="#FFFFFF",
            size=13
        ),
        tickfont=dict(
            color="#E0E0E0",
            size=12
        ),
    )

    fig.update_xaxes(
        gridcolor="#2A2E39",
        tickfont=dict(
            color="#E0E0E0",
            size=12
        ),
    )

    fig.for_each_annotation(
        lambda a: a.update(
            font=dict(
                color="#FFFFFF",
                size=13
            )
        )
    )

    return fig


def main():

    st.set_page_config(
        page_title="Auto Equity Trendline Viewer",
        layout="wide"
    )

    # ---------------------------------------------------------
    # CSS
    # ---------------------------------------------------------

    st.markdown(
        """
        <style>

        .stApp {
            background-color: #0E1117;
            color: #D1D4DC;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700;
            color: #00E676;
        }

        .chart-header-container {
            position: relative;
            margin-bottom: -45px;
            z-index: 10;
            padding-left: 40px;
            pointer-events: none;
        }

        .header-readout-text {
            font-size: 13px;
            font-weight: 600;
            color: #FFFFFF;
            background: rgba(19, 23, 34, 0.9);
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #363C4E;
            display: inline-block;
        }

        /* Hide ONLY floating tooltip boxes/text, NOT crosshair spikelines */
        div[data-testid="stPlotlyChart"] .hoverlayer .g-hovertext,
        div[data-testid="stPlotlyChart"] .hoverlayer .g-x-hover,
        div[data-testid="stPlotlyChart"] .hoverlayer .hovertext,
        div[data-testid="stPlotlyChart"] .hoverlayer rect {
            display: none !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }

        /* Ensure vertical & horizontal dotted crosshairs remain fully visible */
        div[data-testid="stPlotlyChart"] .spikelayer,
        div[data-testid="stPlotlyChart"] .spikeline,
        div[data-testid="stPlotlyChart"] .js-plotly-plot .spikelayer {
            display: block !important;
            opacity: 1 !important;
            visibility: visible !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    st.title(
        "📈 Automatic Equity Trendline & TD 9 Viewer"
    )

    st.markdown(
        "Select timeframe & duration, toggle "
        "**Finviz Channels**, "
        "**Independent Trendlines**, and "
        "**Tom DeMark TD Sequential 9 Signals**."
    )

    # ---------------------------------------------------------
    # Main controls
    # ---------------------------------------------------------

    c1, c2, c3 = st.columns(
        [2, 2, 2]
    )

    with c1:

        symbol = st.text_input(
            "Ticker Symbol",
            value="NVDA"
        ).strip().upper()

    with c2:

        timeframe = st.selectbox(
            "Timeframe (Interval)",
            [
                "1 Hour",
                "4 Hours",
                "Day",
                "Week",
                "Month"
            ],
            index=2
        )

    with c3:

        duration_options = (
            [
                "1 Month",
                "3 Months",
                "6 Months",
                "1 Year",
                "2 Years",
                "Max"
            ]
            if timeframe in ["1 Hour", "4 Hours"]
            else [
                "1 Month",
                "3 Months",
                "6 Months",
                "1 Year",
                "2 Years",
                "5 Years",
                "Max"
            ]
        )

        default_dur_idx = (
            0
            if timeframe in ["1 Hour", "4 Hours"]
            else 3
        )

        duration = st.selectbox(
            "Chart Duration",
            duration_options,
            index=(
                default_dur_idx
                if default_dur_idx < len(duration_options)
                else 0
            )
        )

    # ---------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------

    st.sidebar.header(
        "🎯 Trendline Overlay Options"
    )

    show_finviz_channel = (
        st.sidebar.checkbox(
            "Finviz Parallel Channel",
            value=True,
            help=(
                "Renders parallel upper "
                "& lower trend channel boundaries."
            ),
        )
    )

    show_channel_midline = (
        st.sidebar.checkbox(
            "Channel Midline (50%)",
            value=True,
            help=(
                "Renders the 50% midpoint "
                "line inside the channel."
            ),
        )
    )

    show_independent_pivots = (
        st.sidebar.checkbox(
            "Independent Extremum Fitting",
            value=False,
            help=(
                "Connects local swing highs "
                "and swing lows independently."
            ),
        )
    )

    st.sidebar.header(
        "⏱️ Tom DeMark Indicators"
    )

    show_td_9 = (
        st.sidebar.checkbox(
            "Tom DeMark TD Sequential (TD 9)",
            value=True,
            help=(
                "Highlights TD Buy 9 (green) "
                "and TD Sell 9 (red) reversal signals."
            ),
        )
    )

    show_td_all_counts = (
        st.sidebar.checkbox(
            "Show TD Setup Counts (1-8)",
            value=False,
            help=(
                "Displays intermediate setup "
                "numbers 1 to 8 above/below candles."
            ),
        )
    )

    st.sidebar.header(
        "⚙️ Display & Chart Settings"
    )

    show_smas = (
        st.sidebar.checkbox(
            "Overlay SMAs (20, 50, 200)",
            value=True
        )
    )

    hide_non_trading = (
        st.sidebar.checkbox(
            "Continuous Axis (Hide Weekends)",
            value=True
        )
    )

    use_log_scale = (
        st.sidebar.checkbox(
            "Logarithmic Y-Axis Scale",
            value=False
        )
    )

    pivot_window = (
        st.sidebar.slider(
            "Pivot Window Sensitivity",
            min_value=2,
            max_value=20,
            value=5
        )
    )

    # ---------------------------------------------------------
    # Fetch and draw
    # ---------------------------------------------------------

    if symbol:

        with st.spinner(
            f"Fetching {duration} "
            f"{timeframe} data for {symbol}..."
        ):

            try:

                df = fetch_stock_data(
                    symbol,
                    timeframe=timeframe,
                    duration=duration
                )

                pivot_highs, pivot_lows = (
                    find_pivots(
                        df,
                        window=pivot_window
                    )
                )

                support_lines = (
                    fit_best_trendline(
                        df,
                        pivot_lows,
                        is_support=True,
                        max_lines=2
                    )
                )

                resistance_lines = (
                    fit_best_trendline(
                        df,
                        pivot_highs,
                        is_support=False,
                        max_lines=2
                    )
                )

                (
                    p_supp,
                    p_res,
                    p_mid
                ) = fit_finviz_parallel_channel(
                    df,
                    pivot_lows,
                    pivot_highs
                )

                # -------------------------------------------------
                # Latest candle information
                # -------------------------------------------------

                latest_candle = df.iloc[-1]

                latest_date = (
                    df.index[-1]
                    .strftime("%Y-%m-%d %H:%M")
                    if timeframe in ["1 Hour", "4 Hours"]
                    else df.index[-1]
                    .strftime("%Y-%m-%d")
                )

                prev_close = (
                    df["Close"].iloc[-2]
                    if len(df) > 1
                    else latest_candle["Close"]
                )

                price_chg = (
                    latest_candle["Close"]
                    - prev_close
                )

                pct_chg = (
                    price_chg
                    / prev_close
                    * 100
                    if prev_close != 0
                    else 0
                )

                chg_color = (
                    "#00E676"
                    if price_chg >= 0
                    else "#FF5252"
                )

                sma20_str = (
                    f"${latest_candle['SMA20']:.2f}"
                    if (
                        "SMA20" in df.columns
                        and not pd.isna(latest_candle["SMA20"])
                    )
                    else None
                )

                sma50_str = (
                    f"${latest_candle['SMA50']:.2f}"
                    if (
                        "SMA50" in df.columns
                        and not pd.isna(latest_candle["SMA50"])
                    )
                    else None
                )

                sma200_str = (
                    f"${latest_candle['SMA200']:.2f}"
                    if (
                        "SMA200" in df.columns
                        and not pd.isna(latest_candle["SMA200"])
                    )
                    else None
                )

                supp_str = (
                    f"${p_supp['y_end']:.2f}"
                    if p_supp
                    else "N/A"
                )

                mid_str = (
                    f"${p_mid['y_end']:.2f}"
                    if (p_mid and show_channel_midline)
                    else None
                )

                res_str = (
                    f"${p_res['y_end']:.2f}"
                    if p_res
                    else "N/A"
                )

                # -------------------------------------------------
                # Header
                # -------------------------------------------------

                header_parts = [
                    f"<b>{symbol.upper()}</b> ({timeframe}, {duration})",
                    f"<b>DATE:</b> {latest_date}",
                    (
                        f"<b>O:</b> ${latest_candle['Open']:.2f} &nbsp;"
                        f"<b>H:</b> ${latest_candle['High']:.2f} &nbsp;"
                        f"<b>L:</b> ${latest_candle['Low']:.2f} &nbsp;"
                        f"<b>C:</b> ${latest_candle['Close']:.2f} &nbsp;"
                        f"<span style='color:{chg_color};'>({price_chg:+.2f}, {pct_chg:+.2f}%)</span>"
                    )
                ]

                if show_smas:
                    if sma20_str:
                        header_parts.append(
                            f"<b>SMA20:</b> <span style='color:#42A5F5;'>{sma20_str}</span>"
                        )
                    if sma50_str:
                        header_parts.append(
                            f"<b>SMA50:</b> <span style='color:#EF5350;'>{sma50_str}</span>"
                        )
                    if sma200_str:
                        header_parts.append(
                            f"<b>SMA200:</b> <span style='color:#FFCA28;'>{sma200_str}</span>"
                        )

                if p_supp:
                    header_parts.append(
                        f"<b>SUPP:</b> <span style='color:#00B0FF;'>{supp_str}</span>"
                    )

                if p_mid and show_channel_midline and mid_str:
                    header_parts.append(
                        f"<b>MID:</b> <span style='color:#B388FF;'>{mid_str}</span>"
                    )

                if p_res:
                    header_parts.append(
                        f"<b>RES:</b> <span style='color:#E040FB;'>{res_str}</span>"
                    )

                default_header_text = " &nbsp;|&nbsp; ".join(header_parts)

                st.markdown(
                    f"""
                    <div class="chart-header-container">
                        <div
                            id="live-header-readout"
                            class="header-readout-text"
                        >
                            {default_header_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # -------------------------------------------------
                # Create chart
                # -------------------------------------------------

                fig = create_candlestick_chart(
                    symbol=symbol,
                    df=df,
                    support_lines=support_lines,
                    resistance_lines=resistance_lines,
                    parallel_support=p_supp,
                    parallel_resistance=p_res,
                    parallel_midline=p_mid,
                    timeframe=timeframe,
                    duration=duration,
                    show_independent_pivots=(
                        show_independent_pivots
                    ),
                    show_finviz_channel=(
                        show_finviz_channel
                    ),
                    show_channel_midline=(
                        show_channel_midline
                    ),
                    show_smas=show_smas,
                    show_td_9=show_td_9,
                    show_td_all_counts=(
                        show_td_all_counts
                    ),
                    hide_non_trading=(
                        hide_non_trading
                    ),
                    use_log_scale=use_log_scale,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # =================================================
                # LIVE Y-AXIS PRICE + HOVER HEADER JAVASCRIPT
                # =================================================

                js_code = """
                <script>

                (function() {

                    /*
                     * Patch removeChild safely to prevent React reconciliation errors
                     * when Streamlit re-renders components.
                     */
                    try {
                        if (parent && parent.Node && !parent.Node.prototype._safeRemoveChildPatched) {
                            parent.Node.prototype._safeRemoveChildPatched = true;
                            var _origRemoveChild = parent.Node.prototype.removeChild;
                            parent.Node.prototype.removeChild = function(child) {
                                if (child && child.parentNode !== this) {
                                    if (child.parentNode) {
                                        return child.parentNode.removeChild(child);
                                    }
                                    return child;
                                }
                                return _origRemoveChild.apply(this, arguments);
                            };
                        }
                    } catch(e) {}

                    /*
                     * Find the newest Plotly chart.
                     *
                     * Streamlit creates a new chart every time
                     * the Python script reruns. Therefore we must
                     * NOT use querySelector(), because it returns
                     * the first/old chart.
                     */
                    function getCurrentPlot() {

                        var charts =
                            parent.document.querySelectorAll(
                                'div[data-testid="stPlotlyChart"] .js-plotly-plot'
                            );

                        if (
                            !charts ||
                            charts.length === 0
                        ) {
                            return null;
                        }

                        return charts[
                            charts.length - 1
                        ];
                    }


                    /*
                     * Remove badges from old charts safely.
                     */
                    function removeOldBadges() {

                        try {
                            var oldBadges =
                                parent.document.querySelectorAll(
                                    '.live-y-price-badge'
                                );

                            oldBadges.forEach(
                                function(badge) {
                                    try {
                                        if (badge && badge.parentNode) {
                                            badge.parentNode.removeChild(badge);
                                        }
                                    } catch(e) {}
                                }
                            );
                        } catch(e) {}
                    }


                    /*
                     * Initialize interaction on current chart.
                     */
                    function initYPriceInteractions() {

                        var plotEl =
                            getCurrentPlot();


                        /*
                         * Wait until Plotly is completely initialized.
                         */
                        if (
                            !plotEl ||
                            !plotEl._fullLayout ||
                            !plotEl._fullLayout.yaxis
                        ) {

                            setTimeout(
                                initYPriceInteractions,
                                300
                            );

                            return;
                        }


                        /*
                         * Do not attach twice to the same chart.
                         */
                        if (
                            plotEl.getAttribute(
                                'data-yprice-initialized'
                            ) === 'true'
                        ) {
                            return;
                        }


                        plotEl.setAttribute(
                            'data-yprice-initialized',
                            'true'
                        );


                        /*
                         * Remove badge belonging to previous chart.
                         */
                        removeOldBadges();


                        /*
                         * Find chart container.
                         */
                        var chartWrapper =
                            plotEl.closest(
                                'div[data-testid="stPlotlyChart"]'
                            );


                        if (!chartWrapper) {

                            chartWrapper =
                                plotEl.parentElement;
                        }


                        if (chartWrapper) {

                            chartWrapper.style.position =
                                'relative';
                        }


                        /*
                         * Create live Y-axis price badge.
                         */
                        var badgeEl =
                            parent.document.createElement(
                                'div'
                            );


                        badgeEl.className =
                            'live-y-price-badge';


                        badgeEl.style.position =
                            'absolute';


                        badgeEl.style.zIndex =
                            '99999';


                        badgeEl.style.pointerEvents =
                            'none';


                        badgeEl.style.padding =
                            '3px 7px';


                        badgeEl.style.minWidth =
                            '70px';


                        badgeEl.style.textAlign =
                            'right';


                        badgeEl.style.boxSizing =
                            'border-box';


                        badgeEl.style.borderRadius =
                            '3px';


                        badgeEl.style.fontSize =
                            '11px';


                        badgeEl.style.fontWeight =
                            'bold';


                        badgeEl.style.fontFamily =
                            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace';


                        badgeEl.style.color =
                            '#FFFFFF';


                        badgeEl.style.backgroundColor =
                            '#2962FF';


                        badgeEl.style.border =
                            '1px solid #1E53E5';


                        badgeEl.style.boxShadow =
                            '0px 2px 5px rgba(0,0,0,0.5)';


                        badgeEl.style.display =
                            'none';


                        badgeEl.style.transform =
                            'translateY(-50%)';


                        try {
                            if (chartWrapper) {
                                chartWrapper.appendChild(
                                    badgeEl
                                );
                            } else {
                                parent.document.body.appendChild(
                                    badgeEl
                                );
                            }
                        } catch(e) {
                            try {
                                parent.document.body.appendChild(
                                    badgeEl
                                );
                            } catch(err) {}
                        }


                        /*
                         * Mouse movement handler.
                         */
                        plotEl.addEventListener(
                            'mousemove',
                            function(e) {

                                /*
                                 * Make sure this is still
                                 * the currently displayed chart.
                                 */
                                var currentPlot =
                                    getCurrentPlot();


                                if (
                                    currentPlot !== plotEl
                                ) {

                                    badgeEl.style.display =
                                        'none';

                                    return;
                                }


                                if (
                                    !plotEl._fullLayout ||
                                    !plotEl._fullLayout.yaxis
                                ) {
                                    return;
                                }


                                var yaxis =
                                    plotEl._fullLayout.yaxis;


                                var xaxis =
                                    plotEl._fullLayout.xaxis;


                                var rect =
                                    plotEl.getBoundingClientRect();


                                var mouseY =
                                    e.clientY
                                    - rect.top;


                                var mouseX =
                                    e.clientX
                                    - rect.left;


                                /*
                                 * Price chart vertical range.
                                 */
                                var yTop =
                                    yaxis._offset;


                                var yBottom =
                                    yaxis._offset
                                    + yaxis._length;


                                /*
                                 * Price chart horizontal range.
                                 */
                                var xLeft =
                                    xaxis
                                    ? xaxis._offset
                                    : 0;


                                var xRight =
                                    xaxis
                                    ? (
                                        xaxis._offset
                                        + xaxis._length
                                    )
                                    : rect.width;


                                /*
                                 * Only show the badge when
                                 * cursor is inside the price area.
                                 */
                                if (
                                    mouseY >= yTop &&
                                    mouseY <= yBottom &&
                                    mouseX >= xLeft &&
                                    mouseX <= xRight
                                ) {

                                    /*
                                     * Convert mouse Y coordinate
                                     * to normalized price position.
                                     */
                                    var fraction =
                                        (
                                            yBottom
                                            - mouseY
                                        )
                                        /
                                        yaxis._length;


                                    fraction =
                                        Math.max(
                                            0,
                                            Math.min(
                                                1,
                                                fraction
                                            )
                                        );


                                    var price;


                                    /*
                                     * Linear Y-axis.
                                     */
                                    if (
                                        yaxis.type !== 'log'
                                    ) {

                                        var minVal =
                                            yaxis.range[0];


                                        var maxVal =
                                            yaxis.range[1];


                                        price =
                                            minVal
                                            +
                                            fraction
                                            *
                                            (
                                                maxVal
                                                - minVal
                                            );

                                    }


                                    /*
                                     * Logarithmic Y-axis.
                                     */
                                    else {

                                        var logMin =
                                            yaxis.range[0];


                                        var logMax =
                                            yaxis.range[1];


                                        price =
                                            Math.pow(
                                                10,
                                                logMin
                                                +
                                                fraction
                                                *
                                                (
                                                    logMax
                                                    - logMin
                                                )
                                            );
                                    }


                                    /*
                                     * Format the price.
                                     */
                                    var priceText;


                                    if (
                                        price > 0 &&
                                        price < 1
                                    ) {

                                        priceText =
                                            '$'
                                            + price.toFixed(4);

                                    }

                                    else if (
                                        price >= 1000
                                    ) {

                                        priceText =
                                            '$'
                                            +
                                            price.toLocaleString(
                                                undefined,
                                                {
                                                    minimumFractionDigits: 2,
                                                    maximumFractionDigits: 2
                                                }
                                            );

                                    }

                                    else {

                                        priceText =
                                            '$'
                                            + price.toFixed(2);
                                    }


                                    badgeEl.textContent =
                                        priceText;


                                    /*
                                     * Position the badge so that
                                     * its right edge touches the
                                     * Y-axis.
                                     */
                                    var badgeWidth =
                                        70;


                                    var leftPos =
                                        Math.max(
                                            0,
                                            xLeft
                                            - badgeWidth
                                        );


                                    badgeEl.style.left =
                                        leftPos + 'px';


                                    badgeEl.style.top =
                                        mouseY + 'px';


                                    badgeEl.style.display =
                                        'block';

                                }

                                else {

                                    badgeEl.style.display =
                                        'none';
                                }

                            }
                        );


                        /*
                         * Hide badge when cursor leaves chart.
                         */
                        plotEl.addEventListener(
                            'mouseleave',
                            function() {

                                badgeEl.style.display =
                                    'none';

                            }
                        );


                        /*
                         * -----------------------------------------
                         * Plotly hover handler
                         * -----------------------------------------
                         */
                        if (
                            !plotEl.getAttribute(
                                'data-hover-attached'
                            )
                        ) {

                            plotEl.setAttribute(
                                'data-hover-attached',
                                'true'
                            );


                            plotEl.on(
                                'plotly_hover',
                                function(data) {

                                    if (
                                        !data ||
                                        !data.points ||
                                        data.points.length === 0
                                    ) {
                                        return;
                                    }


                                    var pts =
                                        data.points;


                                    var dateStr = '';

                                    var o = 0;
                                    var h = 0;
                                    var l = 0;
                                    var c = 0;

                                    var sma20 = null;
                                    var sma50 = null;
                                    var sma200 = null;
                                    var supp = null;
                                    var mid = null;
                                    var res = null;


                                    /*
                                     * Extract data from traces.
                                     */
                                    for (
                                        var i = 0;
                                        i < pts.length;
                                        i++
                                    ) {

                                        var p =
                                            pts[i];


                                        if (
                                            p.open !== undefined
                                        ) {

                                            dateStr =
                                                p.x;

                                            o =
                                                p.open;

                                            h =
                                                p.high;

                                            l =
                                                p.low;

                                            c =
                                                p.close;
                                        }


                                        if (
                                            p.data &&
                                            p.data.name
                                        ) {

                                            if (
                                                p.data.name.includes(
                                                    'SMA 20'
                                                )
                                            ) {

                                                sma20 =
                                                    p.y;
                                            }


                                            if (
                                                p.data.name.includes(
                                                    'SMA 50'
                                                )
                                            ) {

                                                sma50 =
                                                    p.y;
                                            }


                                            if (
                                                p.data.name.includes(
                                                    'SMA 200'
                                                )
                                            ) {

                                                sma200 =
                                                    p.y;
                                            }


                                            if (
                                                p.data.name.includes(
                                                    'Finviz Support'
                                                ) ||
                                                p.data.name.includes(
                                                    'Support'
                                                )
                                            ) {

                                                supp =
                                                    p.y;
                                            }


                                            if (
                                                p.data.name.includes(
                                                    'Finviz Midline'
                                                ) ||
                                                p.data.name.includes(
                                                    'Midline'
                                                )
                                            ) {

                                                mid =
                                                    p.y;
                                            }


                                            if (
                                                p.data.name.includes(
                                                    'Finviz Resistance'
                                                ) ||
                                                p.data.name.includes(
                                                    'Resistance'
                                                )
                                            ) {

                                                res =
                                                    p.y;
                                            }
                                        }
                                    }


                                    var readoutEl =
                                        parent.document.getElementById(
                                            'live-header-readout'
                                        );


                                    if (
                                        readoutEl &&
                                        o > 0
                                    ) {

                                        if (
                                            !readoutEl.getAttribute(
                                                'data-default-html'
                                            )
                                        ) {

                                            readoutEl.setAttribute(
                                                'data-default-html',
                                                readoutEl.innerHTML
                                            );
                                        }


                                        /*
                                         * Find previous close.
                                         */
                                        var previousClose =
                                            null;


                                        try {

                                            var candleTrace =
                                                plotEl.data.find(
                                                    function(trace) {

                                                        return (
                                                            trace.type ===
                                                            'candlestick'
                                                        );

                                                    }
                                                );


                                            if (
                                                candleTrace &&
                                                candleTrace.close
                                            ) {

                                                var idx =
                                                    pts[0].pointIndex;


                                                if (
                                                    idx !== undefined &&
                                                    idx > 0
                                                ) {

                                                    previousClose =
                                                        Number(
                                                            candleTrace
                                                            .close[
                                                                idx - 1
                                                            ]
                                                        );
                                                }
                                            }

                                        }

                                        catch (err) {

                                            previousClose =
                                                null;
                                        }


                                        /*
                                         * Use previous close for
                                         * the day's percentage change.
                                         */
                                        var basePrice;


                                        if (
                                            previousClose !== null &&
                                            !isNaN(previousClose)
                                        ) {

                                            basePrice =
                                                previousClose;

                                        }

                                        else {

                                            basePrice =
                                                o;
                                        }


                                        var chg =
                                            c
                                            - basePrice;


                                        var pct =
                                            (
                                                chg
                                                / basePrice
                                            )
                                            * 100;


                                        var color =
                                            chg >= 0
                                            ? '#00E676'
                                            : '#FF5252';


                                        var parts = [
                                            '<b>DATE:</b> '
                                            + dateStr,
                                            '<b>O:</b> $'
                                            + o.toFixed(2)
                                            + ' &nbsp;<b>H:</b> $'
                                            + h.toFixed(2)
                                            + ' &nbsp;<b>L:</b> $'
                                            + l.toFixed(2)
                                            + ' &nbsp;<b>C:</b> $'
                                            + c.toFixed(2)
                                            + ' &nbsp;<span style="color:'
                                            + color
                                            + '">'
                                            + (
                                                chg >= 0
                                                ? '+'
                                                : ''
                                            )
                                            + chg.toFixed(2)
                                            + ', '
                                            + (
                                                pct >= 0
                                                ? '+'
                                                : ''
                                            )
                                            + pct.toFixed(2)
                                            + '%</span>'
                                        ];


                                        if (
                                            sma20 !== null
                                        ) {

                                            parts.push(
                                                '<b>SMA20:</b> '
                                                + '<span style="color:#42A5F5">$'
                                                + Number(
                                                    sma20
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        if (
                                            sma50 !== null
                                        ) {

                                            parts.push(
                                                '<b>SMA50:</b> '
                                                + '<span style="color:#EF5350">$'
                                                + Number(
                                                    sma50
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        if (
                                            sma200 !== null
                                        ) {

                                            parts.push(
                                                '<b>SMA200:</b> '
                                                + '<span style="color:#FFCA28">$'
                                                + Number(
                                                    sma200
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        if (
                                            supp !== null
                                        ) {

                                            parts.push(
                                                '<b>SUPP:</b> '
                                                + '<span style="color:#00B0FF">$'
                                                + Number(
                                                    supp
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        if (
                                            mid !== null
                                        ) {

                                            parts.push(
                                                '<b>MID:</b> '
                                                + '<span style="color:#B388FF">$'
                                                + Number(
                                                    mid
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        if (
                                            res !== null
                                        ) {

                                            parts.push(
                                                '<b>RES:</b> '
                                                + '<span style="color:#E040FB">$'
                                                + Number(
                                                    res
                                                ).toFixed(2)
                                                + '</span>'
                                            );
                                        }


                                        readoutEl.innerHTML =
                                            parts.join(
                                                ' &nbsp;|&nbsp; '
                                            );
                                    }

                                }
                            );


                            plotEl.on(
                                'plotly_unhover',
                                function() {

                                    var readoutEl =
                                        parent.document.getElementById(
                                            'live-header-readout'
                                        );


                                    if (
                                        readoutEl &&
                                        readoutEl.getAttribute(
                                            'data-default-html'
                                        )
                                    ) {

                                        readoutEl.innerHTML =
                                            readoutEl.getAttribute(
                                                'data-default-html'
                                            );
                                    }

                                }
                            );
                        }

                    }


                    /*
                     * Initial attempt.
                     */
                    setTimeout(
                        initYPriceInteractions,
                        500
                    );


                    /*
                     * ------------------------------------------------
                     * MutationObserver
                     * ------------------------------------------------
                     *
                     * Streamlit rebuilds the chart whenever:
                     *
                     *   - ticker changes
                     *   - timeframe changes
                     *   - duration changes
                     *   - sidebar option changes
                     *
                     * This observer detects the new chart and
                     * attaches the interaction code to it.
                     */
                    var observer =
                        new MutationObserver(
                            function() {

                                var currentPlot =
                                    getCurrentPlot();


                                if (
                                    currentPlot &&
                                    currentPlot._fullLayout &&
                                    currentPlot._fullLayout.yaxis
                                ) {

                                    if (
                                        currentPlot.getAttribute(
                                            'data-yprice-initialized'
                                        ) !== 'true'
                                    ) {

                                        setTimeout(
                                            initYPriceInteractions,
                                            300
                                        );
                                    }

                                }

                            }
                        );


                    observer.observe(
                        parent.document.body,
                        {
                            childList: true,
                            subtree: true
                        }
                    );

                })();

                </script>
                """

                components.html(
                    js_code,
                    height=0,
                    width=0
                )

                # -------------------------------------------------
                # Metrics
                # -------------------------------------------------

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Latest Close",
                    f"${latest_candle['Close']:.2f}",
                    f"{price_chg:+.2f} "
                    f"({pct_chg:+.2f}%)"
                )

                m2.metric(
                    "Candle Count",
                    f"{len(df)}"
                )

                m3.metric(
                    "TD Buy 9 Signals",
                    f"{len(df[df['TD_Buy'] == 9])}"
                )

                m4.metric(
                    "TD Sell 9 Signals",
                    f"{len(df[df['TD_Sell'] == 9])}"
                )

                # -------------------------------------------------
                # Raw data
                # -------------------------------------------------

                with st.expander(
                    "📊 Raw Stock Data Table"
                ):

                    st.dataframe(
                        df.tail(20),
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"Error loading chart for "
                    f"{symbol}: {str(e)}"
                )


if __name__ == "__main__":
    main()
else:
    main()