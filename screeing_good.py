import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import re
import streamlit as st
import streamlit.components.v1 as components
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import time
import ssl
import urllib.request
from io import StringIO, BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rsi_divergence_detector import fetch_ohlc, detect_divergences
from stock_news import fetch_stock_news

# --- Global Page Setup & Theme Styling ---
st.set_page_config(
    page_title="Multi-Asset Technical Analysis Suite", 
    page_icon="📊",
    layout="wide"
)

# Initialize navigation state
if "current_page" not in st.session_state:
    st.session_state.current_page = "landing"

# Helper navigation callbacks
def go_to_sp500():
    st.session_state.current_page = "sp500"

def go_to_nasdaq():
    st.session_state.current_page = "nasdaq"

def go_to_ratio():
    st.session_state.current_page = "ratio"

def go_to_marketcap():
    st.session_state.current_page = "marketcap"

def go_to_rsi_divergence():
    st.session_state.current_page = "rsi_divergence"

def go_to_stock_news():
    st.session_state.current_page = "stock_news"

def go_to_landing():
    st.session_state.current_page = "landing"

def go_to_advdecl():
    st.session_state.current_page = "advance_decline"


# --- Shared Cached Data Engine Functions ---
@st.cache_data(ttl=86400)
def get_nasdaq100_tickers():
    urls = [
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100"
    ]
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    context = ssl._create_unverified_context()
    
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=context) as response:
                html_content = response.read().decode('utf-8')
            payload = pd.read_html(StringIO(html_content))
            for table in payload:
                cols = [c[0] if isinstance(c, tuple) else c for c in table.columns]
                if 'Ticker' in cols or 'Symbol' in cols:
                    col_idx = cols.index('Ticker') if 'Ticker' in cols else cols.index('Symbol')
                    col_name = table.columns[col_idx]
                    tickers = table[col_name].dropna().astype(str).str.replace('.', '-', regex=False).tolist()
                    tickers = [t.strip() for t in tickers if t.strip()]
                    if len(tickers) >= 50:
                        return tickers
        except Exception:
            continue

    st.error("Error fetching NASDAQ-100 ticker list from Wikipedia.")
    return []

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        context = ssl._create_unverified_context()
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=context) as response:
            html_content = response.read().decode('utf-8')
        payload = pd.read_html(StringIO(html_content))
        df = payload[0]
        col_name = 'Symbol' if 'Symbol' in df.columns else 'Ticker'
        return df[col_name].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        st.error(f"Error fetching S&P 500 ticker list: {e}")
        return []

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(np.where(np.isnan(rs), 100.0, rsi), index=series.index)

def get_period(interval, period_or_years=None):
    if period_or_years is None or period_or_years == "":
        return "2y" if interval in ["1h", "4h"] else "max"
    if isinstance(period_or_years, int) or (isinstance(period_or_years, str) and period_or_years.isdigit()):
        years = int(period_or_years)
        if interval in ["1h", "4h"]:
            if years > 2:
                st.sidebar.warning("Capping intraday data to 2 years.")
                return "2y"
            return f"{years}y"
        return f"{years}y"
    p = str(period_or_years).strip().lower()
    if interval in ["1h", "4h"]:
        if p in ["3y", "5y", "10y", "max"]:
            st.sidebar.warning("Capping intraday data (1h/4h) to 2 years due to Yahoo Finance limits.")
            return "2y"
        return p
    return p

@st.cache_data(ttl=900, show_spinner=False)
def download_ratio_data(symbol, interval, period=None, years=None):
    p_val = period if period is not None else years
    p = get_period(interval, p_val)
    yf_interval = "1h" if interval == "4h" else interval
    df = yf.Ticker(symbol).history(period=p, interval=yf_interval)
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol}")
    prices = df["Close"]
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    if interval == "4h":
        prices = prices.resample("4h").last()
    return prices


# --- Market Cap Ratio Studio: Data Engine ---
# Logic: 1) if Yahoo gives a direct market cap, it's a stock, use it.
#        2) if not, treat as ETF: get FULL holdings from State Street's
#           (SSGA) public holdings file and sum each holding's market cap
#           via Yahoo Finance.
#        3) if it isn't a State Street fund, SSGA won't have it; fall back
#           to Yahoo's own top-~10 holdings, scaled up by 1/coverage% as
#           an approximation (clearly labeled as such in the chart).

SSGA_URL_TEMPLATE = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
SSGA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
MCAP_EXCLUDE_TICKERS = {"", "CASH", "USD", "N/A", "NAN", "-"}
MCAP_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-]{1,10}$")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ssga_holdings(ticker: str):
    """Returns a list of component ticker symbols, or None if this isn't
    a State Street fund (or the file couldn't be parsed)."""
    url = SSGA_URL_TEMPLATE.format(ticker=ticker.lower())
    try:
        resp = requests.get(url, headers=SSGA_HEADERS, timeout=15)
    except Exception:
        return None

    if resp.status_code != 200 or len(resp.content) < 1000:
        return None

    try:
        raw = pd.read_excel(BytesIO(resp.content), header=None, engine="openpyxl")
    except Exception:
        return None

    header_row_idx = None
    for i in range(min(20, len(raw))):
        row_vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if "ticker" in row_vals and any("weight" in v for v in row_vals):
            header_row_idx = i
            break
    if header_row_idx is None:
        return None

    try:
        data = pd.read_excel(BytesIO(resp.content), header=header_row_idx, engine="openpyxl")
    except Exception:
        return None

    data.columns = [str(c).strip() for c in data.columns]
    if "Ticker" not in data.columns:
        return None

    tickers = data["Ticker"].dropna().astype(str).str.strip().str.upper()
    tickers = tickers[tickers.str.match(MCAP_TICKER_PATTERN)]
    tickers = [t for t in tickers if t not in MCAP_EXCLUDE_TICKERS]
    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order

    return tickers if tickers else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_direct_market_cap(symbol: str):
    """Current market cap straight from Yahoo, if Yahoo has one (stocks do)."""
    info = yf.Ticker(symbol).info
    return info.get("marketCap")


@st.cache_data(ttl=86400, show_spinner=False)
def get_shares_outstanding(symbol: str):
    info = yf.Ticker(symbol).info
    return info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")


def get_shares_outstanding_bulk(symbols, progress_callback=None):
    """Parallelized shares-outstanding lookup (network-bound, so threads help a lot)."""
    result = {}
    total = len(symbols)
    done = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(get_shares_outstanding, s): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                result[s] = fut.result()
            except Exception:
                result[s] = None
            done += 1
            if progress_callback:
                progress_callback(done, total, s)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_etf_top_holdings_fallback(symbol: str):
    """Fallback for non-State-Street ETFs: Yahoo's own top-holdings list (~10 names)."""
    t = yf.Ticker(symbol)
    try:
        holdings = t.funds_data.top_holdings
    except Exception:
        holdings = None
    if holdings is None or holdings.empty:
        raise ValueError(
            f"Could not retrieve holdings for '{symbol}' from either SSGA or Yahoo Finance."
        )
    symbols = list(holdings.index)
    pct_col = None
    for col in holdings.columns:
        if "percent" in col.lower() or "weight" in col.lower():
            pct_col = col
            break
    if pct_col is not None:
        coverage = float(holdings[pct_col].sum())
        if coverage > 1.5:
            coverage = coverage / 100.0
    else:
        coverage = None
    return symbols, coverage


@st.cache_data(ttl=900, show_spinner=False)
def download_prices_batch(symbols: tuple, interval: str, period=None, years=None):
    """Batch close-price history for many symbols in as few requests as possible."""
    p_val = period if period is not None else years
    period_str = get_period(interval, p_val)
    yf_interval = "1h" if interval == "4h" else interval
    symbols = list(symbols)

    try:
        data = yf.download(
            symbols, period=period_str, interval=yf_interval,
            progress=False, group_by="ticker", auto_adjust=True, threads=True,
        )
    except Exception:
        data = None

    prices = {}
    if data is not None and not data.empty:
        if len(symbols) == 1:
            sym = symbols[0]
            try:
                col = data["Close"] if "Close" in data.columns else data[sym]["Close"]
                if col.notna().any():
                    prices[sym] = col
            except Exception:
                pass
        else:
            for s in symbols:
                try:
                    col = data[s]["Close"]
                    if col.notna().any():
                        prices[s] = col
                except Exception:
                    continue

    df = pd.DataFrame(prices)
    if not df.empty:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if interval == "4h":
            df = df.resample("4h").last()
    return df


def fmt_cap(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    if abs(x) >= 1e12:
        return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"


def plot_rsi_divergence_chart(df, bullish_signals, bearish_signals, symbol: str):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{symbol} Price", "RSI (14)"),
    )

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4", width=1.5)), row=1, col=1)
    if bullish_signals:
        for signal in bullish_signals:
            fig.add_trace(
                go.Scatter(
                    x=[signal["start_date"], signal["end_date"]],
                    y=[signal["start_price"], signal["end_price"]],
                    mode="lines+markers",
                    name="Bullish Divergence",
                    line=dict(color="green", width=2),
                    marker=dict(color="green", size=12, symbol="triangle-up"),
                ),
                row=1,
                col=1,
            )
    if bearish_signals:
        for signal in bearish_signals:
            fig.add_trace(
                go.Scatter(
                    x=[signal["start_date"], signal["end_date"]],
                    y=[signal["start_price"], signal["end_price"]],
                    mode="lines+markers",
                    name="Bearish Divergence",
                    line=dict(color="red", width=2),
                    marker=dict(color="red", size=12, symbol="triangle-down"),
                ),
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#ff7f0e", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
    fig.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")

    if bullish_signals:
        for signal in bullish_signals:
            fig.add_trace(
                go.Scatter(
                    x=[signal["start_date"], signal["end_date"]],
                    y=[signal["start_rsi"], signal["end_rsi"]],
                    mode="lines+markers",
                    name="Bullish RSI",
                    line=dict(color="green", width=2),
                    marker=dict(color="green", size=10, symbol="triangle-up"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
    if bearish_signals:
        for signal in bearish_signals:
            fig.add_trace(
                go.Scatter(
                    x=[signal["start_date"], signal["end_date"]],
                    y=[signal["start_rsi"], signal["end_rsi"]],
                    mode="lines+markers",
                    name="Bearish RSI",
                    line=dict(color="red", width=2),
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        height=800,
        hovermode="x",
        spikedistance=-1,
        hoverdistance=-1,
        showlegend=True,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_xaxes(
        showspikes=True,
        spikemode="across+toaxis",
        spikesnap="cursor",
        spikecolor="#888888",
        spikedash="dot",
        spikethickness=1
    )
    return fig


def build_synced_chart_html(fig, df, symbol1, symbol2, interval, height=780):
    """Renders a 3-panel market-cap-ratio figure as raw HTML with a custom JS
    crosshair that manually links all three panels on hover. Plotly's native
    hoversubplots="axis" feature doesn't work through st.plotly_chart (a
    confirmed Streamlit bug, streamlit/streamlit#9118), so instead a small JS
    snippet listens for Plotly's hover event on any one panel and manually
    draws a full-height line plus a value readout on all three via
    Plotly.relayout -- this only depends on Plotly's basic shapes/annotations
    API, not on any hover-linking feature."""
    import json as _json

    intraday = interval in ("1h", "4h")
    records = [
        {
            "t": int(ts.value // 10**6),  # epoch milliseconds
            "s1": None if pd.isna(v1) else float(v1),
            "s2": None if pd.isna(v2) else float(v2),
            "ratio": None if pd.isna(vr) else float(vr),
            "ratio_ma": None if pd.isna(vma) else float(vma),
            "diff": None if pd.isna(vr) or pd.isna(vma) else float(vr - vma),
        }
        for ts, v1, v2, vr, vma in zip(
            df.index,
            df[symbol1],
            df[symbol2],
            df["Ratio"],
            df.get("Ratio_MA", pd.Series([np.nan] * len(df), index=df.index)),
        )
    ]
    records_json = _json.dumps(records)

    post_script = f"""
(function() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd) return;
    var records = {records_json};
    var times = records.map(function(r) {{ return r.t; }});
    var intraday = {str(intraday).lower()};

    function nearestIdx(t) {{
        var lo = 0, hi = times.length - 1;
        if (hi < 0) return -1;
        if (t <= times[0]) return 0;
        if (t >= times[hi]) return hi;
        while (lo < hi) {{
            var mid = Math.floor((lo + hi) / 2);
            if (times[mid] === t) return mid;
            if (times[mid] < t) lo = mid + 1; else hi = mid;
        }}
        var a = lo - 1 < 0 ? 0 : lo - 1;
        return (Math.abs(times[a] - t) < Math.abs(times[lo] - t)) ? a : lo;
    }}

    function fmtCap(x) {{
        if (x === null || x === undefined) return 'n/a';
        var ax = Math.abs(x);
        if (ax >= 1e12) return '$' + (x / 1e12).toFixed(2) + 'T';
        if (ax >= 1e9) return '$' + (x / 1e9).toFixed(2) + 'B';
        if (ax >= 1e6) return '$' + (x / 1e6).toFixed(2) + 'M';
        return '$' + x.toFixed(0);
    }}

function fmtRatio(x) {{
        return x === null || x === undefined ? 'n/a' : x.toFixed(4);
    }}

    function fmtDiff(x) {{
        if (x === null || x === undefined) return 'n/a';
        return (x >= 0 ? '+' : '') + x.toFixed(4);
    }}

    function fmtDate(t) {{
        var d = new Date(t);
        if (intraday) {{
            return d.toLocaleString(undefined, {{month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}});
        }}
        return d.toLocaleDateString(undefined, {{year:'numeric', month:'short', day:'numeric'}});
    }}
    gd.on('plotly_hover', function(evt) {{
        if (!evt.points || !evt.points.length) return;
        var xVal = evt.points[0].x;
        var t = new Date(xVal).getTime();
        var idx = nearestIdx(t);
        if (idx < 0) return;
        var r = records[idx];
        var dateStr = fmtDate(r.t);

        var shapes = [
            {{type:'line', xref:'x', yref:'paper', x0:xVal, x1:xVal, y0:0, y1:1, line:{{color:'#888888', width:1.5, dash:'dot'}}}}
        ];

        var annotations = [
            {{x:0.015, y:0.96, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>{symbol1}</b>: ' + fmtCap(r.s1),
              font:{{color:'#1f77b4', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}},
            {{x:0.015, y:0.60, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>{symbol2}</b>: ' + fmtCap(r.s2),
              font:{{color:'#ff7f0e', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}},
            {{x:0.015, y:0.32, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>Ratio</b>: ' + fmtRatio(r.ratio),
              font:{{color:'purple', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}},
            {{x:0.015, y:0.18, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>Ratio MA</b>: ' + fmtRatio(r.ratio_ma) + ' | <b>Diff</b>: ' + fmtDiff(r.diff),
              font:{{color:'#228B22', size:12}}, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}}
        ];

        Plotly.relayout(gd, {{shapes: shapes, annotations: annotations}});
    }});

    gd.on('plotly_unhover', function(evt) {{
        Plotly.relayout(gd, {{shapes: [], annotations: []}});
    }});
}})();
"""
    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        post_script=post_script,
    )




def build_synced_adv_decl_chart_html(fig, df, symbol, interval, height=780):
    """Renders a 3-panel advance/decline figure as raw HTML with a custom JS
    crosshair that manually links all three panels on hover."""
    import json as _json

    intraday = interval in ("1h", "4h")
    records = [
        {
            "t": int(ts.value // 10**6),  # epoch milliseconds
            "p": None if pd.isna(p) else float(p),
            "a": int(a),
            "d": int(d),
            "u": int(u),
            "pa": None if pd.isna(pa) else float(pa),
            "na": None if pd.isna(na) else float(na),
        }
        for ts, p, a, d, u, pa, na in zip(
            df.index,
            df["Price"],
            df["Advancing"],
            df["Declining"],
            df["Unchanged"],
            df["Pct_Adv"],
            df["Net_Adv"],
        )
    ]
    records_json = _json.dumps(records)

    post_script = f"""
(function() {{
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd) return;
    var records = {records_json};
    var times = records.map(function(r) {{ return r.t; }});
    var intraday = {str(intraday).lower()};

    function nearestIdx(t) {{
        var lo = 0, hi = times.length - 1;
        if (hi < 0) return -1;
        if (t <= times[0]) return 0;
        if (t >= times[hi]) return hi;
        while (lo < hi) {{
            var mid = Math.floor((lo + hi) / 2);
            if (times[mid] === t) return mid;
            if (times[mid] < t) lo = mid + 1; else hi = mid;
        }}
        var a = lo - 1 < 0 ? 0 : lo - 1;
        return (Math.abs(times[a] - t) < Math.abs(times[lo] - t)) ? a : lo;
    }}

    function fmtDate(t) {{
        var d = new Date(t);
        if (intraday) {{
            return d.toLocaleString(undefined, {{month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}});
        }}
        return d.toLocaleDateString(undefined, {{year:'numeric', month:'short', day:'numeric'}});
    }}

    gd.on('plotly_hover', function(evt) {{
        if (!evt.points || !evt.points.length) return;
        var xVal = evt.points[0].x;
        var t = new Date(xVal).getTime();
        var idx = nearestIdx(t);
        if (idx < 0) return;
        var r = records[idx];
        var dateStr = fmtDate(r.t);

        var shapes = [
            {{type:'line', xref:'x', yref:'paper', x0:xVal, x1:xVal, y0:0, y1:1, line:{{color:'#888888', width:1.5, dash:'dot'}}}}
        ];

        var annotations = [
            {{x:0.015, y:0.96, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>{symbol} Price</b>: $' + (r.p ? r.p.toFixed(2) : 'n/a'),
              font:{{color:'#1f77b4', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}},
            {{x:0.015, y:0.60, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>Adv</b>: ' + r.a + ' | <b>Dec</b>: ' + r.d + ' | <b>Unc</b>: ' + r.u,
              font:{{color:'#2ca02c', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}},
            {{x:0.015, y:0.32, xref:'paper', yref:'paper', xanchor:'left', yanchor:'top', showarrow:false,
              text: dateStr + '&nbsp;&nbsp;<b>% Adv</b>: ' + (r.pa ? (r.pa*100).toFixed(1) + "%" : 'n/a') + ' | <b>Net Adv</b>: ' + (r.na ? (r.na*100).toFixed(1) + "%" : 'n/a'),
              font:{{color:'purple', size:12}}, bgcolor:'rgba(255,255,255,0.75)', bordercolor:'rgba(0,0,0,0.15)', borderpad:3}}
        ];

        Plotly.relayout(gd, {{shapes: shapes, annotations: annotations}});
    }});

    gd.on('plotly_unhover', function() {{
        Plotly.relayout(gd, {{shapes: [], annotations: []}});
    }});
}})();
"""

    return fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True},
        post_script=post_script,
    )





def get_market_cap_series(symbol: str, interval: str, period=None, years=None, progress_slot=None):
    """Returns (market_cap_series, descriptive_label) for a stock or ETF."""
    p_val = period if period is not None else years

    # Step 1: does Yahoo give this symbol a direct market cap? -> treat as a stock.
    direct_mc = get_direct_market_cap(symbol)
    if direct_mc:
        shares = get_shares_outstanding(symbol)
        if not shares:
            raise ValueError(f"'{symbol}' has a market cap but no shares-outstanding figure.")
        prices = download_ratio_data(symbol, interval, period=p_val)
        mc = prices * shares
        return mc, f"{symbol} (stock)"

    # Step 2: no direct market cap -> treat as ETF. Try SSGA full holdings first.
    components = fetch_ssga_holdings(symbol)
    full_holdings = True
    coverage = 1.0

    if not components:
        # Step 3 fallback: not a State Street fund -> Yahoo's top-holdings only.
        components, coverage = get_etf_top_holdings_fallback(symbol)
        full_holdings = False
        coverage = coverage or 1.0

    total = len(components)
    if progress_slot is not None:
        progress_slot.progress(0.0, text=f"Looking up shares outstanding for {symbol}'s {total} holdings...")

    def _update(done, tot, tkr):
        if progress_slot is not None:
            progress_slot.progress(done / tot, text=f"{symbol}: shares outstanding {done}/{tot} ({tkr})")

    shares_map = get_shares_outstanding_bulk(components, progress_callback=_update)

    if progress_slot is not None:
        progress_slot.empty()

    usable = [c for c in components if shares_map.get(c)]
    if not usable:
        raise ValueError(f"No usable holdings market-cap data for '{symbol}'")

    prices_df = download_prices_batch(tuple(usable), interval, period=p_val)
    if prices_df.empty:
        raise ValueError(f"No price data for {symbol}'s holdings")

    mc_total = pd.Series(0.0, index=prices_df.index)
    used = 0
    for c in usable:
        if c not in prices_df.columns:
            continue
        shares = shares_map.get(c)
        mc_total = mc_total.add(prices_df[c] * shares, fill_value=0)
        used += 1

    if used == 0:
        raise ValueError(f"No usable holdings price data for '{symbol}'")

    if full_holdings:
        label = f"{symbol} (ETF, full SSGA holdings, {used}/{total} names priced)"
    else:
        mc_total = mc_total / coverage
        label = f"{symbol} (ETF approx., top holdings only, ~{coverage*100:.0f}% coverage, {used} names)"

    return mc_total, label


@st.cache_data(ttl=3600, show_spinner=False)
def get_components_for_symbol(symbol: str):
    """Return a list of component tickers for an ETF/index or None on failure."""
    s = symbol.strip().upper()
    # Common index shortcuts
    if s in ("SPY", "S&P500", "^GSPC", "SP500"):
        return get_sp500_tickers()
    if s in ("QQQ", "NASDAQ-100", "NDX"):
        return get_nasdaq100_tickers()

    # Try State Street holdings (SSGA) for ETFs
    comps = fetch_ssga_holdings(s)
    if comps:
        return comps

    # Fallback to Yahoo's top holdings for ETFs
    try:
        syms, coverage = get_etf_top_holdings_fallback(s)
        return syms
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def compute_advance_decline(components: list, interval: str, years=None):
    """Compute advancing/declining counts series for a list of component tickers.

    Returns a tuple (adv_count, dec_count, unchanged_count, pct_adv, net_adv, prices_df)
    where each of the first five are pd.Series indexed by date and prices_df is
    the components' close price DataFrame.
    """
    if not components:
        raise ValueError("No component tickers provided")

    prices = download_prices_batch(tuple(components), interval, years)
    if prices is None or prices.empty:
        raise ValueError("No price data for components")

    # Align to common index and compute daily change sign
    diff = prices.diff()
    adv = (diff > 0).sum(axis=1)
    dec = (diff < 0).sum(axis=1)
    unc = prices.notna().sum(axis=1) - adv - dec
    total = prices.notna().sum(axis=1).replace(0, np.nan)
    pct_adv = adv / total
    net_adv = (adv - dec) / total

    return adv, dec, unc, pct_adv, net_adv, prices


def plot_advance_decline_figure(symbol_price, adv, dec, unc, pct_adv, net_adv, symbol: str, interval: str):
    """Build a 3-row Plotly figure showing price, advance/decline counts, and percent advancing."""
    fig = go.Figure(layout=dict(
        title=dict(text=f"{symbol} Breadth Analysis — Price & Adv/Decl ({interval})", x=0.5, font=dict(size=14)),
        height=800,
        hovermode="closest",
        showlegend=True,
        margin=dict(t=60, b=40, l=40, r=40),
    ))

    # Top: symbol price
    fig.add_trace(go.Scatter(x=symbol_price.index, y=symbol_price, name=f"{symbol} Price", xaxis="x", yaxis="y", line=dict(color="#1f77b4")))

    # Middle: stacked bars for adv/dec
    fig.add_trace(go.Bar(x=adv.index, y=adv, name="Advancing", xaxis="x2", yaxis="y2", marker_color="#2ca02c"))
    fig.add_trace(go.Bar(x=dec.index, y=dec, name="Declining", xaxis="x2", yaxis="y2", marker_color="#d62728"))
    fig.add_trace(go.Bar(x=unc.index, y=unc, name="Unchanged", xaxis="x2", yaxis="y2", marker_color="#7f7f7f"))

    # Bottom: percent advancing and net line
    fig.add_trace(go.Scatter(x=pct_adv.index, y=pct_adv, name="% Advancing", xaxis="x3", yaxis="y3", line=dict(color="purple")))
    fig.add_trace(go.Bar(x=net_adv.index, y=net_adv, name="Net Adv", xaxis="x3", yaxis="y3", marker_color="#9467bd", opacity=0.5))

    fig.update_layout(
        xaxis=dict(anchor="y", domain=[0, 1], matches="x3", showticklabels=False),
        yaxis=dict(domain=[0.72, 1.00], title="Price"),
        xaxis2=dict(anchor="y2", domain=[0, 1], matches="x3", showticklabels=False),
        yaxis2=dict(domain=[0.36, 0.64], title="Count"),
        xaxis3=dict(anchor="y3", domain=[0, 1], title="Date"),
        yaxis3=dict(domain=[0.00, 0.28], title="% / Net"),
    )
    return fig



# ==========================================
# PAGE 1: LANDING ENTRYWAY
# ==========================================
if st.session_state.current_page == "landing":
    st.title("📊 Multi-Asset Technical Analysis Suite")
    st.write("Welcome to your centralized market intelligence desk. Select one of the modules below to get started:")
    st.markdown("---")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.subheader("🇺🇸 S&P 500 Core Screener")
        st.write("Scan the broad index sectors for institutional value turnarounds using custom RSI setups and volume qualifiers.")
        st.button("Launch S&P 500 Dashboard", on_click=go_to_sp500, type="primary", use_container_width=True)
        
    with col2:
        st.subheader("💻 NASDAQ-100 Momentum Screener")
        st.write("Focus purely on mega-cap tech, innovation champions, and structural wave momentum contractions.")
        st.button("Launch NASDAQ-100 Dashboard", on_click=go_to_nasdaq, type="primary", use_container_width=True)

    with col3:
        st.subheader("📈 Inter-Asset Price Ratio Studio")
        st.write("Track relative performance metrics between pairs of assets (e.g., MAGS vs QQQ) using synchronized crosshairs.")
        st.button("Launch Ratio Studio Dashboard", on_click=go_to_ratio, type="primary", use_container_width=True)

    with col4:
        st.subheader("🏦 Market Cap Ratio Studio")
        st.write("Compare true market-cap weight between a stock and an ETF (full holdings via SSGA) or between two stocks.")
        st.button("Launch Market Cap Dashboard", on_click=go_to_marketcap, type="primary", use_container_width=True)

    with col5:
        st.subheader("📉 RSI Divergence Analyzer")
        st.write("Find bullish and bearish RSI divergences on a stock chart using Yahoo Finance data.")
        st.button("Launch RSI Divergence", on_click=go_to_rsi_divergence, type="primary", use_container_width=True)

    with col6:
        st.subheader("📰 Stock News Hub")
        st.write("Look up today's Yahoo Finance and CNBC headlines for a ticker and open them directly.")
        st.button("Launch Stock News", on_click=go_to_stock_news, type="primary", use_container_width=True)

    # Extra quick access: Advance/Decline Studio
    st.markdown("---")
    st.button("Launch Advance/Decline Studio", on_click=go_to_advdecl, type="secondary", use_container_width=True)


# ==========================================
# PAGE 2: S&P 500 DASHBOARD
# ==========================================
elif st.session_state.current_page == "sp500":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True)
        
    st.title("🇺🇸 S&P 500 Live Technical Screener")
    st.caption("Scrapes live S&P 500 components and filters them through Yahoo Finance.")
    
    st.sidebar.header("S&P 500 Parameters")
    drop_threshold = st.sidebar.slider("📉 Min Price Drop from High (%)", 1.0, 90.0, 5.0, 0.5, key="sp_drop")
    rsi_threshold = st.sidebar.slider("⚡ Max Daily RSI Ceiling", 10, 90, 30, key="sp_rsi")
    vol_ma_window = st.sidebar.number_input("📊 Volume MA Window (Days)", 5, 50, 10, step=1, key="sp_vol")
    
    st.sidebar.markdown("---")
    run_screener = st.sidebar.button("🚀 Run S&P 500 Screener", type="primary", use_container_width=True)
    re_run = st.sidebar.button("🔄 Refresh / Recalculate Grid", use_container_width=True)

    if run_screener or re_run:
        tickers = get_sp500_tickers()
        if not tickers:
            st.error("Could not load tickers.")
        else:
            status_container = st.container()
            with status_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
            
            matched_stocks = []
            for i, ticker in enumerate(tickers, 1):
                progress_bar.progress(i / len(tickers))
                status_text.caption(f"**Scanning Status:** Processing {i}/{len(tickers)} (`{ticker}`)...")
                try:
                    df = yf.Ticker(ticker).history(period="2y", interval="1d")
                    if df is None or len(df) < max(20, vol_ma_window): continue
                    
                    current_close = df['Close'].iloc[-1]
                    current_volume = df['Volume'].iloc[-1]
                    recent_high = df['High'].max()
                    pct_drop = ((recent_high - current_close) / recent_high) * 100
                    
                    df['RSI'] = calculate_rsi(df['Close'], period=14)
                    current_rsi = df['RSI'].iloc[-1]
                    df['Vol_MA'] = df['Volume'].rolling(window=vol_ma_window).mean()
                    current_vol_ma = df['Vol_MA'].iloc[-1]
                    
                    if (pct_drop > drop_threshold) and (current_rsi < rsi_threshold) and (current_volume > current_vol_ma):
                        matched_stocks.append({
                            "Ticker": ticker, "Link": f"https://finance.yahoo.com/quote/{ticker}",
                            "Current Price": current_close, "Recent High": recent_high,
                            "Drop from High": round(pct_drop, 1), "Daily RSI": current_rsi,
                            "Today's Volume": current_volume, "Volume MA": current_vol_ma
                        })
                    time.sleep(0.01)
                except Exception: continue
                    
            status_container.empty()
            if matched_stocks:
                results_df = pd.DataFrame(matched_stocks).sort_values(by="Drop from High", ascending=False)
                m1, m2, m3 = st.columns(3)
                m1.metric("Signals Triggered", f"🔥 {len(matched_stocks)} Stocks")
                m2.metric("RSI Ceiling", f"Under {rsi_threshold}")
                m3.metric("Pullback Base", f"💥 > {drop_threshold}%")
                
                st.dataframe(
                    results_df,
                    column_config={
                        "Link": st.column_config.LinkColumn(
                            "Yahoo Chart",
                            display_text="View Chart ↗"
                        ),
                        "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Recent High": st.column_config.NumberColumn(format="$%.2f"),
                        "Drop from High": st.column_config.NumberColumn(format="%.1f%%"),
                        "Daily RSI": st.column_config.NumberColumn(format="%.2f"),
                        "Today's Volume": st.column_config.NumberColumn(format="%d"),
                        "Volume MA": st.column_config.NumberColumn(format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else: st.warning("No S&P 500 stocks match your criteria.")


# ==========================================
# PAGE 3: NASDAQ-100 DASHBOARD
# ==========================================
elif st.session_state.current_page == "nasdaq":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True)
        
    st.title("💻 NASDAQ-100 High-Growth Screener")
    st.caption("Scrapes live NASDAQ-100 metrics and processes them with daily volume qualifiers.")
    
    st.sidebar.header("NASDAQ-100 Parameters")
    drop_threshold = st.sidebar.slider("📉 Min Price Drop from High (%)", 1.0, 90.0, 5.0, 0.5, key="ndx_drop")
    rsi_threshold = st.sidebar.slider("⚡ Max Daily RSI Ceiling", 10, 90, 30, key="ndx_rsi")
    vol_ma_window = st.sidebar.number_input("📊 Volume MA Window (Days)", 5, 50, 10, step=1, key="ndx_vol")
    
    st.sidebar.markdown("---")
    run_screener = st.sidebar.button("🚀 Run NASDAQ Screener", type="primary", use_container_width=True)
    re_run = st.sidebar.button("🔄 Refresh / Recalculate Grid", use_container_width=True)

    if run_screener or re_run:
        tickers = get_nasdaq100_tickers()
        if not tickers: st.error("Could not load tickers.")
        else:
            status_container = st.container()
            with status_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
            
            matched_stocks = []
            for i, ticker in enumerate(tickers, 1):
                progress_bar.progress(i / len(tickers))
                status_text.caption(f"**Scanning Status:** Processing {i}/{len(tickers)} (`{ticker}`)...")
                try:
                    df = yf.Ticker(ticker).history(period="2y", interval="1d")
                    if df is None or len(df) < max(20, vol_ma_window): continue
                    
                    current_close = df['Close'].iloc[-1]
                    current_volume = df['Volume'].iloc[-1]
                    recent_high = df['High'].max()
                    pct_drop = ((recent_high - current_close) / recent_high) * 100
                    
                    df['RSI'] = calculate_rsi(df['Close'], period=14)
                    current_rsi = df['RSI'].iloc[-1]
                    df['Vol_MA'] = df['Volume'].rolling(window=vol_ma_window).mean()
                    current_vol_ma = df['Vol_MA'].iloc[-1]
                    
                    if (pct_drop > drop_threshold) and (current_rsi < rsi_threshold) and (current_volume > current_vol_ma):
                        matched_stocks.append({
                            "Ticker": ticker, "Link": f"https://finance.yahoo.com/quote/{ticker}",
                            "Current Price": current_close, "Recent High": recent_high,
                            "Drop from High": round(pct_drop, 1), "Daily RSI": current_rsi,
                            "Today's Volume": current_volume, "Volume MA": current_vol_ma
                        })
                    time.sleep(0.01)
                except Exception: continue
                    
            status_container.empty()
            if matched_stocks:
                results_df = pd.DataFrame(matched_stocks).sort_values(by="Drop from High", ascending=False)
                m1, m2, m3 = st.columns(3)
                m1.metric("NASDAQ Signals Triggered", f"🔥 {len(matched_stocks)} Stocks")
                m2.metric("RSI Ceiling", f"Under {rsi_threshold}")
                m3.metric("Pullback Base", f"💥 > {drop_threshold}%")
                
                st.dataframe(
                    results_df,
                    column_config={
                        "Link": st.column_config.LinkColumn(
                            "Yahoo Chart",
                            display_text="View Chart ↗"
                        ),
                        "Current Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Recent High": st.column_config.NumberColumn(format="$%.2f"),
                        "Drop from High": st.column_config.NumberColumn(format="%.1f%%"),
                        "Daily RSI": st.column_config.NumberColumn(format="%.2f"),
                        "Today's Volume": st.column_config.NumberColumn(format="%d"),
                        "Volume MA": st.column_config.NumberColumn(format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else: st.warning("No NASDAQ-100 stocks match your criteria.")


# ==========================================
# PAGE 4: INTER-ASSET PRICE RATIO STUDIO
# ==========================================
elif st.session_state.current_page == "ratio":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True)
        
    st.title("📈 Inter-Asset Price Ratio Studio")
    st.caption("Plots historical individual metrics alongside structural pricing ratio graphs with crosshair tracking.")
    
    st.sidebar.header("Ratio Studio Adjustments")
    s1 = st.sidebar.text_input("Asset 1 (Numerator)", "MAGS").strip().upper()
    s2 = st.sidebar.text_input("Asset 2 (Denominator)", "QQQ").strip().upper()
    
    interval = st.sidebar.radio("Interval Frequency", ("1d", "1wk", "1mo", "1h", "4h"), index=0)
    
    period_options = ("1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max")
    period = st.sidebar.selectbox("Lookback Window (Period)", period_options, index=period_options.index("1y"), key="ratio_period")
    
    ma_window = st.sidebar.number_input("Ratio Moving Average Window", min_value=0, max_value=200, value=20, step=1)
    
    st.sidebar.markdown("---")
    refresh_ratio = st.sidebar.button("🔄 Generate & Refresh Charts", type="primary", use_container_width=True)
    
    if s1 and s2:
        try:
            s1_data = download_ratio_data(s1, interval, period)
            s2_data = download_ratio_data(s2, interval, period)
            
            df_global = pd.concat([s1_data, s2_data], axis=1, join="inner")
            df_global.columns = [s1, s2]
            if df_global.index.tz is not None:
                df_global.index = df_global.index.tz_localize(None)
                
            df_global["Ratio"] = df_global[s1] / df_global[s2]
            if ma_window > 0:
                df_global["Ratio_MA"] = df_global["Ratio"].rolling(window=ma_window).mean()
                df_global["Ratio_Diff"] = df_global["Ratio"] - df_global["Ratio_MA"]

            ratio_max_pos_idx = ratio_max_neg_idx = None
            if ma_window > 0 and "Ratio_MA" in df_global.columns:
                ratio_deviation = df_global["Ratio_Diff"].dropna()
                if not ratio_deviation.empty:
                    ratio_max_pos_idx = ratio_deviation.idxmax()
                    ratio_max_neg_idx = ratio_deviation.idxmin()
                    ratio_max_pos_val = ratio_deviation.loc[ratio_max_pos_idx]
                    ratio_max_neg_val = ratio_deviation.loc[ratio_max_neg_idx]

                    st.metric(
                        f"Biggest Positive Deviation (Ratio above MA)",
                        f"+{ratio_max_pos_val:.4f}",
                        f"on {ratio_max_pos_idx.strftime('%Y-%m-%d')}",
                    )
                    st.metric(
                        f"Biggest Negative Deviation (Ratio below MA)",
                        f"{ratio_max_neg_val:.4f}",
                        f"on {ratio_max_neg_idx.strftime('%Y-%m-%d')}",
                        delta_color="inverse",
                    )

            if ma_window > 0:
                st.info("Hover over the ratio line to see the live Ratio vs MA difference for each point.")

            fig = go.Figure(layout=dict(
                title=dict(text=f"{s1} vs {s2} — Price & Ratio ({interval})", x=0.5, font=dict(size=14)),
                height=750,
                hovermode="closest",
                showlegend=True,
                margin=dict(t=60, b=40, l=40, r=40),
            ))

            fig.add_trace(go.Scatter(x=df_global.index, y=df_global[s1], name=s1, xaxis="x", yaxis="y", line=dict(color='#1f77b4', width=1.5)))
            fig.add_trace(go.Scatter(x=df_global.index, y=df_global[s2], name=s2, xaxis="x2", yaxis="y2", line=dict(color='#ff7f0e', width=1.5)))
            if ma_window > 0:
                ratio_customdata = np.vstack([
                    df_global["Ratio_MA"].to_numpy(dtype=float),
                    df_global["Ratio_Diff"].to_numpy(dtype=float),
                ]).T
                fig.add_trace(go.Scatter(
                    x=df_global.index,
                    y=df_global["Ratio"],
                    name="Price Ratio",
                    xaxis="x3",
                    yaxis="y3",
                    line=dict(color='purple', width=1.5),
                    customdata=ratio_customdata,
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Ratio: %{y:.4f}<br>"
                        "MA: %{customdata[0]:.4f}<br>"
                        "Diff: %{customdata[1]:+.4f}<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter(x=df_global.index, y=df_global["Ratio_MA"], name=f"{ma_window}-Period MA", xaxis="x3", yaxis="y3", line=dict(color='green', width=1.5, dash='dash')))
            else:
                fig.add_trace(go.Scatter(
                    x=df_global.index,
                    y=df_global["Ratio"],
                    name="Price Ratio",
                    xaxis="x3",
                    yaxis="y3",
                    line=dict(color='purple', width=1.5),
                    hovertemplate='Date: %{x|%Y-%m-%d}<br>Ratio: %{y:.4f}<extra></extra>',
                ))

            if ratio_max_pos_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[ratio_max_pos_idx], y=[df_global.loc[ratio_max_pos_idx, "Ratio"]], mode="markers", xaxis="x3", yaxis="y3",
                        marker=dict(color="#00E676", size=11, symbol="triangle-up", line=dict(color="white", width=1)),
                        name="Max +Deviation",
                    ),
                )
            if ratio_max_neg_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[ratio_max_neg_idx], y=[df_global.loc[ratio_max_neg_idx, "Ratio"]], mode="markers", xaxis="x3", yaxis="y3",
                        marker=dict(color="#FF5252", size=11, symbol="triangle-down", line=dict(color="white", width=1)),
                        name="Max -Deviation",
                    ),
                )

            fig.update_layout(
                xaxis=dict(anchor="y", domain=[0, 1], matches="x3", showticklabels=False),
                yaxis=dict(domain=[0.72, 1.00], title=f"{s1} Price", tickformat="~s"),
                xaxis2=dict(anchor="y2", domain=[0, 1], matches="x3", showticklabels=False),
                yaxis2=dict(domain=[0.36, 0.64], title=f"{s2} Price", tickformat="~s"),
                xaxis3=dict(anchor="y3", domain=[0, 1], title="Date"),
                yaxis3=dict(domain=[0.00, 0.28], title=f"{s1}/{s2} Ratio"),
            )

            ratio_chart_html = build_synced_chart_html(fig, df_global, s1, s2, interval, height=780)
            components.html(ratio_chart_html, height=800, scrolling=False)
            
        except Exception as e:
            st.error(f"Error drawing matrix parameters: {e}")


# ==========================================
# PAGE 5: RSI DIVERGENCE ANALYZER
# ==========================================
elif st.session_state.current_page == "rsi_divergence":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True, key="rsi_back")

    st.title("📉 RSI Divergence Analyzer")
    st.caption("Download Yahoo Finance price data, compute RSI, and highlight bullish/bearish divergences directly in Streamlit.")

    st.sidebar.header("RSI Divergence Parameters")
    rsi_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL", key="rsi_symbol").strip().upper()
    rsi_interval = st.sidebar.selectbox("Interval", ("1d", "1wk", "1mo"), index=0, key="rsi_interval")
    rsi_years_input = st.sidebar.text_input("Lookback Window (Years)", "2", key="rsi_years")
    rsi_years = int(rsi_years_input) if rsi_years_input.isdigit() else 2
    rsi_period = st.sidebar.number_input("RSI Period", min_value=2, max_value=50, value=14, step=1, key="rsi_period")
    rsi_lookback = st.sidebar.number_input("Pivot Lookback", min_value=1, max_value=10, value=2, step=1, key="rsi_lookback")

    st.sidebar.markdown("---")
    rsi_run = st.sidebar.button("🔍 Analyze Divergences", type="primary", use_container_width=True, key="rsi_run")

    if rsi_run or rsi_symbol:
        try:
            with st.spinner(f"Fetching {rsi_symbol} data..."):
                period = f"{rsi_years}y"
                df = fetch_ohlc(rsi_symbol, period=period, interval=rsi_interval)
                df, bullish_signals, bearish_signals = detect_divergences(df, rsi_period=rsi_period, lookback=rsi_lookback)

            st.subheader(f"{rsi_symbol} RSI Divergence Analysis")
            fig = plot_rsi_divergence_chart(df, bullish_signals, bearish_signals, rsi_symbol)
            st.plotly_chart(fig, use_container_width=True)

            if bullish_signals or bearish_signals:
                summary_rows = []
                for s in bullish_signals:
                    summary_rows.append({"Type": "Bullish", "Start": s["start_date"].strftime("%Y-%m-%d"), "End": s["end_date"].strftime("%Y-%m-%d"), "Price": f"{s['start_price']:.2f}->{s['end_price']:.2f}", "RSI": f"{s['start_rsi']:.2f}->{s['end_rsi']:.2f}"})
                for s in bearish_signals:
                    summary_rows.append({"Type": "Bearish", "Start": s["start_date"].strftime("%Y-%m-%d"), "End": s["end_date"].strftime("%Y-%m-%d"), "Price": f"{s['start_price']:.2f}->{s['end_price']:.2f}", "RSI": f"{s['start_rsi']:.2f}->{s['end_rsi']:.2f}"})
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            else:
                st.info("No clear bullish or bearish divergences were detected with the current settings.")

        except Exception as e:
            st.error(f"Error analyzing RSI divergences: {e}")

# ==========================================
# PAGE 6: STOCK NEWS HUB
# ==========================================
elif st.session_state.current_page == "stock_news":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True, key="news_back")

    st.title("📰 Stock News Hub")
    st.caption("Enter a ticker to pull today's Yahoo Finance and CNBC news links and open them directly.")

    st.sidebar.header("Stock News")
    news_symbol = st.sidebar.text_input("Ticker Symbol", "AAPL", key="news_symbol").strip().upper()
    news_run = st.sidebar.button("🔎 Fetch Today’s News", type="primary", use_container_width=True, key="news_run")

    if news_run or news_symbol:
        try:
            with st.spinner(f"Fetching news for {news_symbol}..."):
                news_data = fetch_stock_news(news_symbol)

            st.subheader(f"{news_symbol} — Today's News")

            yahoo_items = news_data.get("yahoo", [])
            cnbc_items = news_data.get("cnbc", [])

            col_yahoo, col_cnbc, col_finviz = st.columns(3)
            with col_yahoo:
                st.markdown("### Yahoo Finance")
                if yahoo_items:
                    for item in yahoo_items:
                        st.markdown(f"- [{item['title']}]({item['link']})")
                else:
                    st.info("No Yahoo Finance headlines found for today.")

            with col_cnbc:
                st.markdown("### CNBC")
                if cnbc_items:
                    for item in cnbc_items:
                        st.markdown(f"- [{item['title']}]({item['link']})")
                else:
                    st.info("No CNBC headlines found for today.")

            with col_finviz:
                st.markdown("### Finviz")
                finviz_items = news_data.get("finviz", [])
                if finviz_items:
                    for item in finviz_items:
                        st.markdown(f"- [{item['title']}]({item['link']})")
                else:
                    st.info("No Finviz headlines found for today.")

        except Exception as e:
            st.error(f"Error fetching stock news: {e}")

# ==========================================
# PAGE 7: MARKET CAP RATIO STUDIO
# ==========================================
elif st.session_state.current_page == "advance_decline":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True, key="adv_back")

    st.title("📊 Advance / Decline Studio")
    st.caption("Compute how many components of an ETF or index are up vs down and visualize the breadth.")

    st.sidebar.header("Advance / Decline Parameters")
    adv_symbol = st.sidebar.text_input("ETF / Index Symbol", "SPY", key="adv_symbol").strip().upper()
    adv_interval = st.sidebar.radio("Interval Frequency", ("1d", "1wk", "1mo", "1h", "4h"), index=0, key="adv_interval")
    adv_years_input = st.sidebar.text_input("Lookback Window (Years or blank for Max)", "", key="adv_years")
    adv_years = int(adv_years_input) if adv_years_input.isdigit() else None
    st.sidebar.markdown("---")
    adv_run = st.sidebar.button("🔄 Generate Advance/Decline Chart", type="primary", use_container_width=True, key="adv_run")

    if adv_run or adv_symbol:
        try:
            with st.spinner(f"Resolving components for {adv_symbol}..."):
                comps = get_components_for_symbol(adv_symbol)

            if not comps:
                st.error(f"Could not determine components for '{adv_symbol}'. Try a different symbol or supply a list manually.")
            else:
                with st.spinner(f"Downloading component prices ({len(comps)} symbols)..."):
                    adv, dec, unc, pct_adv, net_adv, prices_df = compute_advance_decline(comps, adv_interval, adv_years)

                                # Get symbol price series for top panel
                try:
                    sym_price = download_ratio_data(adv_symbol, adv_interval, adv_years)
                except Exception:
                    # fallback to attempt basic yf history
                    t = yf.Ticker(adv_symbol)
                    h = t.history(period=f"{adv_years}y" if adv_years else "max", interval=adv_interval)
                    sym_price = h["Close"] if h is not None and not h.empty else pd.Series(dtype=float)

                # Combine into a single DF for synced crosshair
                df_adv = pd.DataFrame({
                    "Price": sym_price,
                    "Advancing": adv,
                    "Declining": dec,
                    "Unchanged": unc,
                    "Pct_Adv": pct_adv,
                    "Net_Adv": net_adv
                }).dropna(subset=["Advancing", "Declining"])
                if df_adv.index.tz is not None:
                    df_adv.index = df_adv.index.tz_localize(None)

                fig = plot_advance_decline_figure(
                    df_adv["Price"], 
                    df_adv["Advancing"], 
                    df_adv["Declining"], 
                    df_adv["Unchanged"], 
                    df_adv["Pct_Adv"], 
                    df_adv["Net_Adv"], 
                    adv_symbol, 
                    adv_interval
                )

                st.plotly_chart(fig, use_container_width=True)


                with st.expander("Show underlying sample data"):
                    display_df = pd.DataFrame({
                        "Advancing": adv, "Declining": dec, "Unchanged": unc,
                        "% Advancing": pct_adv, "Net Adv": net_adv
                    })
                    st.dataframe(display_df.tail(500), use_container_width=True)

        except Exception as e:
            st.error(f"Error computing advance/decline: {e}")
elif st.session_state.current_page == "marketcap":
    col_nav1, _ = st.columns([1, 6])
    with col_nav1: st.button("⬅ Back Home", on_click=go_to_landing, use_container_width=True, key="mcap_back")

    st.title("🏦 Market Cap Ratio Studio")
    st.caption(
        "Step 1: if Yahoo has a direct market cap for the symbol, it's treated as a stock. "
        "Step 2: otherwise it's treated as an ETF -- full holdings are pulled from State "
        "Street's (SSGA) daily holdings file and summed via Yahoo Finance market caps. "
        "Step 3: if it isn't a State Street fund, falls back to Yahoo's top-~10 holdings, scaled up."
    )

    st.sidebar.header("Market Cap Ratio Adjustments")
    mcap_s1 = st.sidebar.text_input("Symbol 1", "XLV", key="mcap_s1").strip().upper()
    mcap_s2 = st.sidebar.text_input("Symbol 2", "SPY", key="mcap_s2").strip().upper()

    mcap_interval = st.sidebar.radio("Interval Frequency", ("1d", "1wk", "1mo", "1h", "4h"), index=0, key="mcap_interval")

    mcap_period_options = ("1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max")
    mcap_period = st.sidebar.selectbox("Lookback Window (Period)", mcap_period_options, index=mcap_period_options.index("1y"), key="mcap_period")

    mcap_ma_window = st.sidebar.number_input("Ratio Moving Average Window", min_value=0, max_value=500, value=20, step=1, key="mcap_ma")

    st.sidebar.markdown("---")
    mcap_refresh = st.sidebar.button("🔄 Generate & Refresh Charts", type="primary", use_container_width=True, key="mcap_refresh")
    st.sidebar.caption(
        "ETFs are matched against State Street's (SSGA) full holdings file first "
        "(works for SPY, XLV, XLF, XLK, and other SPDR funds). Other ETFs fall back "
        "to Yahoo's top-~10 holdings, scaled up as an approximation."
    )

    if mcap_refresh:
        st.cache_data.clear()
        st.rerun()

    if mcap_s1 and mcap_s2:
        try:
            mcap_progress1 = st.empty()
            with st.spinner(f"Fetching {mcap_s1}..."):
                mc1, mcap_label1 = get_market_cap_series(mcap_s1, mcap_interval, period=mcap_period, progress_slot=mcap_progress1)

            mcap_progress2 = st.empty()
            with st.spinner(f"Fetching {mcap_s2}..."):
                mc2, mcap_label2 = get_market_cap_series(mcap_s2, mcap_interval, period=mcap_period, progress_slot=mcap_progress2)

            df_mcap = pd.concat([mc1, mc2], axis=1, join="inner")
            df_mcap.columns = [mcap_s1, mcap_s2]
            if df_mcap.index.tz is not None:
                df_mcap.index = df_mcap.index.tz_localize(None)

            if df_mcap.empty:
                st.error("No overlapping dates between the two symbols.")
                st.stop()

            df_mcap["Ratio"] = df_mcap[mcap_s1] / df_mcap[mcap_s2]
            if mcap_ma_window > 0:
                df_mcap["Ratio_MA"] = df_mcap["Ratio"].rolling(window=mcap_ma_window).mean()

            # Keep the ratio on its natural scale so the displayed value matches the actual market-cap relationship.
            mcap_ratio_scale = 1
            mcap_ratio_suffix = ""

            mcap_latest = df_mcap.iloc[-1]
            ratio_pct = mcap_latest['Ratio'] * 100
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{mcap_s1} Market Cap", fmt_cap(mcap_latest[mcap_s1]))
            m2.metric(f"{mcap_s2} Market Cap", fmt_cap(mcap_latest[mcap_s2]))
            m3.metric(f"{mcap_s1}/{mcap_s2} Ratio{mcap_ratio_suffix}", f"{ratio_pct:.2f}%")

            st.caption(f"**{mcap_s1}:** {mcap_label1}   |   **{mcap_s2}:** {mcap_label2}")

            # --- Biggest deviation from the moving average, over the selected period ---
            mcap_max_pos_idx = mcap_max_neg_idx = None
            if mcap_ma_window > 0 and "Ratio_MA" in df_mcap.columns:
                mcap_deviation = (df_mcap["Ratio"] - df_mcap["Ratio_MA"]).dropna()
                if not mcap_deviation.empty:
                    mcap_max_pos_idx = mcap_deviation.idxmax()
                    mcap_max_neg_idx = mcap_deviation.idxmin()
                    mcap_max_pos_val = mcap_deviation.loc[mcap_max_pos_idx]
                    mcap_max_neg_val = mcap_deviation.loc[mcap_max_neg_idx]

                    dt_fmt = "%Y-%m-%d %H:%M" if mcap_interval in ["1h", "4h"] else "%Y-%m-%d"
                    d1, d2 = st.columns(2)
                    d1.metric(
                        f"Biggest Positive Deviation (Ratio above MA){mcap_ratio_suffix}",
                        f"+{mcap_max_pos_val * 100:.2f}%",
                        f"on {mcap_max_pos_idx.strftime(dt_fmt)}",
                    )
                    d2.metric(
                        f"Biggest Negative Deviation (Ratio below MA){mcap_ratio_suffix}",
                        f"{mcap_max_neg_val * 100:.2f}%",
                        f"on {mcap_max_neg_idx.strftime(dt_fmt)}",
                        delta_color="inverse",
                    )

            # Plotly's native hoversubplots="axis" feature is a dead end here --
            # it's a confirmed Streamlit bug (streamlit/streamlit#9118) that
            fig = go.Figure(layout=dict(
                title=dict(text=f"{mcap_s1} vs {mcap_s2} — Market Cap & Ratio ({mcap_interval})", x=0.5, font=dict(size=14)),
                height=750,
                hovermode="closest",
                showlegend=True,
                margin=dict(t=60, b=40, l=40, r=40),
            ))

            fig.add_trace(go.Scatter(x=df_mcap.index, y=df_mcap[mcap_s1], name=mcap_s1, xaxis="x", yaxis="y", line=dict(color='#1f77b4', width=1.5)))
            fig.add_trace(go.Scatter(x=df_mcap.index, y=df_mcap[mcap_s2], name=mcap_s2, xaxis="x2", yaxis="y2", line=dict(color='#ff7f0e', width=1.5)))
            fig.add_trace(go.Scatter(x=df_mcap.index, y=df_mcap["Ratio"], name="Market-Cap Ratio", xaxis="x3", yaxis="y3", line=dict(color='purple', width=1.5)))

            if mcap_ma_window > 0:
                fig.add_trace(go.Scatter(x=df_mcap.index, y=df_mcap["Ratio_MA"], name=f"{mcap_ma_window}-Period MA", xaxis="x3", yaxis="y3", line=dict(color='green', width=1.5, dash='dash')))

            if mcap_max_pos_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[mcap_max_pos_idx], y=[df_mcap.loc[mcap_max_pos_idx, "Ratio"]], mode="markers", xaxis="x3", yaxis="y3",
                        marker=dict(color="#00E676", size=11, symbol="triangle-up", line=dict(color="white", width=1)),
                        name="Max +Deviation",
                    ),
                )
            if mcap_max_neg_idx is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[mcap_max_neg_idx], y=[df_mcap.loc[mcap_max_neg_idx, "Ratio"]], mode="markers", xaxis="x3", yaxis="y3",
                        marker=dict(color="#FF5252", size=11, symbol="triangle-down", line=dict(color="white", width=1)),
                        name="Max -Deviation",
                    ),
                )

            fig.update_layout(
                xaxis=dict(anchor="y", domain=[0, 1], matches="x3", showticklabels=False),
                yaxis=dict(domain=[0.72, 1.00], title=f"{mcap_s1} Market Cap", tickformat="~s"),
                xaxis2=dict(anchor="y2", domain=[0, 1], matches="x3", showticklabels=False),
                yaxis2=dict(domain=[0.36, 0.64], title=f"{mcap_s2} Market Cap", tickformat="~s"),
                xaxis3=dict(anchor="y3", domain=[0, 1], title="Date"),
                yaxis3=dict(domain=[0.00, 0.28], title=f"Ratio (%){mcap_ratio_suffix}", tickformat=".0%"),
            )

            mcap_chart_html = build_synced_chart_html(fig, df_mcap, mcap_s1, mcap_s2, mcap_interval, height=780)
            components.html(mcap_chart_html, height=800, scrolling=False)

            with st.expander("Show underlying data"):
                st.dataframe(df_mcap.tail(500), use_container_width=True)

        except Exception as e:
            st.error(f"Error drawing market cap ratio: {e}")