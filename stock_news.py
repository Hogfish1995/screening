import re
from datetime import datetime, timezone
from typing import List, Dict
import requests
import xml.etree.ElementTree as ET

HEADERS = {"User-Agent": "Mozilla/5.0"}


def _parse_rss_items(xml_text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        items.append({"title": title, "link": link, "pubDate": pub_date})
    return items


def _is_today(pub_date: str) -> bool:
    if not pub_date:
        return True
    try:
        dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
    except ValueError:
        try:
            dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            return True
    now = datetime.now(dt.tzinfo or timezone.utc)
    return dt.date() == now.date()


def fetch_yahoo_news(symbol: str) -> List[Dict[str, str]]:
    symbol = symbol.strip().upper()
    if not symbol:
        return []
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception:
        return []
    items = _parse_rss_items(response.text)
    return [it for it in items if _is_today(it.get("pubDate", ""))][:8]


def fetch_cnbc_news(symbol: str) -> List[Dict[str, str]]:
    symbol = symbol.strip().upper()
    if not symbol:
        return []
    search_url = f"https://www.cnbc.com/search/?query={symbol}"
    try:
        response = requests.get(search_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception:
        return [{"title": "CNBC search results", "link": search_url, "pubDate": ""}]

    # Try a lightweight scrape of the page for actual article links if available.
    urls = []
    for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', response.text, re.I | re.S):
        href = match.group(1)
        text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        if not text:
            continue
        if href.startswith("http") and "cnbc.com" in href:
            if href not in urls:
                urls.append(href)
        elif href.startswith("/") and not href.startswith("//"):
            full_url = "https://www.cnbc.com" + href
            if full_url not in urls:
                urls.append(full_url)
    if urls:
        return [{"title": "CNBC search results", "link": search_url, "pubDate": ""}] + [
            {"title": f"CNBC article {i+1}", "link": url, "pubDate": ""}
            for i, url in enumerate(urls[:5])
        ]
    return [{"title": "CNBC search results", "link": search_url, "pubDate": ""}]


def fetch_finviz_news(symbol: str) -> List[Dict[str, str]]:
    symbol = symbol.strip().upper()
    if not symbol:
        return []
    quote_url = f"https://finviz.com/quote.ashx?t={symbol}"
    try:
        response = requests.get(quote_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except Exception:
        return [{"title": f"Finviz quote for {symbol}", "link": quote_url, "pubDate": ""}]

    items: List[Dict[str, str]] = []
    seen = set()
    for match in re.finditer(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', response.text, re.I | re.S):
        href = match.group(1).strip()
        title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        if not title:
            continue
        if href.startswith("/"):
            href = "https://finviz.com" + href
        elif not href.startswith("http"):
            href = f"https://finviz.com/{href.lstrip('/')}"
        if href.startswith("http") and "finviz.com" in href and href not in seen:
            seen.add(href)
            items.append({"title": title or f"Finviz link for {symbol}", "link": href, "pubDate": ""})
        if len(items) >= 6:
            break

    if items:
        return [{"title": f"Finviz page for {symbol}", "link": quote_url, "pubDate": ""}] + items[:5]
    return [{"title": f"Finviz page for {symbol}", "link": quote_url, "pubDate": ""}]


def fetch_stock_news(symbol: str) -> Dict[str, List[Dict[str, str]]]:
    return {
        "yahoo": fetch_yahoo_news(symbol),
        "cnbc": fetch_cnbc_news(symbol),
        "finviz": fetch_finviz_news(symbol),
    }
