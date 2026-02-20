"""
MCP server for Polygon market data.
Set POLYGON_API_KEY environment variable.
For get_earnings_dates: set STEADY_API_KEY.
"""
import asyncio
import csv
import io
import os
import ssl
from datetime import date, datetime, timedelta
from typing import Optional

import aiohttp
import certifi
import numpy as np
from mcp.server.fastmcp import FastMCP
from polygon import RESTClient

_ssl_ctx = ssl.create_default_context(cafile=certifi.where())

API_KEY = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY", "")
STEADY_API_KEY = os.environ.get("STEADY_API_KEY", "")


def _get_client() -> RESTClient:
    if not API_KEY:
        raise ValueError("POLYGON_API_KEY or MASSIVE_API_KEY environment variable is required")
    return RESTClient(API_KEY)


mcp = FastMCP("Polygon Market Data", json_response=True)


@mcp.tool()
async def get_aggs(
    ticker: str,
    timespan: str = "day",
    multiplier: int = 1,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: Optional[int] = 120,
    adjusted: Optional[bool] = True,
) -> str:
    """Get aggregate bars (OHLC) for a ticker over a date range."""
    import json
    try:
        client = _get_client()
        from_ = from_date or datetime.now().strftime("%Y-%m-%d")
        to = to_date or datetime.now().strftime("%Y-%m-%d")
        data = client.get_aggs(ticker, multiplier, timespan, from_, to, adjusted=adjusted, limit=limit or 120)
        if hasattr(data, "results"):
            out = [{"t": r.timestamp, "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume} for r in data.results]
            return json.dumps({"results": out}, indent=2)
        return json.dumps({"results": []})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_ticker_details(ticker: str) -> str:
    """Get detailed information about a ticker."""
    import json
    try:
        client = _get_client()
        data = client.get_ticker_details(ticker)
        d = {"ticker": data.ticker, "name": data.name} if hasattr(data, "ticker") else {"raw": str(data)}
        return json.dumps(d, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_daily_open_close_agg(ticker: str, date: str, adjusted: Optional[bool] = True) -> str:
    """Get daily open, high, low, close for a ticker on a date."""
    import json
    try:
        client = _get_client()
        data = client.get_daily_open_close_agg(ticker, date, adjusted=adjusted)
        d = {"open": data.open, "high": data.high, "low": data.low, "close": data.close} if hasattr(data, "open") else {"raw": str(data)}
        return json.dumps(d, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _trading_days_between(d1: str, d2: str) -> int:
    try:
        return int(np.busday_count(d1, d2))
    except Exception:
        return 0


# SteadyAPI rate limit: 15 req/sec. Limit concurrency to avoid 429/timeouts.
_STEADY_SEMAPHORE: asyncio.Semaphore | None = None


def _get_steady_semaphore() -> asyncio.Semaphore:
    global _STEADY_SEMAPHORE
    if _STEADY_SEMAPHORE is None:
        _STEADY_SEMAPHORE = asyncio.Semaphore(5)  # max 5 concurrent SteadyAPI requests
    return _STEADY_SEMAPHORE


async def _fetch_earnings_for_ticker(
    session: aiohttp.ClientSession, ticker: str, api_key: str, semaphore: asyncio.Semaphore
) -> dict:
    result = {"ticker": ticker, "last_earnings_date": "", "next_earnings_date": "", "next_earnings_is_estimate": "", "past_earnings": []}
    if not api_key:
        return result

    async def _get(url: str, params: dict) -> dict | None:
        async with semaphore:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(1.0)
                        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as retry_resp:
                            if retry_resp.status == 200:
                                return await retry_resp.json()
                            return None
                    if resp.status == 200:
                        return await resp.json()
            except Exception:
                pass
            return None

    url1 = "https://api.steadyapi.com/v1/markets/stock/modules"
    params1 = {"module": "calendar-events", "ticker": ticker, "apikey": api_key}
    data1 = await _get(url1, params1)
    if data1:
        earnings = data1.get("body", {}).get("earnings", {})
        next_dates = earnings.get("earningsDate", [])
        if next_dates and next_dates[0].get("fmt"):
            result["next_earnings_date"] = next_dates[0]["fmt"]
        is_estimate = earnings.get("isEarningsDateEstimate")
        if is_estimate is not None:
            result["next_earnings_is_estimate"] = str(is_estimate).lower()
        call_dates = earnings.get("earningsCallDate", [])
        if call_dates and call_dates[0].get("fmt"):
            result["last_earnings_date"] = call_dates[0]["fmt"]

    url2 = "https://api.steadyapi.com/v1/markets/calendar/earnings"
    params2 = {"ticker": ticker, "apikey": api_key}
    data2 = await _get(url2, params2)
    if data2:
        rows = data2.get("body", {}).get("earningsSurpriseTable", {}).get("rows", [])
        for row in rows:
            date_str = row.get("dateReported", "")
            if date_str:
                try:
                    parsed = datetime.strptime(date_str, "%m/%d/%Y").date()
                    result["past_earnings"].append(parsed.strftime("%Y-%m-%d"))
                except ValueError:
                    pass
        if not result["last_earnings_date"] and result["past_earnings"]:
            result["last_earnings_date"] = result["past_earnings"][0]

    return result


@mcp.tool()
async def get_earnings_dates(tickers: str) -> str:
    """Get past and upcoming earnings dates for one or more tickers. Returns CSV with last_earnings_date, next_earnings_date, days_since_last_earnings, days_until_next_earnings, past_earnings_dates. Requires STEADY_API_KEY."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return "ticker,last_earnings_date,next_earnings_date,next_earnings_is_estimate,days_since_last_earnings,days_until_next_earnings,past_earnings_dates"
    today = date.today().isoformat()
    rows = []
    semaphore = _get_steady_semaphore()
    connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_earnings_for_ticker(session, t, STEADY_API_KEY, semaphore) for t in ticker_list]
        results = await asyncio.gather(*tasks)
    for r in results:
        days_since = ""
        if r["last_earnings_date"]:
            d = _trading_days_between(r["last_earnings_date"], today)
            days_since = str(d) if d >= 0 else ""
        days_until = ""
        if r["next_earnings_date"]:
            try:
                next_d = datetime.strptime(r["next_earnings_date"], "%Y-%m-%d").date()
                days_until = str((next_d - date.today()).days)
            except ValueError:
                pass
        past_str = "|".join(r["past_earnings"][:4])
        row = [r["ticker"], r["last_earnings_date"], r["next_earnings_date"], r["next_earnings_is_estimate"], days_since, days_until, past_str]
        rows.append(row)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ticker", "last_earnings_date", "next_earnings_date", "next_earnings_is_estimate", "days_since_last_earnings", "days_until_next_earnings", "past_earnings_dates"])
    w.writerows(rows)
    return out.getvalue()


def _timestamp_to_date(ts_ms: int) -> str:
    """Convert Polygon ms timestamp to YYYY-MM-DD."""
    try:
        return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return ""


def _compute_52w_high_low(ticker: str, bars: list, current_price_override: float | None = None) -> dict:
    """Compute 52-week high/low metrics from OHLC bars. Bars are oldest-first.
    Use current_price_override (e.g. from get_previous_close_agg) when available for accurate last price."""
    empty = {
        "ticker": ticker,
        "current_price": "",
        "high_52w": "",
        "high_52w_date": "",
        "low_52w": "",
        "low_52w_date": "",
        "pct_from_high": "",
        "pct_above_low": "",
        "range_position": "",
        "recovery_room": "",
        "floor_cushion": "",
        "new_high": "false",
        "new_low": "false",
    }
    if not bars or len(bars) < 2:
        if current_price_override is not None:
            empty["current_price"] = round(current_price_override, 2)
        return empty
    high_52w = max(float(r.high) for r in bars)
    low_52w = min(float(r.low) for r in bars)
    high_idx = next(i for i, r in enumerate(bars) if float(r.high) == high_52w)
    low_idx = next(i for i, r in enumerate(bars) if float(r.low) == low_52w)
    high_52w_date = _timestamp_to_date(bars[high_idx].timestamp)
    low_52w_date = _timestamp_to_date(bars[low_idx].timestamp)
    current_price = current_price_override if current_price_override is not None else float(bars[-1].close)
    n = len(bars)
    # new_high: high_52w_date within last 5 trading days (from today)
    # new_low: low_52w_date within last 5 trading days (from today)
    # Use today as reference, not last_bar_date — aggs may end earlier than today
    ref_date = date.today().isoformat()
    days_from_high = _trading_days_between(high_52w_date, ref_date) if high_52w_date else 999
    days_from_low = _trading_days_between(low_52w_date, ref_date) if low_52w_date else 999
    new_high = days_from_high <= 5 if high_52w_date else False
    new_low = days_from_low <= 5 if low_52w_date else False
    pct_from_high = round((current_price - high_52w) / high_52w * 100, 2) if high_52w > 0 else ""
    pct_above_low = round((current_price - low_52w) / low_52w * 100, 2) if low_52w > 0 else ""
    if high_52w > low_52w and high_52w > 0:
        range_position = round((current_price - low_52w) / (high_52w - low_52w) * 100, 2)
    else:
        range_position = ""
    recovery_room = round((high_52w - current_price) / current_price * 100, 2) if current_price > 0 else ""
    floor_cushion = round((current_price - low_52w) / current_price * 100, 2) if current_price > 0 else ""
    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "high_52w": round(high_52w, 2),
        "high_52w_date": high_52w_date,
        "low_52w": round(low_52w, 2),
        "low_52w_date": low_52w_date,
        "pct_from_high": pct_from_high,
        "pct_above_low": pct_above_low,
        "range_position": range_position,
        "recovery_room": recovery_room,
        "floor_cushion": floor_cushion,
        "new_high": str(new_high).lower(),
        "new_low": str(new_low).lower(),
    }


def _fetch_52w_for_ticker(ticker: str) -> dict:
    """Fetch 252 days of aggs and compute 52w high/low. Sync, run in executor.
    Uses get_previous_close_agg for current_price (matches vol-index / standard last-price)."""
    current_price_override: float | None = None
    try:
        client = _get_client()
        prev = client.get_previous_close_agg(ticker, adjusted=True)
        if isinstance(prev, list) and prev and hasattr(prev[0], "close"):
            current_price_override = float(prev[0].close)
        elif hasattr(prev, "results") and prev.results and hasattr(prev.results[0], "close"):
            current_price_override = float(prev.results[0].close)
    except Exception:
        pass
    to_date = date.today().isoformat()
    from_date = (date.today() - timedelta(days=400)).isoformat()
    try:
        client = _get_client()
        data = client.get_aggs(ticker, 1, "day", from_date, to_date, adjusted=True, limit=252)
        bars = data.results if hasattr(data, "results") and data.results else (data if isinstance(data, list) and data else [])
        if bars:
            return _compute_52w_high_low(ticker, bars, current_price_override)
    except Exception as e:
        err_row = {
            "ticker": ticker,
            "current_price": round(current_price_override, 2) if current_price_override is not None else "",
            "high_52w": "",
            "high_52w_date": "",
            "low_52w": "",
            "low_52w_date": "",
            "pct_from_high": "",
            "pct_above_low": "",
            "range_position": "",
            "recovery_room": "",
            "floor_cushion": "",
            "new_high": "false",
            "new_low": "false",
            "error": str(e),
        }
        return err_row
    return _compute_52w_high_low(ticker, [], current_price_override)


@mcp.tool()
async def get_52w_high_low(tickers: str) -> str:
    """Get 52-week high, low, and position context for one or more tickers. Returns CSV with current_price, high_52w, high_52w_date, low_52w, low_52w_date, pct_from_high, pct_above_low, range_position, recovery_room, floor_cushion, new_high, new_low."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return "ticker,current_price,high_52w,high_52w_date,low_52w,low_52w_date,pct_from_high,pct_above_low,range_position,recovery_room,floor_cushion,new_high,new_low"
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _fetch_52w_for_ticker, t) for t in ticker_list]
    results = await asyncio.gather(*tasks)
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ticker", "current_price", "high_52w", "high_52w_date", "low_52w", "low_52w_date", "pct_from_high", "pct_above_low", "range_position", "recovery_room", "floor_cushion", "new_high", "new_low"])
    for r in results:
        err = r.pop("error", None)
        row = [
            r["ticker"], r["current_price"], r["high_52w"], r["high_52w_date"],
            r["low_52w"], r["low_52w_date"], r["pct_from_high"], r["pct_above_low"],
            r["range_position"], r["recovery_room"], r["floor_cushion"],
            r["new_high"], r["new_low"],
        ]
        w.writerow(row)
    return out.getvalue()


# --- VIX Regime (SteadyAPI) ---
STEADY_BASE = "https://api.steadyapi.com/api"
VIX_INDICES = ["^VIX", "^VVIX", "^VIX3M"]
_vix_regime_cache: dict | None = None
_vix_regime_cache_date: str | None = None


def _percentile_rank(value: float, values: list[float]) -> float:
    """Percentile rank 0-100 of value within sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    count_below = sum(1 for v in sorted_vals if v < value)
    return round(count_below / len(sorted_vals) * 100, 1)


def _compute_trend(values: list[float], lookback: int) -> str:
    """Trend detection: rising, falling, peaking, bottoming.
    Uses min/max positions and 'near' (within 10% of range)."""
    if len(values) < 3 or lookback < 2:
        return "falling"
    window = values[-lookback:]
    if not window:
        return "falling"
    mn = min(window)
    mx = max(window)
    rng = mx - mn if mx > mn else 0.01
    current = window[-1]
    min_idx = next(i for i, v in enumerate(window) if v == mn)
    max_idx = next(i for i, v in enumerate(window) if v == mx)
    n = len(window)
    near_max = (mx - current) <= 0.10 * rng
    near_min = (current - mn) <= 0.10 * rng
    max_recent = max_idx >= n - 3
    min_recent = min_idx >= n - 3
    if near_max and max_recent:
        return "rising"
    if near_min and min_recent:
        return "falling"
    if max_idx < n - 2 and current < mx:
        return "peaking"
    if min_idx < n - 2 and current > mn:
        return "bottoming"
    return "falling" if current < (mn + mx) / 2 else "rising"


async def _fetch_steady_history(session: aiohttp.ClientSession, ticker: str) -> list[dict]:
    """Fetch 252 trading days of daily bars from SteadyAPI. Returns list of {date, close, ...} oldest-first."""
    url = f"{STEADY_BASE}/v1/markets/stock/history"
    params = {"ticker": ticker, "interval": "1d", "apikey": STEADY_API_KEY}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            body = data.get("body", {})
            if not isinstance(body, dict):
                return []
            rows = [{"date": v.get("date", ""), "close": float(v.get("close", 0) or 0)} for _, v in body.items()]
            rows.sort(key=lambda r: r["date"])
            return rows[-252:] if len(rows) > 252 else rows
    except Exception:
        return []


def _compute_vix_regime_from_bars(
    vix_bars: list[dict], vvix_bars: list[dict], vix3m_bars: list[dict], lookback_days: int
) -> dict:
    """Compute VIX regime snapshot from pre-fetched bars."""
    empty = {
        "vix": "", "vix_prev": "", "vix_5d_change": "", "vix_10d_change": "", "vix_52w_high": "", "vix_52w_low": "",
        "vix_percentile": "", "vix_trend": "", "vvix": "", "vvix_percentile": "", "vvix_trend": "",
        "vix3m": "", "term_structure_ratio": "", "term_structure": "", "term_structure_percentile": "",
        "regime": "", "call_selling_signal": "", "put_selling_signal": "",
    }

    def _closes(bars: list[dict]) -> list[float]:
        return [float(b.get("close", 0) or 0) for b in bars]

    vix_closes = _closes(vix_bars)
    vvix_closes = _closes(vvix_bars)
    vix3m_closes = _closes(vix3m_bars)

    if not vix_bars or not vix_closes:
        return empty

    n = len(vix_bars)
    vix = vix_closes[-1]
    vix_prev = vix_closes[-lookback_days] if n > lookback_days else vix_closes[0]
    vix_5d = vix_closes[-6] if n > 5 else vix_closes[0]
    vix_52w_high = max(vix_closes) if vix_closes else 0
    vix_52w_low = min(vix_closes) if vix_closes else 0
    vix_5d_change = round((vix - vix_5d) / vix_5d * 100, 1) if vix_5d and vix_5d != 0 else ""
    vix_10d_change = round((vix - vix_prev) / vix_prev * 100, 1) if vix_prev and vix_prev != 0 else ""
    vix_percentile = _percentile_rank(vix, vix_closes)
    vix_trend = _compute_trend(vix_closes, min(lookback_days, n))

    vvix = vvix_closes[-1] if vvix_closes else 0
    vvix_percentile = _percentile_rank(vvix, vvix_closes) if vvix_closes else 0
    vvix_trend = _compute_trend(vvix_closes, min(lookback_days, len(vvix_closes))) if len(vvix_closes) >= 3 else "falling"

    vix3m = vix3m_closes[-1] if vix3m_closes else 0
    term_ratio = round(vix / vix3m, 4) if vix3m and vix3m > 0 else 0
    if term_ratio < 0.95:
        term_structure = "contango"
    elif term_ratio > 1.05:
        term_structure = "backwardation"
    else:
        term_structure = "flat"

    vix3m_by_date = {b["date"]: float(b.get("close", 0) or 0) for b in vix3m_bars} if vix3m_bars else {}
    ratios_252 = []
    for b, vc in zip(vix_bars, vix_closes):
        v3 = vix3m_by_date.get(b["date"], 0)
        if v3 and v3 > 0 and vc:
            ratios_252.append(vc / v3)
    term_structure_percentile = _percentile_rank(term_ratio, ratios_252) if ratios_252 else 0

    if vix > 35 and term_structure == "backwardation":
        regime = "crisis"
    elif vix > 28:
        regime = "high_fear"
    elif vix > 22 and term_structure == "backwardation":
        regime = "stressed"
    elif 18 <= vix <= 25 and term_structure == "contango":
        regime = "elevated_normal"
    elif 14 <= vix < 18:
        regime = "low_vol"
    elif vix < 14:
        regime = "complacent"
    else:
        regime = "transitional"

    if regime == "crisis":
        call_signal = "danger"
    elif regime == "elevated_normal" and vix_trend in ("peaking", "falling") and vvix_trend in ("falling", "bottoming"):
        call_signal = "favorable"
    elif vix_percentile > 70 and term_structure == "contango":
        call_signal = "favorable"
    elif vix < 14:
        call_signal = "unfavorable"
    else:
        call_signal = "neutral"

    if regime in ("crisis", "high_fear"):
        put_signal = "favorable"
    elif regime == "stressed" and vvix_trend in ("peaking", "falling"):
        put_signal = "favorable"
    elif term_structure == "backwardation" and vix_trend == "peaking":
        put_signal = "favorable"
    elif vix < 14 and term_structure == "contango":
        put_signal = "unfavorable"
    else:
        put_signal = "neutral"

    result = {
        "vix": round(vix, 2),
        "vix_prev": round(vix_prev, 2),
        "vix_5d_change": vix_5d_change,
        "vix_10d_change": vix_10d_change,
        "vix_52w_high": round(vix_52w_high, 2),
        "vix_52w_low": round(vix_52w_low, 2),
        "vix_percentile": vix_percentile,
        "vix_trend": vix_trend,
        "vvix": round(vvix, 2) if vvix else "",
        "vvix_percentile": round(vvix_percentile, 1) if vvix else "",
        "vvix_trend": vvix_trend if vvix else "",
        "vix3m": round(vix3m, 2) if vix3m else "",
        "term_structure_ratio": term_ratio if term_ratio else "",
        "term_structure": term_structure,
        "term_structure_percentile": round(term_structure_percentile, 1) if ratios_252 else "",
        "regime": regime,
        "call_selling_signal": call_signal,
        "put_selling_signal": put_signal,
    }
    return result


@mcp.tool()
async def get_vix_regime(lookback_days: int = 10) -> str:
    """Get VIX market regime snapshot using ^VIX, ^VVIX, ^VIX3M from SteadyAPI. Returns CSV with vix, vix_prev, vix_5d_change, vix_10d_change, vix_52w_high, vix_52w_low, vix_percentile, vix_trend, vvix, vvix_percentile, vvix_trend, vix3m, term_structure_ratio, term_structure, term_structure_percentile, regime, call_selling_signal, put_selling_signal. Requires STEADY_API_KEY. Cached per trading day."""
    global _vix_regime_cache, _vix_regime_cache_date
    if not STEADY_API_KEY:
        return "error: STEADY_API_KEY not set"
    today = date.today().isoformat()
    if _vix_regime_cache is not None and _vix_regime_cache_date == today:
        result = _vix_regime_cache
    else:
        connector = aiohttp.TCPConnector(ssl=_ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_fetch_steady_history(session, t) for t in VIX_INDICES]
            vix_bars, vvix_bars, vix3m_bars = await asyncio.gather(*tasks)
        result = _compute_vix_regime_from_bars(vix_bars, vvix_bars, vix3m_bars, lookback_days)
        _vix_regime_cache = result
        _vix_regime_cache_date = today
    out = io.StringIO()
    w = csv.writer(out)
    headers = [
        "vix", "vix_prev", "vix_5d_change", "vix_10d_change", "vix_52w_high", "vix_52w_low",
        "vix_percentile", "vix_trend", "vvix", "vvix_percentile", "vvix_trend",
        "vix3m", "term_structure_ratio", "term_structure", "term_structure_percentile",
        "regime", "call_selling_signal", "put_selling_signal",
    ]
    w.writerow(headers)
    w.writerow([result.get(h, "") for h in headers])
    return out.getvalue()


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
