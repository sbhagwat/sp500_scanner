"""Fetch S&P 500 fundamentals and write site/data.json."""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[dict]:
    html = requests.get(SP500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector"]].to_dict("records")


def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _cagr(latest: float, earliest: float, years: int) -> float | None:
    if latest is None or earliest is None or earliest <= 0 or years <= 0:
        return None
    try:
        return (latest / earliest) ** (1 / years) - 1
    except Exception:
        return None


def _next_earnings(t: yf.Ticker) -> str | None:
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                return str(ed[0])
            if ed:
                return str(ed)
        if hasattr(cal, "loc") and "Earnings Date" in cal.index:
            v = cal.loc["Earnings Date"].iloc[0]
            return str(v) if v is not None else None
    except Exception:
        pass
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            future = ed[ed.index >= pd.Timestamp.utcnow().tz_localize(None)]
            if not future.empty:
                return str(future.index.min().date())
    except Exception:
        pass
    return None


def _capex_latest(t: yf.Ticker) -> float | None:
    """Latest annual capital expenditure as a positive absolute value."""
    try:
        cf = t.cashflow
        if cf is None or cf.empty:
            return None
        idx = cf.index.astype(str)
        for key in ("Capital Expenditure", "Capital Expenditures"):
            if key in idx:
                s = cf.loc[cf.index.astype(str) == key].iloc[0].dropna().sort_index()
                if not s.empty:
                    return abs(float(s.iloc[-1]))
    except Exception:
        pass
    return None


def _intensity_label(capex_to_rev: float | None) -> str | None:
    if capex_to_rev is None:
        return None
    if capex_to_rev < 0.05:
        return "light"
    if capex_to_rev < 0.15:
        return "moderate"
    return "heavy"


def _fcf_series(t: yf.Ticker) -> pd.Series:
    """Annual Free Cash Flow series, oldest -> newest."""
    try:
        cf = t.cashflow
        if cf is None or cf.empty:
            return pd.Series(dtype=float)
        idx = cf.index.astype(str)
        for key in ("Free Cash Flow", "FreeCashFlow"):
            if key in idx:
                s = cf.loc[cf.index.astype(str) == key].iloc[0]
                s = s.dropna().sort_index()
                return s
        # Fallback: OCF - CapEx
        ocf_keys = ("Operating Cash Flow", "Total Cash From Operating Activities")
        capex_keys = ("Capital Expenditure", "Capital Expenditures")
        ocf = capex = None
        for k in ocf_keys:
            if k in idx:
                ocf = cf.loc[cf.index.astype(str) == k].iloc[0]
                break
        for k in capex_keys:
            if k in idx:
                capex = cf.loc[cf.index.astype(str) == k].iloc[0]
                break
        if ocf is not None and capex is not None:
            s = (ocf + capex).dropna().sort_index()  # capex is negative
            return s
    except Exception:
        pass
    return pd.Series(dtype=float)


def fetch_one(row: dict) -> dict:
    ticker = row["ticker"]
    out = dict(row)
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        out["price"] = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        out["market_cap"] = _safe(info.get("marketCap"))
        out["pe"] = _safe(info.get("trailingPE"))
        out["forward_pe"] = _safe(info.get("forwardPE"))
        out["ps"] = _safe(info.get("priceToSalesTrailing12Months"))
        out["pb"] = _safe(info.get("priceToBook"))
        out["next_earnings"] = _next_earnings(t)

        revenue = _safe(info.get("totalRevenue"))
        market_cap = out["market_cap"]

        capex = _capex_latest(t)
        out["capex"] = capex
        out["capex_to_revenue"] = (capex / revenue) if (capex is not None and revenue) else None
        out["asset_intensity"] = _intensity_label(out["capex_to_revenue"])

        fcf = _fcf_series(t)
        if not fcf.empty:
            latest = float(fcf.iloc[-1])
            out["fcf_latest"] = latest
            if len(fcf) >= 2:
                prev = float(fcf.iloc[-2])
                out["fcf_yoy"] = (latest / prev - 1) if prev > 0 else None
            if len(fcf) >= 4:
                out["fcf_cagr_3y"] = _cagr(latest, float(fcf.iloc[-4]), 3)
            out["fcf_margin"] = (latest / revenue) if revenue else None
            out["fcf_yield"] = (latest / market_cap) if market_cap else None
        else:
            for k in ("fcf_latest", "fcf_yoy", "fcf_cagr_3y", "fcf_margin", "fcf_yield"):
                out.setdefault(k, None)

        out["error"] = None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    tickers = get_sp500_tickers()
    print(f"[sp500] {len(tickers)} tickers", flush=True)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, row): row["ticker"] for row in tickers}
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(tickers):
                print(f"[fetch] {done}/{len(tickers)}", flush=True)

    results.sort(key=lambda r: (r.get("market_cap") or 0), reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "count": len(results),
        "rows": results,
    }
    out_path = SITE / "data.json"
    out_path.write_text(json.dumps(payload, default=str))
    print(f"[write] {out_path} ({len(results)} rows)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
