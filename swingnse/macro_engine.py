"""Macro regime enrichment for SwingNSE.

Free-source, failure-tolerant macro layer. It fetches broadly available market
proxies where possible and allows manual overrides through macro_inputs.csv.
The technical scanner remains the source of truth; macro adjusts position bias.
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 SwingNSE/1.0"

# Sector sensitivity for macro overlay. Keep conservative; this nudges, not replaces, technical signals.
SECTOR_MACRO_MAP = {
    "BANK": {"rate_sensitive": True, "usd_benefit": False, "crude_sensitive": False},
    "FIN": {"rate_sensitive": True, "usd_benefit": False, "crude_sensitive": False},
    "NBFC": {"rate_sensitive": True, "usd_benefit": False, "crude_sensitive": False},
    "IT": {"rate_sensitive": False, "usd_benefit": True, "crude_sensitive": False},
    "PHARMA": {"rate_sensitive": False, "usd_benefit": True, "crude_sensitive": False},
    "AUTO": {"rate_sensitive": True, "usd_benefit": False, "crude_sensitive": True},
    "CEMENT": {"rate_sensitive": True, "usd_benefit": False, "crude_sensitive": True},
    "PAINT": {"rate_sensitive": False, "usd_benefit": False, "crude_sensitive": True},
    "AVIATION": {"rate_sensitive": False, "usd_benefit": False, "crude_sensitive": True},
    "OIL": {"rate_sensitive": False, "usd_benefit": False, "crude_sensitive": False},
    "ENERGY": {"rate_sensitive": False, "usd_benefit": False, "crude_sensitive": False},
}


def _fetch_yahoo_history(symbol: str, range_: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        r = requests.get(
            YAHOO_CHART.format(symbol=symbol),
            params={"range": range_, "interval": interval},
            headers={"User-Agent": USER_AGENT},
            timeout=18,
        )
        r.raise_for_status()
        data = r.json()["chart"]["result"][0]
        ts = data.get("timestamp") or []
        quote = data["indicators"]["quote"][0]
        close = data["indicators"].get("adjclose", [{}])[0].get("adjclose") or quote.get("close")
        df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s"), "close": close})
        df = df.dropna().sort_values("date")
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


def _ret(df: pd.DataFrame, days: int) -> Optional[float]:
    if df is None or len(df) < days + 2:
        return None
    latest = float(df["close"].iloc[-1])
    old = float(df["close"].iloc[-days-1])
    if old == 0 or math.isnan(old):
        return None
    return (latest / old - 1.0) * 100.0


def _latest(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    return float(df["close"].iloc[-1])


def load_manual_macro(path: str = "macro_inputs.csv") -> dict:
    """Read optional manual macro overrides.

    Expected columns: metric,value,note. Example:
    fii_5d_cr, -2500, foreign investors sold last week
    rbi_policy_bias, -1, hawkish = -1 neutral = 0 dovish = +1
    inflation_bias, -1, high inflation = -1 easing = +1
    results_season_risk, 0, -1 during peak uncertainty
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if "metric" not in df.columns or "value" not in df.columns:
            return {}
        out = {}
        for _, row in df.iterrows():
            k = str(row.get("metric", "")).strip().lower()
            if not k:
                continue
            try:
                v = float(row.get("value"))
            except Exception:
                v = row.get("value")
            out[k] = {"value": v, "note": str(row.get("note", ""))}
        return out
    except Exception:
        return {}


def compute_macro_snapshot(manual_path: str = "macro_inputs.csv") -> pd.DataFrame:
    """Fetch macro proxies and calculate a single market regime snapshot."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    nifty = _fetch_yahoo_history("^NSEI", range_="1y")
    vix = _fetch_yahoo_history("^INDIAVIX", range_="6mo")
    usdinr = _fetch_yahoo_history("INR=X", range_="6mo")
    crude = _fetch_yahoo_history("CL=F", range_="6mo")
    manual = load_manual_macro(manual_path)

    metrics = []
    def add(metric, value, score, risk, note, source="Yahoo/public/manual"):
        metrics.append({
            "as_of": now, "metric": metric, "value": value, "macro_score": score,
            "risk": risk, "note": note, "source": source,
        })

    nifty_20 = _ret(nifty, 20)
    nifty_60 = _ret(nifty, 60)
    nifty_score = 0
    if nifty_20 is not None:
        if nifty_20 > 3: nifty_score += 1
        elif nifty_20 < -3: nifty_score -= 1
    if nifty_60 is not None:
        if nifty_60 > 6: nifty_score += 1
        elif nifty_60 < -6: nifty_score -= 1
    add("nifty_trend", round(nifty_20 if nifty_20 is not None else 0, 2), nifty_score,
        "LOW" if nifty_score >= 0 else "MEDIUM", f"Nifty 20D return {nifty_20}; 60D return {nifty_60}", "Yahoo ^NSEI")

    vix_latest = _latest(vix)
    vix_20 = _ret(vix, 20)
    vix_score = 0
    vix_risk = "LOW"
    if vix_latest is not None:
        if vix_latest > 20: vix_score -= 2; vix_risk = "HIGH"
        elif vix_latest > 16: vix_score -= 1; vix_risk = "MEDIUM"
        elif vix_latest < 12: vix_score += 1
    if vix_20 is not None and vix_20 > 25:
        vix_score -= 1; vix_risk = "HIGH"
    add("india_vix", round(vix_latest or 0, 2), vix_score, vix_risk, f"India VIX latest; 20D change {vix_20}", "Yahoo ^INDIAVIX / NSE VIX proxy")

    usd_20 = _ret(usdinr, 20)
    usd_score = 0
    usd_risk = "LOW"
    if usd_20 is not None:
        if usd_20 > 2: usd_score -= 1; usd_risk = "MEDIUM"
        elif usd_20 < -2: usd_score += 1
    add("usd_inr", round(_latest(usdinr) or 0, 3), usd_score, usd_risk, f"USDINR 20D change {usd_20}", "Yahoo INR=X")

    crude_20 = _ret(crude, 20)
    crude_score = 0
    crude_risk = "LOW"
    if crude_20 is not None:
        if crude_20 > 8: crude_score -= 1; crude_risk = "MEDIUM"
        elif crude_20 < -8: crude_score += 1
    add("crude_oil", round(_latest(crude) or 0, 2), crude_score, crude_risk, f"Crude 20D change {crude_20}", "Yahoo CL=F / global proxy")

    # Manual inputs for richer India macro context.
    fii = manual.get("fii_5d_cr") or manual.get("fpi_5d_cr")
    if fii:
        v = float(fii["value"])
        sc = 1 if v > 3000 else -1 if v < -3000 else 0
        add("fii_5d_cr", v, sc, "MEDIUM" if sc < 0 else "LOW", fii.get("note", "Manual FII/FPI 5-day net flow"), "manual macro_inputs.csv")
    rbi = manual.get("rbi_policy_bias")
    if rbi:
        v = float(rbi["value"])
        sc = 1 if v > 0 else -1 if v < 0 else 0
        add("rbi_policy_bias", v, sc, "MEDIUM" if sc < 0 else "LOW", rbi.get("note", "Manual RBI rate/policy bias"), "manual macro_inputs.csv")
    inflation = manual.get("inflation_bias")
    if inflation:
        v = float(inflation["value"])
        sc = 1 if v > 0 else -1 if v < 0 else 0
        add("inflation_bias", v, sc, "MEDIUM" if sc < 0 else "LOW", inflation.get("note", "Manual inflation bias"), "manual macro_inputs.csv")
    results = manual.get("results_season_risk")
    if results:
        v = float(results["value"])
        sc = -1 if v < 0 else 0
        add("results_season_risk", v, sc, "MEDIUM" if sc < 0 else "LOW", results.get("note", "Manual results season/event risk"), "manual macro_inputs.csv")

    df = pd.DataFrame(metrics)
    df.to_csv("macro_snapshot.csv", index=False)
    return df


def macro_regime(snapshot: pd.DataFrame) -> dict:
    if snapshot is None or snapshot.empty:
        return {"macro_score": 0, "macro_risk": "UNKNOWN", "macro_regime": "NO_DATA", "macro_reason": "No macro snapshot available"}
    score = int(snapshot["macro_score"].fillna(0).sum())
    high = int((snapshot["risk"] == "HIGH").sum())
    med = int((snapshot["risk"] == "MEDIUM").sum())
    if high or score <= -3:
        risk = "HIGH"
    elif med or score < 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    if score >= 3:
        regime = "RISK_ON"
    elif score <= -3:
        regime = "RISK_OFF"
    elif score < 0:
        regime = "CAUTIOUS"
    else:
        regime = "NEUTRAL_TO_POSITIVE"
    reason = "; ".join(snapshot.sort_values("macro_score")["note"].astype(str).head(4).tolist())
    return {"macro_score": score, "macro_risk": risk, "macro_regime": regime, "macro_reason": reason}


def infer_sector(symbol: str, reason: str = "") -> str:
    txt = f"{symbol} {reason}".upper()
    for key in SECTOR_MACRO_MAP:
        if key in txt:
            return key
    return "GENERAL"


def apply_macro_to_recommendations(recs: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    recs = recs.copy()
    reg = macro_regime(snapshot)
    base = int(reg["macro_score"])
    # Cap macro overlay to avoid dominating technicals.
    capped = max(min(base, 2), -3)
    recs["macro_score"] = capped
    recs["macro_risk"] = reg["macro_risk"]
    recs["macro_regime"] = reg["macro_regime"]
    recs["macro_reason"] = reg["macro_reason"]

    # Sector-sensitive nudges using stock reason text and symbol hints. If sector data is later added, replace this inference.
    macro_sector_adj = []
    for _, r in recs.iterrows():
        # Prefer analytics sector if available; fallback to legacy symbol/reason inference.
        sector = str(r.get("sector", "") or infer_sector(str(r.get("symbol", "")), str(r.get("reason", "")))).upper()
        adj = 0
        if reg["macro_risk"] == "HIGH" and r.get("action") == "BUY":
            adj -= 1
        # USDINR weakness is often less negative / sometimes positive for exporters like IT/pharma.
        usd_row = snapshot[snapshot["metric"] == "usd_inr"] if snapshot is not None and not snapshot.empty else pd.DataFrame()
        if not usd_row.empty:
            usd_sc = int(usd_row["macro_score"].iloc[0])
            if sector in ["IT", "PHARMA", "HEALTHCARE"] and usd_sc < 0:
                adj += 1
        crude_row = snapshot[snapshot["metric"] == "crude_oil"] if snapshot is not None and not snapshot.empty else pd.DataFrame()
        if not crude_row.empty:
            crude_sc = int(crude_row["macro_score"].iloc[0])
            if sector in ["AUTO", "AUTO_ANC", "CEMENT", "PAINTS", "AVIATION"] and crude_sc < 0:
                adj -= 1
        macro_sector_adj.append(adj)
    recs["macro_sector_adj"] = macro_sector_adj
    if "sector_score" in recs.columns:
        existing_sector_score = pd.to_numeric(recs["sector_score"], errors="coerce").fillna(0).clip(-2, 2)
    else:
        existing_sector_score = pd.Series(0, index=recs.index)
    recs["sector_macro_adj"] = (existing_sector_score + pd.Series(macro_sector_adj, index=recs.index)).clip(-3, 3)

    score_col = "final_score" if "final_score" in recs.columns else "score"
    recs["macro_adjusted_score"] = recs[score_col].fillna(0).astype(float) + recs["macro_score"].fillna(0).astype(float) + recs["sector_macro_adj"].fillna(0).astype(float)

    def action(r):
        ms = r.get("macro_adjusted_score", r.get(score_col, 0))
        if r.get("macro_risk") == "HIGH" and r.get("action") == "BUY":
            return "WATCH_MACRO_RISK"
        if ms >= 7 and r.get("macro_risk") != "HIGH":
            return "BUY"
        if ms >= 4:
            return "WATCH"
        if ms <= -3:
            return "SELL/AVOID"
        return "NEUTRAL"

    recs["macro_adjusted_action"] = recs.apply(action, axis=1)
    return recs
