"""Advanced analytics layer for SwingNSE v10.

Adds institutional-style score components that are independent and auditable:
- sector breadth/relative-strength score
- liquidity quality score
- volatility/risk score
- trend/momentum/volume component scores
- position-size suggestion
- setup grade and explainability
"""
from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Dict

import numpy as np
import pandas as pd

# Broad default map for common NSE symbols. Users can extend/override via sector_map.csv.
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    # IT
    "TCS":"IT", "INFY":"IT", "WIPRO":"IT", "HCLTECH":"IT", "TECHM":"IT", "LTIM":"IT", "LTTS":"IT", "MPHASIS":"IT", "COFORGE":"IT", "PERSISTENT":"IT", "OFSS":"IT", "KPITTECH":"IT", "TATAELXSI":"IT",
    # Banks/Financials
    "HDFCBANK":"BANK", "ICICIBANK":"BANK", "SBIN":"BANK", "AXISBANK":"BANK", "KOTAKBANK":"BANK", "INDUSINDBK":"BANK", "BANKBARODA":"BANK", "PNB":"BANK", "CANBK":"BANK", "FEDERALBNK":"BANK", "IDFCFIRSTB":"BANK", "AUBANK":"BANK", "BANDHANBNK":"BANK", "EQUITASBNK":"BANK", "UJJIVANSFB":"BANK",
    "BAJFINANCE":"NBFC", "BAJAJFINSV":"NBFC", "CHOLAFIN":"NBFC", "MUTHOOTFIN":"NBFC", "SHRIRAMFIN":"NBFC", "MANAPPURAM":"NBFC", "LICHSGFIN":"NBFC", "SBICARD":"NBFC", "HDFCAMC":"FINANCIALS", "ICICIGI":"FINANCIALS", "ICICIPRULI":"FINANCIALS", "SBILIFE":"FINANCIALS", "HDFCLIFE":"FINANCIALS", "LICI":"FINANCIALS", "MCX":"FINANCIALS", "CDSL":"FINANCIALS", "CAMS":"FINANCIALS", "BSE":"FINANCIALS",
    # Energy/Oil/Gas/Utilities
    "RELIANCE":"ENERGY", "ONGC":"OIL_GAS", "OIL":"OIL_GAS", "IOC":"OIL_GAS", "BPCL":"OIL_GAS", "HINDPETRO":"OIL_GAS", "GAIL":"OIL_GAS", "PETRONET":"OIL_GAS", "MGL":"OIL_GAS", "IGL":"OIL_GAS", "ATGL":"OIL_GAS",
    "NTPC":"POWER", "POWERGRID":"POWER", "TATAPOWER":"POWER", "ADANIGREEN":"POWER", "ADANIPOWER":"POWER", "NHPC":"POWER", "SJVN":"POWER", "JSWENERGY":"POWER", "TORNTPOWER":"POWER", "CESC":"POWER",
    # FMCG/Consumer
    "HINDUNILVR":"FMCG", "ITC":"FMCG", "NESTLEIND":"FMCG", "BRITANNIA":"FMCG", "DABUR":"FMCG", "MARICO":"FMCG", "COLPAL":"FMCG", "GODREJCP":"FMCG", "TATACONSUM":"FMCG", "VBL":"FMCG", "UBL":"FMCG", "UNITDSPR":"FMCG",
    "DMART":"RETAIL", "TRENT":"RETAIL", "NYKAA":"RETAIL", "ABFRL":"RETAIL", "SHOPERSTOP":"RETAIL", "METROBRAND":"RETAIL", "ZOMATO":"CONSUMER_TECH", "PAYTM":"FINTECH", "POLICYBZR":"FINTECH", "NAUKRI":"INTERNET",
    # Auto
    "MARUTI":"AUTO", "TATAMOTORS":"AUTO", "M&M":"AUTO", "BAJAJ-AUTO":"AUTO", "HEROMOTOCO":"AUTO", "EICHERMOT":"AUTO", "TVSMOTOR":"AUTO", "ASHOKLEY":"AUTO", "ESCORTS":"AUTO", "BOSCHLTD":"AUTO_ANC", "MOTHERSON":"AUTO_ANC", "SONACOMS":"AUTO_ANC", "UNOMINDA":"AUTO_ANC", "BALKRISIND":"AUTO_ANC", "MRF":"AUTO_ANC", "APOLLOTYRE":"AUTO_ANC", "CEATLTD":"AUTO_ANC", "EXIDEIND":"AUTO_ANC", "AMARAJABAT":"AUTO_ANC",
    # Pharma/Healthcare
    "SUNPHARMA":"PHARMA", "DRREDDY":"PHARMA", "CIPLA":"PHARMA", "DIVISLAB":"PHARMA", "LUPIN":"PHARMA", "AUROPHARMA":"PHARMA", "TORNTPHARM":"PHARMA", "ALKEM":"PHARMA", "ZYDUSLIFE":"PHARMA", "GLENMARK":"PHARMA", "BIOCON":"PHARMA", "LAURUSLABS":"PHARMA", "IPCALAB":"PHARMA", "ABBOTINDIA":"PHARMA", "MAXHEALTH":"HEALTHCARE", "APOLLOHOSP":"HEALTHCARE", "FORTIS":"HEALTHCARE", "LALPATHLAB":"HEALTHCARE", "METROPOLIS":"HEALTHCARE",
    # Metals/Mining/Cement/Materials
    "TATASTEEL":"METALS", "JSWSTEEL":"METALS", "JINDALSTEL":"METALS", "SAIL":"METALS", "HINDALCO":"METALS", "NATIONALUM":"METALS", "VEDL":"METALS", "NMDC":"METALS", "HINDZINC":"METALS", "COALINDIA":"MINING",
    "ULTRACEMCO":"CEMENT", "SHREECEM":"CEMENT", "AMBUJACEM":"CEMENT", "ACC":"CEMENT", "DALBHARAT":"CEMENT", "RAMCOCEM":"CEMENT", "JKCEMENT":"CEMENT",
    "ASIANPAINT":"PAINTS", "BERGEPAINT":"PAINTS", "KANSAINER":"PAINTS", "PIDILITIND":"CHEMICALS", "SRF":"CHEMICALS", "AARTIIND":"CHEMICALS", "PIIND":"CHEMICALS", "TATACHEM":"CHEMICALS", "DEEPAKNTR":"CHEMICALS", "NAVINFLUOR":"CHEMICALS", "UPL":"CHEMICALS",
    # Infra/Capital goods/Realty/Telecom
    "LT":"CAPITAL_GOODS", "SIEMENS":"CAPITAL_GOODS", "ABB":"CAPITAL_GOODS", "BHEL":"CAPITAL_GOODS", "CGPOWER":"CAPITAL_GOODS", "POLYCAB":"CAPITAL_GOODS", "HAVELLS":"CAPITAL_GOODS", "DIXON":"ELECTRONICS", "VOLTAS":"CONSUMER_DURABLES", "BLUESTARCO":"CONSUMER_DURABLES", "CROMPTON":"CONSUMER_DURABLES", "TITAN":"CONSUMER_DURABLES",
    "BHARTIARTL":"TELECOM", "IDEA":"TELECOM", "INDUSTOWER":"TELECOM", "TATACOMM":"TELECOM",
    "DLF":"REALTY", "LODHA":"REALTY", "GODREJPROP":"REALTY", "OBEROIRLTY":"REALTY", "PRESTIGE":"REALTY", "PHOENIXLTD":"REALTY",
    "ADANIENT":"CONGLOMERATE", "ADANIPORTS":"LOGISTICS", "CONCOR":"LOGISTICS", "DELHIVERY":"LOGISTICS", "INDIGO":"AVIATION", "IRCTC":"TRAVEL", "INDHOTEL":"HOTELS", "LEMONTREE":"HOTELS", "JUBLFOOD":"QSR", "DEVYANI":"QSR", "SAPPHIRE":"QSR",
}


def _safe_num(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _load_sector_map(path: str = "sector_map.csv") -> dict:
    sector_map = DEFAULT_SECTOR_MAP.copy()
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_csv(p)
            sym_col = next((c for c in df.columns if c.lower() in {"symbol", "stock"}), None)
            sec_col = next((c for c in df.columns if c.lower() in {"sector", "industry"}), None)
            if sym_col and sec_col:
                for _, r in df.iterrows():
                    s = str(r.get(sym_col, "")).upper().strip()
                    sec = str(r.get(sec_col, "")).upper().strip().replace(" ", "_")
                    if s and sec and sec != "NAN":
                        sector_map[s] = sec
        except Exception:
            pass
    return sector_map


def infer_sector(symbol: str, sector_map: dict | None = None) -> str:
    sector_map = sector_map or _load_sector_map()
    sym = str(symbol or "").upper().strip()
    if sym in sector_map:
        return sector_map[sym]
    # Conservative heuristics only. Unknown remains UNKNOWN so user can extend sector_map.csv.
    if any(k in sym for k in ["BANK"]): return "BANK"
    if any(k in sym for k in ["FIN", "CAP", "CREDIT"]): return "FINANCIALS"
    if any(k in sym for k in ["PHARMA", "LIFE", "BIO"]): return "PHARMA"
    if any(k in sym for k in ["STEEL", "METAL", "ALUM", "ZINC"]): return "METALS"
    if any(k in sym for k in ["CEM"]): return "CEMENT"
    if any(k in sym for k in ["POWER", "ENERGY", "GREEN"]): return "POWER"
    return "UNKNOWN"


def _recent_returns(g: pd.DataFrame, periods=(5, 20, 60)) -> dict:
    g = g.sort_values("date")
    out = {}
    if g.empty:
        return {f"ret_{p}d": np.nan for p in periods}
    latest = _safe_num(g["close"].iloc[-1], np.nan)
    for p in periods:
        if len(g) > p and not pd.isna(latest):
            old = _safe_num(g["close"].iloc[-p-1], np.nan)
            out[f"ret_{p}d"] = round((latest / old - 1) * 100, 2) if old and not pd.isna(old) else np.nan
        else:
            out[f"ret_{p}d"] = np.nan
    return out


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def build_symbol_analytics(with_ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return symbol-level analytics and sector snapshot.

    with_ind must contain price rows with technical columns from indicators.add_indicators.
    """
    if with_ind is None or with_ind.empty:
        return pd.DataFrame(), pd.DataFrame()
    sector_map = _load_sector_map()
    df = with_ind.sort_values(["symbol", "date"]).copy()
    latest = df.groupby("symbol", as_index=False).tail(1).copy()
    ret_rows = []
    for sym, g in df.groupby("symbol"):
        r = {"symbol": sym, **_recent_returns(g)}
        # compute avg traded value proxy from price*volume, rupees; turn into crore
        g = g.sort_values("date").copy()
        traded_value = pd.to_numeric(g.get("value"), errors="coerce")
        if traded_value is None or traded_value.fillna(0).abs().sum() == 0:
            traded_value = pd.to_numeric(g["close"], errors="coerce") * pd.to_numeric(g.get("volume"), errors="coerce")
        r["avg_traded_value_20_cr"] = round(float(traded_value.tail(20).mean()) / 1e7, 2) if len(traded_value.dropna()) else np.nan
        # ATR percentage as volatility proxy
        last = g.iloc[-1]
        close = _safe_num(last.get("close"), np.nan)
        atr = _safe_num(last.get("atr14"), np.nan)
        r["atr_pct"] = round((atr / close) * 100, 2) if close and not pd.isna(close) and not pd.isna(atr) else np.nan
        ret_rows.append(r)
    ret_df = pd.DataFrame(ret_rows)
    latest = latest.merge(ret_df, on="symbol", how="left")
    latest["sector"] = latest["symbol"].map(lambda s: infer_sector(s, sector_map))

    # Component scoring
    rows = []
    for _, r in latest.iterrows():
        close = _safe_num(r.get("close"), np.nan)
        sma20 = _safe_num(r.get("sma20"), np.nan)
        sma50 = _safe_num(r.get("sma50"), np.nan)
        sma200 = _safe_num(r.get("sma200"), np.nan)
        rsi = _safe_num(r.get("rsi14"), np.nan)
        macd_h = _safe_num(r.get("macd_hist"), np.nan)
        adx = _safe_num(r.get("adx14"), np.nan)
        vol_ratio = _safe_num(r.get("vol_ratio"), np.nan)
        atr_pct = _safe_num(r.get("atr_pct"), np.nan)
        avg_val = _safe_num(r.get("avg_traded_value_20_cr"), np.nan)
        ret20 = _safe_num(r.get("ret_20d"), np.nan)
        ret60 = _safe_num(r.get("ret_60d"), np.nan)

        trend_score = 0
        if close and not pd.isna(sma20) and close > sma20: trend_score += 1
        if not pd.isna(sma20) and not pd.isna(sma50) and sma20 > sma50: trend_score += 1
        if close and not pd.isna(sma50) and close > sma50: trend_score += 1
        if close and not pd.isna(sma200) and close > sma200: trend_score += 1
        if not pd.isna(adx) and adx >= 22: trend_score += 1
        if not pd.isna(ret20) and ret20 > 0: trend_score += 1
        trend_score = _clip(trend_score, -3, 6)

        momentum_score = 0
        if not pd.isna(rsi):
            if 45 <= rsi <= 65: momentum_score += 2
            elif 35 <= rsi < 45 or 65 < rsi <= 72: momentum_score += 1
            elif rsi > 78: momentum_score -= 2
            elif rsi < 30: momentum_score -= 1
        if not pd.isna(macd_h): momentum_score += 1 if macd_h > 0 else -1
        if not pd.isna(ret20):
            if 3 <= ret20 <= 15: momentum_score += 1
            elif ret20 > 25: momentum_score -= 1
        momentum_score = _clip(momentum_score, -3, 4)

        volume_score = 0
        if not pd.isna(vol_ratio):
            if vol_ratio >= 2: volume_score += 2
            elif vol_ratio >= 1.2: volume_score += 1
            elif vol_ratio < 0.6: volume_score -= 1
        volume_score = _clip(volume_score, -2, 2)

        liquidity_score = 0
        if not pd.isna(avg_val):
            if avg_val >= 100: liquidity_score = 3
            elif avg_val >= 50: liquidity_score = 2
            elif avg_val >= 10: liquidity_score = 1
            elif avg_val < 2: liquidity_score = -2
            elif avg_val < 5: liquidity_score = -1
        liquidity_score = _clip(liquidity_score, -2, 3)

        volatility_score = 0
        volatility_risk = "UNKNOWN"
        if not pd.isna(atr_pct):
            if atr_pct <= 2.5: volatility_score = 1; volatility_risk = "LOW"
            elif atr_pct <= 5: volatility_score = 0; volatility_risk = "MEDIUM"
            elif atr_pct <= 8: volatility_score = -1; volatility_risk = "HIGH"
            else: volatility_score = -2; volatility_risk = "VERY_HIGH"

        rows.append({
            "symbol": r["symbol"], "sector": r["sector"],
            "ret_5d": r.get("ret_5d"), "ret_20d": r.get("ret_20d"), "ret_60d": r.get("ret_60d"),
            "avg_traded_value_20_cr": avg_val, "atr_pct": atr_pct,
            "trend_score": trend_score, "momentum_score": momentum_score, "volume_score": volume_score,
            "liquidity_score": liquidity_score, "volatility_score": volatility_score, "volatility_risk": volatility_risk,
        })
    sym_analytics = pd.DataFrame(rows)

    # Sector snapshot: breadth and relative strength from latest stock analytics.
    sx = latest.merge(sym_analytics[["symbol","sector","ret_5d","ret_20d","ret_60d"]], on=["symbol","sector"], how="left", suffixes=("", "_a"))
    sector_rows = []
    market_ret20 = pd.to_numeric(sx["ret_20d"], errors="coerce").median()
    for sector, g in sx.groupby("sector"):
        if sector == "UNKNOWN" or len(g) < 2:
            continue
        ret20_med = pd.to_numeric(g["ret_20d"], errors="coerce").median()
        ret60_med = pd.to_numeric(g["ret_60d"], errors="coerce").median()
        above20 = ((pd.to_numeric(g["close"], errors="coerce") > pd.to_numeric(g["sma20"], errors="coerce")).mean() * 100) if "sma20" in g else np.nan
        above50 = ((pd.to_numeric(g["close"], errors="coerce") > pd.to_numeric(g["sma50"], errors="coerce")).mean() * 100) if "sma50" in g else np.nan
        rel = ret20_med - market_ret20 if not pd.isna(ret20_med) and not pd.isna(market_ret20) else 0
        sector_score = 0
        if not pd.isna(ret20_med):
            if ret20_med > 4: sector_score += 2
            elif ret20_med > 0: sector_score += 1
            elif ret20_med < -4: sector_score -= 2
            elif ret20_med < 0: sector_score -= 1
        if not pd.isna(above20):
            if above20 >= 65: sector_score += 1
            elif above20 <= 35: sector_score -= 1
        if rel > 2: sector_score += 1
        elif rel < -2: sector_score -= 1
        sector_score = _clip(sector_score, -3, 3)
        sector_rows.append({
            "sector": sector, "symbols": int(len(g)), "median_ret_20d": round(ret20_med,2) if not pd.isna(ret20_med) else np.nan,
            "median_ret_60d": round(ret60_med,2) if not pd.isna(ret60_med) else np.nan,
            "breadth_above_sma20_pct": round(above20,1) if not pd.isna(above20) else np.nan,
            "breadth_above_sma50_pct": round(above50,1) if not pd.isna(above50) else np.nan,
            "relative_strength_vs_market_20d": round(rel,2), "sector_score": int(sector_score),
        })
    sector_snapshot = pd.DataFrame(sector_rows).sort_values("sector_score", ascending=False) if sector_rows else pd.DataFrame()
    return sym_analytics, sector_snapshot


def apply_advanced_analytics(recs: pd.DataFrame, with_ind: pd.DataFrame) -> pd.DataFrame:
    if recs is None or recs.empty:
        return recs
    sym_analytics, sector_snapshot = build_symbol_analytics(with_ind)
    out = recs.copy()
    if not sym_analytics.empty:
        out = out.merge(sym_analytics, on="symbol", how="left")
    else:
        for c in ["sector","ret_5d","ret_20d","ret_60d","avg_traded_value_20_cr","atr_pct","trend_score","momentum_score","volume_score","liquidity_score","volatility_score","volatility_risk"]:
            out[c] = "UNKNOWN" if c in {"sector","volatility_risk"} else 0
    if not sector_snapshot.empty:
        out = out.merge(sector_snapshot[["sector","sector_score","breadth_above_sma20_pct","relative_strength_vs_market_20d"]], on="sector", how="left")
    else:
        out["sector_score"] = 0
        out["breadth_above_sma20_pct"] = np.nan
        out["relative_strength_vs_market_20d"] = 0
    out["sector_score"] = pd.to_numeric(out.get("sector_score"), errors="coerce").fillna(0).astype(int)
    # Replace weak/old macro sector adjustment with actual sector breadth/RS, but keep capped.
    out["sector_macro_adj"] = out["sector_score"].clip(-2, 2)
    # Risk score and position sizing
    out["risk_quality_score"] = (
        pd.to_numeric(out.get("liquidity_score"), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("volatility_score"), errors="coerce").fillna(0)
    ).clip(-3, 4)
    out["analytics_score"] = (
        pd.to_numeric(out.get("trend_score"), errors="coerce").fillna(0) * 0.45
        + pd.to_numeric(out.get("momentum_score"), errors="coerce").fillna(0) * 0.35
        + pd.to_numeric(out.get("volume_score"), errors="coerce").fillna(0) * 0.25
        + pd.to_numeric(out.get("liquidity_score"), errors="coerce").fillna(0) * 0.25
        + pd.to_numeric(out.get("volatility_score"), errors="coerce").fillna(0) * 0.35
        + pd.to_numeric(out.get("sector_score"), errors="coerce").fillna(0) * 0.35
    ).round(2)
    # Suggested capital risk per trade, not advice; lower for poor liquidity/high volatility/event risk.
    def pos_pct(r):
        base = 1.5
        if _safe_num(r.get("liquidity_score")) >= 2: base += 0.5
        if _safe_num(r.get("volatility_score")) < 0: base -= 0.5
        if str(r.get("event_risk", "")).upper() == "HIGH": base -= 0.75
        if str(r.get("macro_risk", "")).upper() == "HIGH": base -= 0.5
        if _safe_num(r.get("sector_score")) < 0: base -= 0.25
        return round(_clip(base, 0.5, 2.5), 2)
    out["suggested_risk_pct"] = out.apply(pos_pct, axis=1)
    def grade(r):
        score = _safe_num(r.get("combined_signal_score", r.get("score", 0))) + _safe_num(r.get("analytics_score"))
        if score >= 10 and _safe_num(r.get("liquidity_score")) >= 1 and str(r.get("event_risk", "LOW")).upper() != "HIGH": return "A"
        if score >= 7: return "B"
        if score >= 4: return "C"
        if score >= 1: return "D"
        return "E"
    out["setup_grade"] = out.apply(grade, axis=1)
    def ar(r):
        return (
            f"Trend {r.get('trend_score',0)}, Momentum {r.get('momentum_score',0)}, Volume {r.get('volume_score',0)}, "
            f"Liquidity {r.get('liquidity_score',0)} ({r.get('avg_traded_value_20_cr',0)}cr), "
            f"VolRisk {r.get('volatility_risk','')}, Sector {r.get('sector','UNKNOWN')} {r.get('sector_score',0)}"
        )
    out["analytics_reason"] = out.apply(ar, axis=1)
    # Save sector snapshot for UI/debugging.
    if not sector_snapshot.empty:
        sector_snapshot.to_csv("sector_snapshot.csv", index=False)
    return out
