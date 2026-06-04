"""Final all-signal scoring for SwingNSE v8.

Separates:
- Technical score: price/volume/indicator setup.
- News sentiment score: already-known news/result/brokerage/corporate developments.
- Event caution score: upcoming events; caution only, never a positive score.
- Macro score and sector macro adjustment.
"""
from __future__ import annotations

import pandas as pd


def _num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def final_action_from_score(score: float, event_risk: str = "LOW", macro_risk: str = "UNKNOWN", base_action: str = "", event_caution_score: float = 0) -> str:
    event_risk = str(event_risk or "LOW").upper()
    macro_risk = str(macro_risk or "UNKNOWN").upper()
    base_action = str(base_action or "").upper()

    # Risk overrides. Upcoming events should not create a sell call by themselves, but they
    # should downgrade fresh BUY ideas to watch/cautious entries.
    if event_risk == "HIGH" and base_action in {"BUY", "WATCH"}:
        return "WATCH_EVENT_RISK" if score >= 3 else "AVOID_EVENT_RISK"
    if event_caution_score <= -2 and base_action == "BUY":
        return "WATCH_EVENT_CAUTION" if score >= 3 else "NEUTRAL"
    if macro_risk == "HIGH" and base_action == "BUY":
        return "WATCH_MACRO_RISK" if score >= 4 else "NEUTRAL"

    if score >= 8:
        return "STRONG_BUY"
    if score >= 5:
        return "BUY"
    if score >= 3:
        return "WATCH"
    if score <= -3:
        return "SELL/AVOID"
    return "NEUTRAL"


def apply_final_recommendation(recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs.copy()
    for col in ["score", "news_score", "event_caution_score", "macro_score", "sector_macro_adj", "analytics_score", "risk_quality_score", "top_news_materiality"]:
        if col not in recs.columns:
            recs[col] = 0
    for col in ["event_risk", "macro_risk", "action", "news_summary", "upcoming_event", "event_summary", "latest_event", "macro_regime", "top_news_type", "top_news_confidence", "news_audit_reason", "setup_grade", "analytics_reason", "sector", "volatility_risk"]:
        if col not in recs.columns:
            recs[col] = ""

    recs["technical_score"] = _num(recs["score"])
    recs["news_score"] = _num(recs["news_score"]).astype(int)
    recs["event_caution_score"] = _num(recs["event_caution_score"]).astype(int)
    recs["macro_score"] = _num(recs["macro_score"]).astype(int)
    recs["sector_macro_adj"] = _num(recs["sector_macro_adj"]).round(2)
    recs["analytics_score"] = _num(recs["analytics_score"]).round(2)
    recs["risk_quality_score"] = _num(recs["risk_quality_score"]).round(2)

    recs["news_adjusted_score"] = (recs["technical_score"] + recs["news_score"]).round(2)
    recs["event_adjusted_score"] = (recs["news_adjusted_score"] + recs["event_caution_score"]).round(2)
    recs["combined_signal_score"] = (
        recs["technical_score"]
        + recs["news_score"]
        + recs["event_caution_score"]
        + recs["macro_score"]
        + recs["sector_macro_adj"]
        + recs["analytics_score"]
    ).round(2)
    # Keep old final_score name compatible, but now it includes technical + news sentiment + event caution.
    recs["final_score"] = recs["event_adjusted_score"]

    recs["final_recommendation"] = recs.apply(
        lambda r: final_action_from_score(
            float(r.get("combined_signal_score", 0)),
            r.get("event_risk", "LOW"),
            r.get("macro_risk", "UNKNOWN"),
            r.get("action", ""),
            float(r.get("event_caution_score", 0)),
        ),
        axis=1,
    )

    def reason(r):
        parts = [
            f"Tech {r.get('technical_score', 0):g}",
            f"News {int(r.get('news_score', 0))} ({r.get('top_news_type','')}, {r.get('top_news_confidence','')})",
            f"Event caution {int(r.get('event_caution_score', 0))}",
            f"Macro {int(r.get('macro_score', 0))}",
            f"Sector {r.get('sector_macro_adj', 0):g}",
            f"Analytics {r.get('analytics_score', 0):g} ({r.get('setup_grade','')})",
            f"Risk {r.get('event_risk', 'LOW')}/{r.get('macro_risk', 'UNKNOWN')}",
        ]
        news = str(r.get("news_summary") or "").strip()
        event = str(r.get("upcoming_event") or "").strip()
        if news:
            parts.append(news[:160])
        analytics = str(r.get("analytics_reason") or "").strip()
        if analytics:
            parts.append("Analytics: " + analytics[:160])
        audit = str(r.get("news_audit_reason") or "").strip()
        if audit:
            parts.append("News logic: " + audit[:120])
        if event:
            parts.append(event[:160])
        return " | ".join(parts)

    recs["final_signal_reason"] = recs.apply(reason, axis=1)
    return recs
