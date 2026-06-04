"""Structured news taxonomy engine for SwingNSE v9.

v9 replaces one-step keyword scoring with a cleaner pipeline:
raw item -> source quality -> taxonomy type -> materiality -> sentiment ->
freshness -> impact score. This reduces anecdotal patching and makes each
classification auditable in the UI.

The engine is intentionally conservative: only fresh, material catalysts affect
recommendations. Historical shareholder-return / generic price-move articles are
shown for context but score zero.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import html
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SwingNSE/2.0"
)

RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# ---------- Taxonomy dictionaries ----------
# Order matters. Higher-priority groups are evaluated first.
TAXONOMY = [
    {
        "type": "LOW_VALUE_COMMENTARY",
        "sentiment": "NEUTRAL",
        "materiality": 0,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "HIGH",
        "keywords": [
            "investors are sitting on a loss", "if they invested", "shareholders are down",
            "shareholders have endured", "shareholders have lost", "shareholders have gained",
            "shareholders would have", "total shareholder return", "invested a year ago",
            "invested five years ago", "invested three years ago", "held for five years",
            "held for three years", "one year ago", "three years ago", "five years ago",
            "year ago", "one-year loss", "one-year losses", "1-year loss", "1-year losses",
            "adds to one-year losses", "decline adds to", "latest decline", "latest 7.3% decline",
            "investors may consider drastic measures", "institutional investors may consider drastic measures",
            "shareholders may consider drastic measures", "drastic measures", "share price decline",
            "share price has declined", "share price has fallen", "market value declined",
            "market cap declined", "stock is up", "stock is down", "return over",
        ],
        "reason": "Historical shareholder-return / price-performance commentary; not a fresh catalyst.",
    },
    {
        "type": "PRICE_MOVE_COMMENTARY",
        "sentiment": "NEUTRAL",
        "materiality": 0,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "MEDIUM",
        "keywords": [
            "shares fall", "shares fell", "shares slip", "stock falls", "stock slips",
            "share price falls", "stock rises", "shares rise", "shares gain", "stock jumps",
            "hits 52-week high", "hits 52-week low", "trades higher", "trades lower",
        ],
        "reason": "Generic price-move headline; price action is already captured by technicals.",
    },
    {
        "type": "GOVERNANCE_RISK",
        "sentiment": "NEGATIVE",
        "materiality": 3,
        "event_caution": 0,
        "risk": "HIGH",
        "confidence": "HIGH",
        "keywords": [
            "resignation", "resigns", "resigned", "steps down", "stepped down",
            "auditor resignation", "statutory auditor resigns", "cfo resigns", "ceo resigns",
            "company secretary resigns", "independent director resigns", "director resigns",
            "forensic audit", "qualified opinion", "adverse opinion", "governance concern",
        ],
        "reason": "Material management/auditor/governance risk.",
    },
    {
        "type": "REGULATORY_RISK",
        "sentiment": "NEGATIVE",
        "materiality": 3,
        "event_caution": 0,
        "risk": "HIGH",
        "confidence": "HIGH",
        "keywords": [
            "default", "debt default", "fraud", "raid", "ed raid", "income tax search",
            "penalty", "fine imposed", "show cause", "warning letter", "sebi order",
            "regulatory action", "nclt", "insolvency", "litigation", "suspension",
        ],
        "reason": "Regulatory/legal/credit risk event.",
    },
    {
        "type": "RESULTS_NEGATIVE",
        "sentiment": "NEGATIVE",
        "materiality": 3,
        "event_caution": 0,
        "risk": "MEDIUM",
        "confidence": "HIGH",
        "keywords": [
            "profit falls", "profit down", "net profit falls", "profit declines", "profit drops",
            "profit plunges", "loss widens", "revenue falls", "revenue declines", "sales fall",
            "misses estimates", "misses street", "margin contracts", "ebitda falls", "guidance cut",
        ],
        "reason": "Weak reported result or guidance cut.",
    },
    {
        "type": "RESULTS_POSITIVE",
        "sentiment": "POSITIVE",
        "materiality": 3,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "HIGH",
        "keywords": [
            "profit rises", "profit jumps", "profit surges", "profit up", "net profit rises",
            "net profit jumps", "pat rises", "pat jumps", "revenue up", "revenue rises",
            "revenue grows", "sales up", "beats estimates", "beats street", "strong results",
            "margin expands", "ebitda rises", "guidance raised",
        ],
        "reason": "Strong reported result or positive guidance.",
    },
    {
        "type": "UPCOMING_RESULTS",
        "sentiment": "NEUTRAL",
        "materiality": 2,
        "event_caution": -2,
        "risk": "MEDIUM",
        "confidence": "HIGH",
        "keywords": [
            "results today", "results tomorrow", "results next week", "to announce results",
            "will announce results", "quarterly results on", "earnings on", "earnings due",
            "result date", "results date", "board meeting to consider results", "board meeting to consider quarterly results",
            "board meeting for results", "board meeting to approve results",
            "meeting of the board to consider results", "financial results on",
        ],
        "reason": "Upcoming results/earnings can create gap risk; caution only.",
    },
    {
        "type": "CAPITAL_ACTION_CAUTION",
        "sentiment": "NEUTRAL",
        "materiality": 1,
        "event_caution": -1,
        "risk": "MEDIUM",
        "confidence": "MEDIUM",
        "keywords": [
            "fund raising", "fundraise", "preferential issue", "qip", "rights issue",
            "record date", "ex-date", "bonus issue", "stock split", "dividend record date",
            "buyback", "merger", "amalgamation", "demerger", "board meeting",
        ],
        "reason": "Scheduled corporate action / board agenda can create event volatility; caution only.",
    },
    {
        "type": "INVESTOR_COMMUNICATION",
        "sentiment": "NEUTRAL",
        "materiality": 0,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "HIGH",
        "keywords": [
            "analyst meet", "analyst/institutional investor meet", "institutional investor meet",
            "investor meet", "investor meeting", "conference call", "con. call", "concalls",
            "earnings call", "investor conference", "investor presentation", "analysts/investors meet",
        ],
        "reason": "Investor/analyst communication is informational unless paired with adverse content.",
    },
    {
        "type": "ORDER_WIN",
        "sentiment": "POSITIVE",
        "materiality": 2,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "MEDIUM",
        "keywords": [
            "order win", "wins order", "wins contract", "bagged order", "bags order", "secures order",
            "contract win", "letter of award", "large order", "awarded contract",
        ],
        "reason": "Order/contract win can be a positive catalyst.",
    },
    {
        "type": "BUSINESS_POSITIVE",
        "sentiment": "POSITIVE",
        "materiality": 2,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "MEDIUM",
        "keywords": [
            "capacity expansion", "strategic partnership", "new plant", "commissioned",
            "approval received", "acquisition", "merger approved", "launches new",
        ],
        "reason": "Positive business development.",
    },
    {
        "type": "BROKERAGE_POSITIVE",
        "sentiment": "POSITIVE",
        "materiality": 1,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "MEDIUM",
        "keywords": [
            "top pick", "top picks", "preferred pick", "buy call", "buy rating", "maintains buy",
            "reiterates buy", "initiates buy", "upgrade to buy", "overweight", "outperform",
            "target price raised", "raises target", "target price hike", "brokerage bullish",
        ],
        "reason": "Positive brokerage/research view.",
    },
    {
        "type": "BROKERAGE_NEGATIVE",
        "sentiment": "NEGATIVE",
        "materiality": 1,
        "event_caution": 0,
        "risk": "LOW",
        "confidence": "MEDIUM",
        "keywords": [
            "sell rating", "reduce rating", "underperform", "downgrade", "downgrades",
            "target price cut", "cuts target", "lower target", "bearish", "neutral rating",
        ],
        "reason": "Negative brokerage/research view.",
    },
]

SOURCE_WEIGHTS = {
    "NSE Corporate Announcements": 1.00,
    "BSE Corporate Announcements": 1.00,
    "Moneycontrol": 0.65,
    "Economic Times": 0.65,
    "The Economic Times": 0.65,
    "Business Standard": 0.65,
    "CNBCTV18": 0.60,
    "CNBCTV18.com": 0.60,
    "Reuters": 0.75,
    "Bloomberg": 0.75,
    "Mint": 0.60,
    "Google News": 0.45,
    "Simply Wall St": 0.10,
}

OVERRIDE_PATHS = [Path("news_overrides.csv"), Path("config/news_overrides.csv")]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    return s


def clean_text(x: object) -> str:
    if x is None:
        return ""
    txt = html.unescape(str(x))
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _parse_date(x: object) -> str:
    txt = clean_text(x)
    if not txt:
        return ""
    try:
        d = email.utils.parsedate_to_datetime(txt)
        return d.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return txt


def _has_any(text: str, keywords: list[str]) -> tuple[bool, str]:
    for k in keywords:
        if k in text:
            return True, k
    return False, ""


def _source_weight(source: str) -> float:
    src = clean_text(source)
    for key, w in SOURCE_WEIGHTS.items():
        if key.lower() in src.lower():
            return float(w)
    return 0.45 if src else 0.35


def _sentiment_base(sentiment: str) -> int:
    sentiment = str(sentiment or "NEUTRAL").upper()
    if sentiment == "POSITIVE":
        return 1
    if sentiment == "NEGATIVE":
        return -1
    return 0


def _score_from(sentiment: str, materiality: int, source_weight: float) -> int:
    base = _sentiment_base(sentiment)
    if base == 0 or int(materiality) <= 0:
        return 0
    # Source weight scales low-quality sources down. Materiality is capped to keep one headline from dominating.
    raw = base * min(int(materiality), 3) * float(source_weight)
    # For high-confidence official items, this still reaches +/-3. For generic Google items it is usually +/-1.
    if raw > 0:
        return int(max(1, round(raw)))
    return int(min(-1, round(raw)))


def load_overrides() -> pd.DataFrame:
    for p in OVERRIDE_PATHS:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return pd.DataFrame()
    return pd.DataFrame()


def _apply_override(symbol: str, text: str, source: str = "") -> dict | None:
    ov = load_overrides()
    if ov.empty:
        return None
    symbol = str(symbol or "").upper().strip()
    for _, row in ov.iterrows():
        sym_pat = str(row.get("symbol", "")).upper().strip()
        contains = str(row.get("contains", "")).lower().strip()
        src_pat = str(row.get("source_contains", "")).lower().strip()
        if sym_pat and sym_pat not in {"*", symbol}:
            continue
        if contains and contains not in text:
            continue
        if src_pat and src_pat not in str(source).lower():
            continue
        news_type = str(row.get("forced_type", "OVERRIDE")).strip() or "OVERRIDE"
        sentiment = str(row.get("forced_sentiment", "NEUTRAL")).upper().strip() or "NEUTRAL"
        try:
            score = int(float(row.get("forced_score", 0)))
        except Exception:
            score = 0
        try:
            caution = int(float(row.get("forced_event_caution", 0)))
        except Exception:
            caution = 0
        risk = str(row.get("forced_risk", "LOW")).upper().strip() or "LOW"
        reason = str(row.get("notes", "Manual override")).strip() or "Manual override"
        return {
            "item_type": news_type,
            "news_type": news_type,
            "sentiment": sentiment,
            "materiality": int(abs(score)) if score else 0,
            "source_weight": _source_weight(source),
            "confidence": "HIGH",
            "news_sentiment_score": score,
            "event_caution_score": caution,
            "event_score": score + caution,
            "event_risk": risk,
            "category": news_type.lower(),
            "tags": news_type.lower(),
            "matched_keyword": contains,
            "classification_reason": reason,
        }
    return None


def make_summary(headline: str, item_type: str = "NEWS", sentiment: str = "NEUTRAL", source: str = "") -> str:
    h = clean_text(headline)
    if not h:
        return ""
    if len(h) > 120:
        h = re.sub(r"\s+-\s+[^-]{2,80}$", "", h).strip()
    item_type = str(item_type or "NEWS").upper()
    sentiment = str(sentiment or "NEUTRAL").upper()
    if item_type in {"UPCOMING_RESULTS", "CAPITAL_ACTION_CAUTION"}:
        prefix = "Upcoming event caution: "
    elif item_type == "INVESTOR_COMMUNICATION":
        prefix = "Informational event: "
    elif item_type in {"LOW_VALUE_COMMENTARY", "PRICE_MOVE_COMMENTARY", "LOW_VALUE_NEWS"}:
        prefix = "Neutral / low-value news: "
    elif sentiment == "POSITIVE":
        prefix = "Positive catalyst: "
    elif sentiment == "NEGATIVE":
        prefix = "Negative catalyst: "
    else:
        prefix = "Neutral news: "
    return (prefix + h)[:360]


def _regex_result_classification(text: str) -> dict | None:
    pos = re.search(r"\b(q[1-4]|quarter)\b.*\b(profit|pat|revenue|sales|ebitda)\b.*\b(rise|rises|jump|jumps|surge|surges|grow|grows|beat|beats)\b", text)
    neg = re.search(r"\b(q[1-4]|quarter)\b.*\b(profit|pat|revenue|sales|ebitda)\b.*\b(fall|falls|decline|declines|drop|drops|miss|misses)\b", text)
    if pos:
        return {"type": "RESULTS_POSITIVE", "sentiment": "POSITIVE", "materiality": 3, "event_caution": 0, "risk": "LOW", "confidence": "MEDIUM", "matched": pos.group(0), "reason": "Pattern matched positive quarterly result."}
    if neg:
        return {"type": "RESULTS_NEGATIVE", "sentiment": "NEGATIVE", "materiality": 3, "event_caution": 0, "risk": "MEDIUM", "confidence": "MEDIUM", "matched": neg.group(0), "reason": "Pattern matched weak quarterly result."}
    return None


def classify_item(headline: str, symbol: str = "", source: str = "") -> dict:
    """Classify one headline into auditable fields.

    Returned fields are backward compatible with v8 plus new v9 audit fields:
    news_type, materiality, source_weight, confidence, matched_keyword,
    classification_reason.
    """
    text = clean_text(headline).lower()
    source_weight = _source_weight(source)

    override = _apply_override(symbol, text, source)
    if override:
        return override

    # Source-level override for low-value sources, but only when the headline is historical/price commentary.
    if "simply wall st" in str(source).lower():
        for group in TAXONOMY[:2]:
            matched, kw = _has_any(text, group["keywords"])
            if matched:
                news_type = group["type"]
                return {
                    "item_type": news_type,
                    "news_type": news_type,
                    "sentiment": "NEUTRAL",
                    "materiality": 0,
                    "source_weight": source_weight,
                    "confidence": "HIGH",
                    "news_sentiment_score": 0,
                    "event_caution_score": 0,
                    "event_score": 0,
                    "event_risk": "LOW",
                    "category": news_type.lower(),
                    "tags": news_type.lower(),
                    "matched_keyword": kw,
                    "classification_reason": group["reason"],
                }

    # Hard taxonomy-first classification. This order handles adverse events before generic positives.
    for group in TAXONOMY:
        matched, kw = _has_any(text, group["keywords"])
        if not matched:
            continue
        news_type = group["type"]
        sentiment = group["sentiment"]
        materiality = int(group["materiality"])
        caution = int(group["event_caution"])
        risk = group["risk"]
        confidence = group["confidence"]
        score = _score_from(sentiment, materiality, source_weight)
        # Material governance/regulatory risks should override source dilution.
        # A resignation/default/penalty should not become a mild -1 merely because it came via Google News.
        if news_type in {"GOVERNANCE_RISK", "REGULATORY_RISK"}:
            score = -3
        # Official NSE announcements should retain full high-materiality score.
        elif "nse corporate" in str(source).lower() and materiality >= 3 and sentiment != "NEUTRAL":
            score = 3 if sentiment == "POSITIVE" else -3
        return {
            "item_type": news_type,
            "news_type": news_type,
            "sentiment": sentiment,
            "materiality": materiality,
            "source_weight": round(source_weight, 2),
            "confidence": confidence,
            "news_sentiment_score": int(max(min(score, 4), -5)),
            "event_caution_score": int(max(min(caution, 0), -3)),
            "event_score": int(max(min(score + caution, 4), -5)),
            "event_risk": risk,
            "category": news_type.lower(),
            "tags": news_type.lower(),
            "matched_keyword": kw,
            "classification_reason": group["reason"],
        }

    regex_cls = _regex_result_classification(text)
    if regex_cls:
        score = _score_from(regex_cls["sentiment"], regex_cls["materiality"], source_weight)
        return {
            "item_type": regex_cls["type"],
            "news_type": regex_cls["type"],
            "sentiment": regex_cls["sentiment"],
            "materiality": regex_cls["materiality"],
            "source_weight": round(source_weight, 2),
            "confidence": regex_cls["confidence"],
            "news_sentiment_score": int(max(min(score, 4), -5)),
            "event_caution_score": 0,
            "event_score": int(max(min(score, 4), -5)),
            "event_risk": regex_cls["risk"],
            "category": regex_cls["type"].lower(),
            "tags": regex_cls["type"].lower(),
            "matched_keyword": regex_cls["matched"],
            "classification_reason": regex_cls["reason"],
        }

    return {
        "item_type": "GENERIC_NEWS",
        "news_type": "GENERIC_NEWS",
        "sentiment": "NEUTRAL",
        "materiality": 0,
        "source_weight": round(source_weight, 2),
        "confidence": "LOW",
        "news_sentiment_score": 0,
        "event_caution_score": 0,
        "event_score": 0,
        "event_risk": "LOW",
        "category": "generic_news",
        "tags": "generic_news",
        "matched_keyword": "",
        "classification_reason": "No material catalyst/event taxonomy match.",
    }


def _google_queries(symbol: str, company_name: str = "") -> list[str]:
    sym = str(symbol).upper().strip()
    name = clean_text(company_name)
    base = []
    if name:
        base += [
            f'"{name}" stock news India',
            f'"{name}" quarterly results earnings',
            f'"{name}" brokerage buy sell target price',
            f'"{name}" company announcement NSE',
        ]
    base += [
        f'"{sym}" NSE stock news',
        f'"{sym}" quarterly results',
        f'"{sym}" brokerage target price buy sell',
        f'"{sym}" board meeting results stock',
        f'"{sym}" order win penalty resignation',
    ]
    return base


def _row_from_classification(symbol: str, date_val: str, source: str, headline: str, url: str, cls: dict) -> dict:
    summary = make_summary(headline, cls["item_type"], cls["sentiment"], source)
    return {
        "symbol": symbol.upper(),
        "event_date": date_val,
        "source": source,
        "category": cls.get("category") or cls.get("tags"),
        "headline": headline,
        "summary": summary,
        "url": url,
        "event_score": cls["event_score"],
        "event_risk": cls["event_risk"],
        "item_type": cls["item_type"],
        "news_type": cls.get("news_type", cls["item_type"]),
        "news_sentiment_score": cls["news_sentiment_score"],
        "event_caution_score": cls["event_caution_score"],
        "sentiment": cls["sentiment"],
        "materiality": cls.get("materiality", 0),
        "source_weight": cls.get("source_weight", _source_weight(source)),
        "confidence": cls.get("confidence", "LOW"),
        "matched_keyword": cls.get("matched_keyword", ""),
        "classification_reason": cls.get("classification_reason", ""),
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def fetch_google_news_rss(symbol: str, company_name: str = "", limit: int = 12) -> pd.DataFrame:
    rows = []
    seen = set()
    for terms in _google_queries(symbol, company_name=company_name):
        url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(terms) + "&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            r = _session().get(url, timeout=15)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                title = clean_text(item.findtext("title"))
                link = clean_text(item.findtext("link"))
                pub = _parse_date(item.findtext("pubDate"))
                source_node = item.find("source")
                source = clean_text(source_node.text if source_node is not None else "Google News")
                if not title:
                    continue
                norm = re.sub(r"\W+", " ", title.lower()).strip()
                if norm in seen:
                    continue
                seen.add(norm)
                cls = classify_item(title, symbol=symbol, source=source)
                rows.append(_row_from_classification(symbol, pub, source, title, link, cls))
                if len(rows) >= limit:
                    return pd.DataFrame(rows)
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_nse_announcements(symbol: str, limit: int = 15) -> pd.DataFrame:
    s = _session()
    try:
        s.get("https://www.nseindia.com/companies-listing/corporate-filings-announcements", timeout=12)
    except Exception:
        pass
    sym = urllib.parse.quote(symbol.upper())
    urls = [
        f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}",
        f"https://www.nseindia.com/api/corporate-announcements?symbol={sym}",
        f"https://www.nseindia.com/api/corporate-announcements?index=equities&issuer={sym}",
    ]
    rows = []
    for url in urls:
        try:
            r = s.get(url, timeout=18)
            if r.status_code >= 400:
                continue
            data = r.json()
            if isinstance(data, dict):
                data = data.get("data") or data.get("rows") or data.get("announcements") or []
            if not isinstance(data, list):
                continue
            for item in data[:limit]:
                if not isinstance(item, dict):
                    continue
                headline = clean_text(
                    item.get("subject") or item.get("desc") or item.get("attchmntText") or
                    item.get("headline") or item.get("sm_name") or item.get("announcement") or
                    item.get("details") or item.get("purpose") or ""
                )
                if not headline:
                    continue
                date_val = _parse_date(item.get("an_dt") or item.get("date") or item.get("dissemDT") or item.get("created_at") or "")
                link = clean_text(item.get("attchmntFile") or item.get("url") or item.get("link") or "")
                if link and link.startswith("/"):
                    link = "https://www.nseindia.com" + link
                cls = classify_item(headline, symbol=symbol, source="NSE Corporate Announcements")
                rows.append(_row_from_classification(symbol, date_val, "NSE Corporate Announcements", headline, link, cls))
            if rows:
                break
        except Exception:
            continue
    return pd.DataFrame(rows)


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "symbol", "event_date", "source", "category", "headline", "summary", "url",
        "event_score", "event_risk", "item_type", "news_type", "news_sentiment_score",
        "event_caution_score", "sentiment", "materiality", "source_weight", "confidence",
        "matched_keyword", "classification_reason", "fetched_at",
    ])


def fetch_events_for_symbol(symbol: str, company_name: str = "", include_google: bool = True) -> pd.DataFrame:
    frames = [fetch_nse_announcements(symbol)]
    if include_google:
        frames.append(fetch_google_news_rss(symbol, company_name=company_name))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return _empty_events()
    df = pd.concat(frames, ignore_index=True)
    df["headline_norm"] = df["headline"].astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["symbol", "headline_norm"]).drop(columns=["headline_norm"])
    return df


def fetch_events_for_symbols(symbols: Iterable[str], limit_symbols: int = 100, sleep_s: float = 0.25,
                             include_google: bool = True, company_names: dict | None = None) -> pd.DataFrame:
    rows = []
    company_names = company_names or {}
    syms = list(dict.fromkeys([str(s).upper().strip() for s in symbols if str(s).strip()]))[:int(limit_symbols)]
    for sym in syms:
        df = fetch_events_for_symbol(sym, company_name=str(company_names.get(sym, "")), include_google=include_google)
        if not df.empty:
            rows.append(df)
        time.sleep(sleep_s)
    if not rows:
        return _empty_events()
    return pd.concat(rows, ignore_index=True)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "symbol", "news_score", "event_caution_score", "event_risk", "news_summary", "upcoming_event",
        "latest_event", "event_summary", "event_source", "event_url", "news_event_count", "upcoming_event_count",
        "top_news_type", "top_news_confidence", "top_news_materiality", "news_audit_reason",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols)
    ev = events.copy()
    for c in ["news_sentiment_score", "event_caution_score", "materiality", "source_weight"]:
        if c not in ev.columns:
            ev[c] = 0
        ev[c] = pd.to_numeric(ev[c], errors="coerce").fillna(0)
    for c in ["event_risk", "summary", "url", "source", "item_type", "news_type", "confidence", "classification_reason"]:
        if c not in ev.columns:
            ev[c] = ""
    rows = []
    for sym, g in ev.groupby("symbol"):
        g = g.copy()
        g["risk_rank"] = g["event_risk"].astype(str).str.upper().map(RISK_RANK).fillna(0)
        # Only material non-neutral catalysts contribute to news_score. Cap to reduce headline spam.
        news_items = g[(g["news_sentiment_score"] != 0) & (~g["item_type"].astype(str).str.upper().isin(["UPCOMING_RESULTS", "CAPITAL_ACTION_CAUTION"]))]
        event_items = g[g["event_caution_score"] < 0]
        info_items = g[g["news_sentiment_score"].eq(0) & g["event_caution_score"].eq(0)]

        news_score = int(max(min(news_items["news_sentiment_score"].sum(), 4), -5)) if not news_items.empty else 0
        caution_score = int(max(event_items["event_caution_score"].sum(), -3)) if not event_items.empty else 0
        max_risk = "LOW"
        if not g.empty:
            max_risk = g.sort_values("risk_rank", ascending=False).iloc[0].get("event_risk", "LOW") or "LOW"

        news_summary = ""
        latest_event = ""
        event_source = ""
        event_url = ""
        top_type = ""
        top_conf = ""
        top_mat = 0
        audit_reason = ""
        if not news_items.empty:
            # Prioritize absolute score, risk, materiality, and source weight.
            pick = news_items.assign(abs_score=news_items["news_sentiment_score"].abs()).sort_values(
                ["abs_score", "risk_rank", "materiality", "source_weight"], ascending=[False, False, False, False]
            ).iloc[0]
            news_summary = pick.get("summary", "")
            latest_event = pick.get("headline", "")
            event_source = pick.get("source", "")
            event_url = pick.get("url", "")
            top_type = pick.get("news_type", pick.get("item_type", ""))
            top_conf = pick.get("confidence", "")
            top_mat = int(pick.get("materiality", 0) or 0)
            audit_reason = pick.get("classification_reason", "")
        elif not info_items.empty:
            pick = info_items.sort_values(["materiality", "source_weight"], ascending=[False, False]).iloc[0]
            latest_event = pick.get("headline", "")
            event_source = pick.get("source", "")
            event_url = pick.get("url", "")
            top_type = pick.get("news_type", pick.get("item_type", ""))
            top_conf = pick.get("confidence", "")
            top_mat = int(pick.get("materiality", 0) or 0)
            audit_reason = pick.get("classification_reason", "")

        upcoming_event = ""
        if not event_items.empty:
            pick_up = event_items.assign(abs_caution=event_items["event_caution_score"].abs()).sort_values(
                ["abs_caution", "risk_rank", "materiality", "source_weight"], ascending=[False, False, False, False]
            ).iloc[0]
            upcoming_event = pick_up.get("summary", "")
            if not latest_event:
                latest_event = pick_up.get("headline", "")
                event_source = pick_up.get("source", "")
                event_url = pick_up.get("url", "")
                top_type = pick_up.get("news_type", pick_up.get("item_type", ""))
                top_conf = pick_up.get("confidence", "")
                top_mat = int(pick_up.get("materiality", 0) or 0)
                audit_reason = pick_up.get("classification_reason", "")

        event_summary = news_summary or upcoming_event or (make_summary(latest_event, top_type, "NEUTRAL") if latest_event else "")
        rows.append({
            "symbol": sym,
            "news_score": news_score,
            "event_caution_score": caution_score,
            "event_risk": max_risk,
            "news_summary": news_summary,
            "upcoming_event": upcoming_event,
            "latest_event": latest_event,
            "event_summary": event_summary,
            "event_source": event_source,
            "event_url": event_url,
            "news_event_count": int(len(news_items)),
            "upcoming_event_count": int(len(event_items)),
            "top_news_type": top_type,
            "top_news_confidence": top_conf,
            "top_news_materiality": top_mat,
            "news_audit_reason": audit_reason,
        })
    return pd.DataFrame(rows)[cols]


def apply_news_to_recommendations(recs: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    recs = recs.copy()
    summary = summarize_events(events)
    defaults = [
        ("news_score", 0), ("event_caution_score", 0), ("event_risk", "LOW"),
        ("latest_event", ""), ("event_summary", ""), ("news_summary", ""), ("upcoming_event", ""),
        ("event_source", ""), ("event_url", ""), ("news_event_count", 0), ("upcoming_event_count", 0),
        ("top_news_type", ""), ("top_news_confidence", ""), ("top_news_materiality", 0), ("news_audit_reason", ""),
    ]
    for col, default in defaults:
        if col not in recs.columns:
            recs[col] = default
    if not summary.empty:
        recs = recs.drop(columns=[c for c, _ in defaults if c in recs.columns], errors="ignore")
        recs = recs.merge(summary, on="symbol", how="left")
        for c in ["news_score", "event_caution_score", "news_event_count", "upcoming_event_count", "top_news_materiality"]:
            recs[c] = pd.to_numeric(recs[c], errors="coerce").fillna(0).astype(int)
        for c, default in [
            ("event_risk", "LOW"), ("latest_event", ""), ("event_summary", ""), ("news_summary", ""),
            ("upcoming_event", ""), ("event_source", ""), ("event_url", ""), ("top_news_type", ""),
            ("top_news_confidence", ""), ("news_audit_reason", ""),
        ]:
            recs[c] = recs[c].fillna(default)
    recs["news_adjusted_score"] = recs["score"].fillna(0).astype(float) + recs["news_score"].fillna(0).astype(float)
    recs["event_adjusted_score"] = recs["news_adjusted_score"] + recs["event_caution_score"].fillna(0).astype(float)
    recs["final_score"] = recs["event_adjusted_score"]

    def adjusted_action(r):
        if r.get("event_risk") == "HIGH" and str(r.get("action")) == "BUY":
            return "WATCH_EVENT_RISK"
        if int(r.get("event_caution_score", 0)) <= -2 and str(r.get("action")) == "BUY":
            return "WATCH_EVENT_CAUTION"
        fs = float(r.get("event_adjusted_score", r.get("score", 0)) or 0)
        if fs >= 7 and r.get("event_risk") != "HIGH":
            return "BUY"
        if fs >= 4:
            return "WATCH"
        if fs <= -3:
            return "SELL/AVOID"
        return "NEUTRAL"

    recs["news_adjusted_action"] = recs.apply(adjusted_action, axis=1)
    recs["news_reason"] = recs.apply(
        lambda r: (
            f"News {int(r.get('news_score',0))}; event caution {int(r.get('event_caution_score',0))}; "
            f"risk {r.get('event_risk','LOW')}; type {r.get('top_news_type','')}; "
            f"confidence {r.get('top_news_confidence','')}; reason {r.get('news_audit_reason','')}. "
            f"{r.get('news_summary','') or r.get('upcoming_event','')}"
        ), axis=1
    )
    return recs
