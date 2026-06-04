import datetime as dt
import time
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .indicators import add_indicators, score_latest
from .news_engine import fetch_events_for_symbols, apply_news_to_recommendations
from .scoring_engine import apply_final_recommendation
from .analytics_engine import apply_advanced_analytics
from .learning_engine import apply_ai_probability, record_recommendations, update_outcomes
from .macro_engine import compute_macro_snapshot, apply_macro_to_recommendations
from .storage import upsert_prices, load_prices, replace_recommendations, init_db, replace_news_events, replace_macro_snapshot, load_macro_snapshot


def _find_col(cols, candidates):
    norm = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r'[^a-z0-9]', '', cand.lower())
        if key in norm:
            return norm[key]
    for c in cols:
        lc = c.lower()
        if any(cand.lower() in lc for cand in candidates):
            return c
    return None


def normalize_bhavcopy(df: pd.DataFrame, date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cmap = {
        'symbol': _find_col(df.columns, ['symbol', 'SYMBOL']),
        'series': _find_col(df.columns, ['series', 'SERIES']),
        'open': _find_col(df.columns, ['open', 'OPEN_PRICE', 'OPEN']),
        'high': _find_col(df.columns, ['high', 'HIGH_PRICE', 'HIGH']),
        'low': _find_col(df.columns, ['low', 'LOW_PRICE', 'LOW']),
        'close': _find_col(df.columns, ['close', 'CLOSE_PRICE', 'CLOSE']),
        'prev_close': _find_col(df.columns, ['prevclose', 'PREV_CLOSE', 'prev_close']),
        'volume': _find_col(df.columns, ['volume', 'tottrdqty', 'TOTTRDQTY', 'TTL_TRD_QNTY', 'totaltradedquantity']),
        'value': _find_col(df.columns, ['tottrdval', 'TOTTRDVAL', 'TURNOVER_LACS', 'value']),
        'trades': _find_col(df.columns, ['totaltrades', 'TOTALTRADES', 'trades']),
        'isin': _find_col(df.columns, ['isin', 'ISIN']),
    }
    required = ['symbol', 'open', 'high', 'low', 'close']
    missing = [k for k in required if cmap.get(k) is None]
    if missing:
        raise ValueError(f'Bhavcopy columns missing: {missing}. Available columns: {list(df.columns)}')

    out = pd.DataFrame()
    for k, src in cmap.items():
        if src:
            out[k] = df[src]
        elif k == 'series':
            # Some nsefin Bhavcopy outputs do not include a series column.
            # Treat such rows as EQ instead of filtering everything out.
            out[k] = 'EQ'
        else:
            out[k] = None
    out['symbol'] = out['symbol'].astype(str).str.upper().str.strip()
    out['series'] = out['series'].astype(str).str.upper().str.strip()
    out['date'] = pd.to_datetime(date)
    for c in ['open','high','low','close','prev_close','volume','value','trades']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['symbol','open','high','low','close'])
    out = out[out['symbol'].str.len() > 0]
    # Keep equity-like series by default when series exists.
    if 'series' in out.columns:
        out = out[out['series'].isin(['EQ', 'BE', 'BZ', 'SM', 'ST', 'SZ'])]
    return out


def read_symbols(path: Optional[str]) -> list[str]:
    if not path or not Path(path).exists():
        return []
    return [x.strip().upper() for x in Path(path).read_text().splitlines() if x.strip() and not x.strip().startswith('#')]


def fetch_bhavcopies(days: int = 120, symbols: Optional[Iterable[str]] = None, mode: str = 'symbols', sleep_s: float = 0.15) -> dict:
    from .nsefin_client import get_equity_bhav_copy_for_date
    init_db()
    symbols_set = set([s.upper() for s in symbols]) if symbols else set()
    current_date = dt.datetime.now()
    days_collected = 0
    attempts = 0
    total_rows = 0
    latest_df = pd.DataFrame()
    logs = []

    # Wider attempt window handles weekends, NSE holidays, and occasional nsefin failures.
    while days_collected < days and attempts < int(days * 2.8) + 35:
        target_date = current_date - dt.timedelta(days=attempts)
        attempts += 1
        try:
            raw = get_equity_bhav_copy_for_date(target_date)  # confirmed user syntax
            if raw is None or raw.empty:
                continue
            df = normalize_bhavcopy(raw, target_date)
            if df.empty:
                continue
            latest_df = df if latest_df.empty or target_date > latest_df['date'].max() else latest_df
            if mode == 'symbols' and symbols_set:
                df = df[df['symbol'].isin(symbols_set)].copy()
            elif mode == 'top500':
                # top 500 by volume within each bhavcopy; this keeps storage manageable.
                df = df.sort_values('volume', ascending=False).head(500).copy()
            # mode all stores full bhavcopy.
            if not df.empty:
                rows = upsert_prices(df)
                total_rows += rows
                days_collected += 1
                logs.append(f'Collected {target_date:%Y-%m-%d}: {rows} rows')
        except Exception as e:
            logs.append(f'Skipped {target_date:%Y-%m-%d}: {type(e).__name__}: {str(e)[:180]}')
        time.sleep(sleep_s)
    return {'days_collected': days_collected, 'rows': total_rows, 'logs': logs[-25:]}


def generate_recommendations(symbols: Optional[Iterable[str]] = None, min_history: int = 50, with_news: bool = False, news_limit_symbols: int = 100, with_macro: bool = False) -> pd.DataFrame:
    prices = load_prices(symbols=list(symbols) if symbols else None)
    if prices.empty:
        replace_recommendations(pd.DataFrame())
        return pd.DataFrame()
    counts = prices.groupby('symbol').size()
    good_symbols = counts[counts >= min_history].index.tolist()
    prices = prices[prices['symbol'].isin(good_symbols)].copy()
    if prices.empty:
        replace_recommendations(pd.DataFrame())
        return pd.DataFrame()
    with_ind = add_indicators(prices)
    latest = with_ind.sort_values('date').groupby('symbol', as_index=False).tail(1).copy()
    rows = []
    for _, row in latest.iterrows():
        s = score_latest(row)
        rows.append({
            'symbol': row['symbol'],
            'date': pd.to_datetime(row['date']).strftime('%Y-%m-%d'),
            'close': round(float(row['close']), 2),
            'volume': None if pd.isna(row.get('volume')) else float(row.get('volume')),
            'score': s['score'],
            'action': s['action'],
            'rsi14': None if pd.isna(row.get('rsi14')) else round(float(row.get('rsi14')), 2),
            'macd_hist': None if pd.isna(row.get('macd_hist')) else round(float(row.get('macd_hist')), 4),
            'adx14': None if pd.isna(row.get('adx14')) else round(float(row.get('adx14')), 2),
            'sma20': None if pd.isna(row.get('sma20')) else round(float(row.get('sma20')), 2),
            'sma50': None if pd.isna(row.get('sma50')) else round(float(row.get('sma50')), 2),
            'sma200': None if pd.isna(row.get('sma200')) else round(float(row.get('sma200')), 2),
            'atr14': None if pd.isna(row.get('atr14')) else round(float(row.get('atr14')), 2),
            'bb_pos': None if pd.isna(row.get('bb_pos')) else round(float(row.get('bb_pos')), 3),
            'stoch14': None if pd.isna(row.get('stoch14')) else round(float(row.get('stoch14')), 2),
            'vol_ratio': None if pd.isna(row.get('vol_ratio')) else round(float(row.get('vol_ratio')), 2),
            'stop_loss': s['stop_loss'],
            'target': s['target'],
            'rr': s['rr'],
            'reason': s['reason'],
        })
    recs = pd.DataFrame(rows).sort_values(['score', 'volume'], ascending=[False, False])
    # Advanced analytics: sector breadth/relative strength, liquidity, volatility, setup grade.
    recs = apply_advanced_analytics(recs, with_ind)
    # Default news columns are present even when news enrichment is off.
    recs['news_score'] = 0
    recs['event_caution_score'] = 0
    recs['event_risk'] = 'LOW'
    recs['latest_event'] = ''
    recs['event_source'] = ''
    recs['event_url'] = ''
    recs['event_summary'] = ''
    recs['news_summary'] = ''
    recs['upcoming_event'] = ''
    recs['news_event_count'] = 0
    recs['upcoming_event_count'] = 0
    recs['top_news_type'] = ''
    recs['top_news_confidence'] = ''
    recs['top_news_materiality'] = 0
    recs['news_audit_reason'] = ''
    recs['news_adjusted_score'] = recs['score']
    recs['event_adjusted_score'] = recs['score']
    recs['final_score'] = recs['score']
    recs['news_adjusted_action'] = recs['action']
    recs['news_reason'] = ''
    recs['macro_score'] = 0
    recs['macro_risk'] = 'UNKNOWN'
    recs['macro_regime'] = 'NO_DATA'
    recs['macro_reason'] = ''
    if 'sector_macro_adj' not in recs.columns:
        recs['sector_macro_adj'] = 0
    recs['macro_adjusted_score'] = recs['final_score']
    recs['macro_adjusted_action'] = recs['news_adjusted_action']
    recs = apply_final_recommendation(recs)

    if with_news and not recs.empty:
        # Enrich only top symbols by technical score to keep free sources respectful and fast.
        top_symbols = recs.head(int(news_limit_symbols))['symbol'].tolist()
        events = fetch_events_for_symbols(top_symbols, limit_symbols=int(news_limit_symbols))
        replace_news_events(events)
        recs = apply_news_to_recommendations(recs, events)
        recs = apply_final_recommendation(recs)
        recs = recs.sort_values(['combined_signal_score', 'final_score', 'score', 'volume'], ascending=[False, False, False, False])

    if with_macro and not recs.empty:
        macro = compute_macro_snapshot()
        replace_macro_snapshot(macro)
        recs = apply_macro_to_recommendations(recs, macro)
        recs = apply_final_recommendation(recs)
        recs = recs.sort_values(['combined_signal_score', 'macro_adjusted_score', 'final_score', 'score', 'volume'], ascending=[False, False, False, False, False])

    # Learning overlay: evaluate old recommendation outcomes, apply trained AI probability if available,
    # then persist the current recommendation snapshot for future learning.
    try:
        update_outcomes()
        recs = apply_ai_probability(recs)
    except Exception as e:
        recs['ai_success_probability'] = None
        recs['ai_confidence'] = 'ERROR'
        recs['ai_adjusted_score'] = recs.get('combined_signal_score', recs.get('score', 0))
        recs['ai_recommendation'] = recs.get('final_recommendation', recs.get('action', ''))
        recs['ai_reason'] = f'Learning overlay skipped: {e}'

    replace_recommendations(recs)
    try:
        record_recommendations(recs)
    except Exception:
        pass
    Path('exports').mkdir(exist_ok=True)
    recs.to_csv('exports/recommendations.csv', index=False)
    # Also write root-level CSV because the user's working Streamlit sample reads recommendations.csv.
    recs.rename(columns={
        'symbol':'Stock', 'action':'Action', 'close':'Price', 'rsi14':'RSI',
        'stop_loss':'SL', 'target':'TP', 'score':'Probability',
        'news_score':'News_Sentiment_Score', 'event_caution_score':'Event_Caution_Score', 'event_risk':'Event_Risk',
        'latest_event':'Latest_Event', 'news_adjusted_action':'News_Adjusted_Action',
        'macro_score':'Macro_Score', 'macro_risk':'Macro_Risk', 'macro_regime':'Macro_Regime',
        'macro_adjusted_score':'Macro_Adjusted_Score', 'macro_adjusted_action':'Macro_Adjusted_Action',
        'event_summary':'Event_Summary', 'news_summary':'News_Summary', 'upcoming_event':'Upcoming_Event', 'news_event_count':'News_Event_Count', 'upcoming_event_count':'Upcoming_Event_Count',
        'news_adjusted_score':'News_Adjusted_Score', 'event_adjusted_score':'Event_Adjusted_Score', 'combined_signal_score':'Combined_Signal_Score', 'final_recommendation':'Final_Recommendation',
        'final_signal_reason':'Final_Signal_Reason',
        'sector':'Sector', 'sector_score':'Sector_Score', 'trend_score':'Trend_Score', 'momentum_score':'Momentum_Score',
        'volume_score':'Volume_Score', 'liquidity_score':'Liquidity_Score', 'volatility_score':'Volatility_Score',
        'risk_quality_score':'Risk_Quality_Score', 'analytics_score':'Analytics_Score', 'setup_grade':'Setup_Grade',
        'suggested_risk_pct':'Suggested_Risk_Pct', 'analytics_reason':'Analytics_Reason',
        'top_news_type':'Top_News_Type', 'top_news_confidence':'Top_News_Confidence',
        'top_news_materiality':'Top_News_Materiality', 'news_audit_reason':'News_Audit_Reason',
        'ai_success_probability':'AI_Success_Probability', 'ai_confidence':'AI_Confidence',
        'ai_adjusted_score':'AI_Adjusted_Score', 'ai_recommendation':'AI_Recommendation', 'ai_reason':'AI_Reason'
    }).to_csv('recommendations.csv', index=False)
    return recs


def refresh_news_only(limit_symbols: int = 100) -> pd.DataFrame:
    """Fetch news/results for existing recommendations and update adjusted scores."""
    from .storage import load_recommendations
    recs = load_recommendations()
    if recs.empty:
        return pd.DataFrame()
    top_symbols = recs.sort_values('score', ascending=False).head(int(limit_symbols))['symbol'].tolist()
    events = fetch_events_for_symbols(top_symbols, limit_symbols=int(limit_symbols))
    replace_news_events(events)
    enriched = apply_news_to_recommendations(recs, events)
    enriched = apply_final_recommendation(enriched)
    enriched = enriched.sort_values(['combined_signal_score', 'final_score', 'score', 'volume'], ascending=[False, False, False, False])
    try:
        update_outcomes()
        enriched = apply_ai_probability(enriched)
    except Exception:
        pass
    replace_recommendations(enriched)
    Path('exports').mkdir(exist_ok=True)
    enriched.to_csv('exports/recommendations.csv', index=False)
    enriched.rename(columns={
        'symbol':'Stock', 'action':'Action', 'close':'Price', 'rsi14':'RSI',
        'stop_loss':'SL', 'target':'TP', 'score':'Probability',
        'news_score':'News_Sentiment_Score', 'event_caution_score':'Event_Caution_Score', 'event_risk':'Event_Risk',
        'latest_event':'Latest_Event', 'news_adjusted_action':'News_Adjusted_Action',
        'macro_score':'Macro_Score', 'macro_risk':'Macro_Risk', 'macro_regime':'Macro_Regime',
        'macro_adjusted_score':'Macro_Adjusted_Score', 'macro_adjusted_action':'Macro_Adjusted_Action',
        'event_summary':'Event_Summary', 'news_summary':'News_Summary', 'upcoming_event':'Upcoming_Event', 'news_event_count':'News_Event_Count', 'upcoming_event_count':'Upcoming_Event_Count',
        'news_adjusted_score':'News_Adjusted_Score', 'event_adjusted_score':'Event_Adjusted_Score', 'combined_signal_score':'Combined_Signal_Score', 'final_recommendation':'Final_Recommendation',
        'final_signal_reason':'Final_Signal_Reason',
        'sector':'Sector', 'sector_score':'Sector_Score', 'trend_score':'Trend_Score', 'momentum_score':'Momentum_Score',
        'volume_score':'Volume_Score', 'liquidity_score':'Liquidity_Score', 'volatility_score':'Volatility_Score',
        'risk_quality_score':'Risk_Quality_Score', 'analytics_score':'Analytics_Score', 'setup_grade':'Setup_Grade',
        'suggested_risk_pct':'Suggested_Risk_Pct', 'analytics_reason':'Analytics_Reason',
        'top_news_type':'Top_News_Type', 'top_news_confidence':'Top_News_Confidence',
        'top_news_materiality':'Top_News_Materiality', 'news_audit_reason':'News_Audit_Reason',
        'ai_success_probability':'AI_Success_Probability', 'ai_confidence':'AI_Confidence',
        'ai_adjusted_score':'AI_Adjusted_Score', 'ai_recommendation':'AI_Recommendation', 'ai_reason':'AI_Reason'
    }).to_csv('recommendations.csv', index=False)
    return enriched


def refresh_macro_only() -> pd.DataFrame:
    """Refresh macro regime and update existing recommendations."""
    from .storage import load_recommendations
    recs = load_recommendations()
    macro = compute_macro_snapshot()
    replace_macro_snapshot(macro)
    if recs.empty:
        return pd.DataFrame()
    enriched = apply_macro_to_recommendations(recs, macro)
    enriched = apply_final_recommendation(enriched)
    enriched = enriched.sort_values(['combined_signal_score', 'macro_adjusted_score', 'final_score', 'score', 'volume'], ascending=[False, False, False, False, False])
    try:
        update_outcomes()
        enriched = apply_ai_probability(enriched)
    except Exception:
        pass
    replace_recommendations(enriched)
    Path('exports').mkdir(exist_ok=True)
    enriched.to_csv('exports/recommendations.csv', index=False)
    enriched.rename(columns={
        'symbol':'Stock', 'action':'Action', 'close':'Price', 'rsi14':'RSI',
        'stop_loss':'SL', 'target':'TP', 'score':'Probability',
        'news_score':'News_Sentiment_Score', 'event_caution_score':'Event_Caution_Score', 'event_risk':'Event_Risk',
        'latest_event':'Latest_Event', 'news_adjusted_action':'News_Adjusted_Action',
        'macro_score':'Macro_Score', 'macro_risk':'Macro_Risk', 'macro_regime':'Macro_Regime',
        'macro_adjusted_score':'Macro_Adjusted_Score', 'macro_adjusted_action':'Macro_Adjusted_Action',
        'event_summary':'Event_Summary', 'news_summary':'News_Summary', 'upcoming_event':'Upcoming_Event', 'news_event_count':'News_Event_Count', 'upcoming_event_count':'Upcoming_Event_Count',
        'news_adjusted_score':'News_Adjusted_Score', 'event_adjusted_score':'Event_Adjusted_Score', 'combined_signal_score':'Combined_Signal_Score', 'final_recommendation':'Final_Recommendation',
        'final_signal_reason':'Final_Signal_Reason',
        'sector':'Sector', 'sector_score':'Sector_Score', 'trend_score':'Trend_Score', 'momentum_score':'Momentum_Score',
        'volume_score':'Volume_Score', 'liquidity_score':'Liquidity_Score', 'volatility_score':'Volatility_Score',
        'risk_quality_score':'Risk_Quality_Score', 'analytics_score':'Analytics_Score', 'setup_grade':'Setup_Grade',
        'suggested_risk_pct':'Suggested_Risk_Pct', 'analytics_reason':'Analytics_Reason',
        'top_news_type':'Top_News_Type', 'top_news_confidence':'Top_News_Confidence',
        'top_news_materiality':'Top_News_Materiality', 'news_audit_reason':'News_Audit_Reason',
        'ai_success_probability':'AI_Success_Probability', 'ai_confidence':'AI_Confidence',
        'ai_adjusted_score':'AI_Adjusted_Score', 'ai_recommendation':'AI_Recommendation', 'ai_reason':'AI_Reason'
    }).to_csv('recommendations.csv', index=False)
    return enriched


def refresh(days=120, symbols_path='symbols.txt', mode='top500', with_news: bool = False, news_limit_symbols: int = 100, with_macro: bool = False) -> tuple[dict, pd.DataFrame]:
    symbols = read_symbols(symbols_path) if mode == 'symbols' else []
    fetch_info = fetch_bhavcopies(days=days, symbols=symbols, mode=mode)
    rec_symbols = symbols if mode == 'symbols' and symbols else None
    recs = generate_recommendations(symbols=rec_symbols, with_news=with_news, news_limit_symbols=news_limit_symbols, with_macro=with_macro)
    return fetch_info, recs
