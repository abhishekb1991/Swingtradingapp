"""Learning engine for SwingNSE.

This module adds a controlled feedback loop:
1) store every recommendation snapshot,
2) evaluate later 5/10/20 trading-day outcomes from stored OHLCV,
3) train a simple probability model only on completed outcomes,
4) apply AI probability as an overlay, not as an uncontrolled auto-trader.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .storage import connect, init_db, load_prices

MODEL_DIR = Path('models')
MODEL_PATH = MODEL_DIR / 'swing_success_model.pkl'
FEATURE_COLUMNS = [
    'technical_score', 'score', 'analytics_score', 'trend_score', 'momentum_score',
    'volume_score', 'liquidity_score', 'volatility_score', 'risk_quality_score',
    'sector_score', 'news_score', 'event_caution_score', 'macro_score',
    'sector_macro_adj', 'combined_signal_score', 'rsi14', 'macd_hist', 'adx14',
    'bb_pos', 'stoch14', 'vol_ratio', 'rr', 'atr_pct', 'avg_traded_value_20_cr',
    'ret_20d', 'ret_60d'
]


def ensure_learning_tables():
    init_db()
    with connect() as con:
        con.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            close REAL,
            stop_loss REAL,
            target REAL,
            action TEXT,
            final_recommendation TEXT,
            combined_signal_score REAL,
            technical_score REAL,
            analytics_score REAL,
            news_score REAL,
            event_caution_score REAL,
            macro_score REAL,
            sector_score REAL,
            ai_success_probability REAL,
            ai_recommendation TEXT,
            features_json TEXT,
            reason TEXT,
            outcome_5d_return REAL,
            outcome_10d_return REAL,
            outcome_20d_return REAL,
            max_favourable_20d REAL,
            max_adverse_20d REAL,
            target_hit_20d INTEGER,
            stop_hit_20d INTEGER,
            outcome_label INTEGER,
            mistake_type TEXT,
            evaluated_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(snapshot_date, symbol)
        )''')
        con.execute('''
        CREATE TABLE IF NOT EXISTS learning_model_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at TEXT DEFAULT CURRENT_TIMESTAMP,
            rows_used INTEGER,
            feature_count INTEGER,
            validation_rows INTEGER,
            validation_accuracy REAL,
            validation_auc REAL,
            model_path TEXT,
            notes TEXT
        )''')
        con.commit()


def _num(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _feature_payload(row: pd.Series) -> dict:
    payload = {}
    for c in FEATURE_COLUMNS:
        payload[c] = _num(row.get(c), 0.0)
    # Add compact categorical context without forcing it into the ML model yet.
    for c in ['sector', 'setup_grade', 'event_risk', 'macro_risk', 'macro_regime']:
        if c in row.index:
            payload[c] = '' if pd.isna(row.get(c)) else str(row.get(c))
    return payload


def record_recommendations(recs: pd.DataFrame) -> int:
    """Persist one immutable recommendation snapshot per symbol/date."""
    ensure_learning_tables()
    if recs is None or recs.empty:
        return 0
    rows_inserted = 0
    with connect() as con:
        for _, r in recs.iterrows():
            snapshot_date = str(r.get('date') or pd.Timestamp.today().date())[:10]
            symbol = str(r.get('symbol', '')).upper().strip()
            if not symbol:
                continue
            features = _feature_payload(r)
            con.execute('''
            INSERT OR IGNORE INTO recommendation_history(
                snapshot_date, symbol, close, stop_loss, target, action, final_recommendation,
                combined_signal_score, technical_score, analytics_score, news_score,
                event_caution_score, macro_score, sector_score, ai_success_probability,
                ai_recommendation, features_json, reason
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                snapshot_date, symbol, _num(r.get('close')), _num(r.get('stop_loss'), None), _num(r.get('target'), None),
                str(r.get('action','')), str(r.get('final_recommendation', r.get('action',''))),
                _num(r.get('combined_signal_score', r.get('final_score', r.get('score', 0)))),
                _num(r.get('technical_score', r.get('score', 0))), _num(r.get('analytics_score', 0)),
                _num(r.get('news_score', 0)), _num(r.get('event_caution_score', 0)),
                _num(r.get('macro_score', 0)), _num(r.get('sector_score', 0)),
                _num(r.get('ai_success_probability'), None), str(r.get('ai_recommendation','')),
                json.dumps(features, ensure_ascii=False), str(r.get('final_signal_reason', r.get('reason','')))
            ))
            rows_inserted += con.total_changes
        con.commit()
    # con.total_changes is cumulative; return approximate unique count by requerying ignored rows is overkill.
    return int(len(recs))


def _classify_mistake(final_rec: str, ret20: float | None, target_hit: int, stop_hit: int, features: dict) -> tuple[int | None, str]:
    if ret20 is None or pd.isna(ret20):
        return None, ''
    rec = (final_rec or '').upper()
    is_long = ('BUY' in rec) or ('WATCH' in rec)
    is_avoid = ('AVOID' in rec) or ('SELL' in rec)
    if is_long:
        if stop_hit:
            return 0, 'Stop-loss hit before 20D outcome'
        if target_hit or ret20 >= 3.0:
            return 1, ''
        if ret20 < 0:
            if _num(features.get('sector_score')) < 0:
                return 0, 'False positive: weak sector context'
            if _num(features.get('volatility_score')) < 0:
                return 0, 'False positive: high volatility / poor risk quality'
            if _num(features.get('news_score')) < 0 or _num(features.get('event_caution_score')) < 0:
                return 0, 'False positive: news/event risk dominated setup'
            return 0, 'False positive: negative 20D return'
        return 0, 'Weak long: did not reach minimum 20D return threshold'
    if is_avoid:
        if ret20 <= 0:
            return 1, ''
        if ret20 >= 5:
            return 0, 'Missed winner: avoid/sell but stock rallied'
        return 1, ''
    # Neutral is not used for supervised success/failure by default.
    return None, 'Neutral recommendation excluded from supervised label'


def update_outcomes(horizon_days: int = 20) -> int:
    """Evaluate history rows once future daily candles are available."""
    ensure_learning_tables()
    prices = load_prices()
    if prices.empty:
        return 0
    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['date'] = pd.to_datetime(prices['date'])
    updated = 0
    with connect() as con:
        hist = pd.read_sql_query('SELECT * FROM recommendation_history WHERE outcome_20d_return IS NULL', con)
        if hist.empty:
            return 0
        for _, h in hist.iterrows():
            sym = h['symbol']
            snap_date = pd.to_datetime(h['snapshot_date'])
            sdata = prices[(prices['symbol'] == sym) & (prices['date'] > snap_date)].sort_values('date').head(horizon_days)
            if len(sdata) < horizon_days:
                continue
            entry = _num(h.get('close'))
            if not entry:
                continue
            closes = sdata['close'].astype(float).reset_index(drop=True)
            highs = sdata['high'].astype(float).reset_index(drop=True)
            lows = sdata['low'].astype(float).reset_index(drop=True)
            ret5 = (closes.iloc[min(4, len(closes)-1)] / entry - 1) * 100 if len(closes) >= 5 else None
            ret10 = (closes.iloc[min(9, len(closes)-1)] / entry - 1) * 100 if len(closes) >= 10 else None
            ret20 = (closes.iloc[horizon_days-1] / entry - 1) * 100
            mfe = (highs.max() / entry - 1) * 100
            mae = (lows.min() / entry - 1) * 100
            target = _num(h.get('target'), None)
            stop = _num(h.get('stop_loss'), None)
            target_hit = int(target is not None and target > 0 and (highs >= target).any())
            stop_hit = int(stop is not None and stop > 0 and (lows <= stop).any())
            try:
                features = json.loads(h.get('features_json') or '{}')
            except Exception:
                features = {}
            label, mistake = _classify_mistake(str(h.get('final_recommendation','')), ret20, target_hit, stop_hit, features)
            con.execute('''
            UPDATE recommendation_history
            SET outcome_5d_return=?, outcome_10d_return=?, outcome_20d_return=?,
                max_favourable_20d=?, max_adverse_20d=?, target_hit_20d=?, stop_hit_20d=?,
                outcome_label=?, mistake_type=?, evaluated_at=CURRENT_TIMESTAMP
            WHERE id=?
            ''', (ret5, ret10, ret20, mfe, mae, target_hit, stop_hit, label, mistake, int(h['id'])))
            updated += 1
        con.commit()
    return updated


def load_learning_history() -> pd.DataFrame:
    ensure_learning_tables()
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM recommendation_history ORDER BY snapshot_date DESC, combined_signal_score DESC', con)


def _history_to_training_frame(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if hist is None or hist.empty:
        return pd.DataFrame()
    for _, h in hist.dropna(subset=['outcome_label']).iterrows():
        try:
            f = json.loads(h.get('features_json') or '{}')
        except Exception:
            f = {}
        row = {c: _num(f.get(c), 0.0) for c in FEATURE_COLUMNS}
        row['outcome_label'] = int(h['outcome_label'])
        rows.append(row)
    return pd.DataFrame(rows)


def train_learning_model(min_rows: int = 80) -> dict:
    """Train a conservative probability model from completed recommendation outcomes."""
    ensure_learning_tables()
    hist = load_learning_history()
    train_df = _history_to_training_frame(hist)
    result = {'trained': False, 'rows_used': len(train_df), 'message': ''}
    if len(train_df) < int(min_rows):
        result['message'] = f'Need at least {min_rows} completed outcomes; currently {len(train_df)}.'
        return result
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score
        import joblib
    except Exception as e:
        result['message'] = f'scikit-learn/joblib not installed: {e}'
        return result
    X = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_df['outcome_label'].astype(int)
    if y.nunique() < 2:
        result['message'] = 'Need both successful and failed completed outcomes before training.'
        return result
    test_size = 0.25 if len(train_df) >= 120 else 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=5,
        min_samples_leaf=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURE_COLUMNS, 'trained_rows': len(train_df)}, MODEL_PATH)
    with connect() as con:
        con.execute('''
        INSERT INTO learning_model_runs(rows_used, feature_count, validation_rows, validation_accuracy, validation_auc, model_path, notes)
        VALUES(?,?,?,?,?,?,?)
        ''', (len(train_df), len(FEATURE_COLUMNS), len(X_test), acc, auc, str(MODEL_PATH), 'RandomForest swing success probability model'))
        con.commit()
    result.update({'trained': True, 'message': 'Model trained.', 'validation_accuracy': acc, 'validation_auc': auc, 'model_path': str(MODEL_PATH)})
    return result


def get_model_runs() -> pd.DataFrame:
    ensure_learning_tables()
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM learning_model_runs ORDER BY id DESC', con)


def apply_ai_probability(recs: pd.DataFrame) -> pd.DataFrame:
    """Add AI probability overlay if a trained model exists. Safe no-op otherwise."""
    out = recs.copy()
    if out.empty:
        return out
    out['ai_success_probability'] = np.nan
    out['ai_confidence'] = 'NO_MODEL'
    out['ai_adjusted_score'] = out.get('combined_signal_score', out.get('score', 0))
    out['ai_recommendation'] = out.get('final_recommendation', out.get('action', ''))
    out['ai_reason'] = 'No trained learning model yet; collect completed outcomes first.'
    if not MODEL_PATH.exists():
        return out
    try:
        import joblib
        bundle = joblib.load(MODEL_PATH)
        model = bundle['model']
        features = bundle.get('features', FEATURE_COLUMNS)
        X = pd.DataFrame(index=out.index)
        for c in features:
            X[c] = pd.to_numeric(out[c], errors='coerce') if c in out.columns else 0.0
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        prob = model.predict_proba(X[features])[:, 1] * 100
        base_score = pd.to_numeric(out.get('combined_signal_score', out.get('score', 0)), errors='coerce').fillna(0)
        out['ai_success_probability'] = np.round(prob, 1)
        out['ai_confidence'] = np.where(prob >= 65, 'HIGH', np.where(prob >= 55, 'MEDIUM', np.where(prob >= 45, 'LOW', 'NEGATIVE')))
        out['ai_adjusted_score'] = np.round(base_score + ((prob - 50) / 10), 2)
        rec = out.get('final_recommendation', out.get('action', pd.Series(['']*len(out), index=out.index))).astype(str)
        out['ai_recommendation'] = rec
        out.loc[(prob >= 65) & rec.str.contains('BUY|WATCH', case=False, na=False), 'ai_recommendation'] = 'AI_CONFIRMED_' + rec
        out.loc[(prob < 45) & rec.str.contains('BUY', case=False, na=False), 'ai_recommendation'] = 'WATCH_AI_WEAK'
        out.loc[(prob < 40) & rec.str.contains('WATCH|BUY', case=False, na=False), 'ai_recommendation'] = 'AVOID_AI_WEAK'
        out['ai_reason'] = 'AI overlay from completed recommendation outcomes; use as probability/risk input, not standalone advice.'
    except Exception as e:
        out['ai_reason'] = f'AI model load/prediction failed: {e}'
    return out


def learning_summary() -> dict:
    hist = load_learning_history()
    if hist.empty:
        return {'total': 0, 'completed': 0, 'success_rate': None, 'pending': 0, 'model_exists': MODEL_PATH.exists()}
    completed = hist.dropna(subset=['outcome_label'])
    return {
        'total': int(len(hist)),
        'completed': int(len(completed)),
        'pending': int(len(hist) - len(completed)),
        'success_rate': None if completed.empty else float(completed['outcome_label'].mean() * 100),
        'model_exists': MODEL_PATH.exists(),
    }
