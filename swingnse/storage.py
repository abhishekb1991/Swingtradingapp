import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path('data/swingnse.db')


def connect(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute('PRAGMA journal_mode=WAL;')
    return con




def _table_columns(con, table_name: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _ensure_column(con, table_name: str, column_name: str, sql_type: str):
    if column_name not in _table_columns(con, table_name):
        con.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}")

def init_db():
    with connect() as con:
        con.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            series TEXT,
            open REAL, high REAL, low REAL, close REAL,
            prev_close REAL, volume REAL, value REAL, trades REAL,
            isin TEXT,
            PRIMARY KEY(symbol, date)
        )''')
        con.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            symbol TEXT PRIMARY KEY,
            date TEXT, close REAL, volume REAL, score INTEGER, action TEXT,
            rsi14 REAL, macd_hist REAL, adx14 REAL, sma20 REAL, sma50 REAL, sma200 REAL,
            atr14 REAL, bb_pos REAL, stoch14 REAL, vol_ratio REAL,
            stop_loss REAL, target REAL, rr REAL, reason TEXT,
            news_score INTEGER DEFAULT 0, event_caution_score INTEGER DEFAULT 0, event_risk TEXT DEFAULT 'LOW',
            latest_event TEXT, event_source TEXT, event_url TEXT,
            news_summary TEXT, upcoming_event TEXT, upcoming_event_count INTEGER DEFAULT 0,
            news_adjusted_score REAL, event_adjusted_score REAL,
            final_score REAL, news_adjusted_action TEXT, news_reason TEXT,
            macro_score REAL DEFAULT 0, macro_risk TEXT DEFAULT 'UNKNOWN', macro_regime TEXT,
            macro_reason TEXT, sector_macro_adj REAL DEFAULT 0, macro_adjusted_score REAL, macro_adjusted_action TEXT,
            event_summary TEXT, news_event_count INTEGER DEFAULT 0, technical_score REAL, combined_signal_score REAL,
            final_recommendation TEXT, final_signal_reason TEXT
        )''')
        # Migrate older v4 databases in-place with the new v5 news columns.
        for col, typ in {
            'news_score':'INTEGER DEFAULT 0', 'event_caution_score':'INTEGER DEFAULT 0', 'event_risk':"TEXT DEFAULT 'LOW'",
            'news_summary':'TEXT', 'upcoming_event':'TEXT', 'upcoming_event_count':'INTEGER DEFAULT 0',
            'news_adjusted_score':'REAL', 'event_adjusted_score':'REAL',
            'latest_event':'TEXT', 'event_source':'TEXT', 'event_url':'TEXT',
            'final_score':'REAL', 'news_adjusted_action':'TEXT', 'news_reason':'TEXT',
            'macro_score':'REAL DEFAULT 0', 'macro_risk':"TEXT DEFAULT 'UNKNOWN'", 'macro_regime':'TEXT',
            'macro_reason':'TEXT', 'sector_macro_adj':'REAL DEFAULT 0', 'macro_adjusted_score':'REAL', 'macro_adjusted_action':'TEXT',
            'event_summary':'TEXT', 'news_event_count':'INTEGER DEFAULT 0', 'technical_score':'REAL',
            'combined_signal_score':'REAL', 'final_recommendation':'TEXT', 'final_signal_reason':'TEXT',
            'top_news_type':'TEXT', 'top_news_confidence':'TEXT', 'top_news_materiality':'REAL DEFAULT 0', 'news_audit_reason':'TEXT'
        }.items():
            _ensure_column(con, 'recommendations', col, typ)
        con.execute('''
        CREATE TABLE IF NOT EXISTS news_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            event_date TEXT,
            source TEXT,
            category TEXT,
            headline TEXT,
            summary TEXT,
            url TEXT,
            event_score INTEGER,
            event_risk TEXT,
            item_type TEXT,
            news_sentiment_score INTEGER DEFAULT 0,
            event_caution_score INTEGER DEFAULT 0,
            sentiment TEXT,
            news_type TEXT,
            materiality INTEGER DEFAULT 0,
            source_weight REAL DEFAULT 0,
            confidence TEXT,
            matched_keyword TEXT,
            classification_reason TEXT,
            fetched_at TEXT,
            UNIQUE(symbol, headline, source)
        )''')

        # Migrate older news_events tables in-place with v8 split news/event columns.
        for col, typ in {
            'item_type':'TEXT', 'news_sentiment_score':'INTEGER DEFAULT 0',
            'event_caution_score':'INTEGER DEFAULT 0', 'sentiment':'TEXT',
            'news_type':'TEXT', 'materiality':'INTEGER DEFAULT 0', 'source_weight':'REAL DEFAULT 0',
            'confidence':'TEXT', 'matched_keyword':'TEXT', 'classification_reason':'TEXT'
        }.items():
            _ensure_column(con, 'news_events', col, typ)

        con.execute('''
        CREATE TABLE IF NOT EXISTS macro_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of TEXT,
            metric TEXT,
            value REAL,
            macro_score REAL,
            risk TEXT,
            note TEXT,
            source TEXT
        )''')
        con.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            entry REAL NOT NULL,
            stop_loss REAL,
            target REAL,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_open INTEGER DEFAULT 1
        )''')
        con.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        con.commit()


def upsert_prices(df: pd.DataFrame):
    if df.empty:
        return 0
    cols = ['symbol','date','series','open','high','low','close','prev_close','volume','value','trades','isin']
    for c in cols:
        if c not in df.columns:
            df[c] = None
    rows = df[cols].copy()
    rows['date'] = pd.to_datetime(rows['date']).dt.strftime('%Y-%m-%d')
    with connect() as con:
        rows.to_sql('prices_tmp', con, if_exists='replace', index=False)
        con.execute('''
        INSERT OR REPLACE INTO prices(symbol,date,series,open,high,low,close,prev_close,volume,value,trades,isin)
        SELECT symbol,date,series,open,high,low,close,prev_close,volume,value,trades,isin FROM prices_tmp
        ''')
        con.execute('DROP TABLE prices_tmp')
        con.commit()
    return len(rows)


def load_prices(symbols=None, min_days=0) -> pd.DataFrame:
    init_db()
    with connect() as con:
        if symbols:
            placeholders = ','.join(['?'] * len(symbols))
            q = f'SELECT * FROM prices WHERE symbol IN ({placeholders}) ORDER BY symbol,date'
            df = pd.read_sql_query(q, con, params=list(symbols))
        else:
            df = pd.read_sql_query('SELECT * FROM prices ORDER BY symbol,date', con)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df


def replace_recommendations(df: pd.DataFrame):
    init_db()
    with connect() as con:
        con.execute('DELETE FROM recommendations')
        if not df.empty:
            cols = list(_table_columns(con, 'recommendations'))
            rows = df.copy()
            for c in cols:
                if c not in rows.columns:
                    rows[c] = None
            rows = rows[[c for c in cols if c in rows.columns and c != 'rowid']]
            rows.to_sql('recommendations', con, if_exists='append', index=False)
        con.commit()


def load_recommendations() -> pd.DataFrame:
    init_db()
    with connect() as con:
        
        cols = _table_columns(con, 'recommendations')
        order_col = 'combined_signal_score' if 'combined_signal_score' in cols else 'score'
        return pd.read_sql_query(f'SELECT * FROM recommendations ORDER BY {order_col} DESC, volume DESC', con)


def add_position(symbol, qty, entry, stop_loss=None, target=None, note=''):
    with connect() as con:
        con.execute('INSERT INTO positions(symbol,qty,entry,stop_loss,target,note) VALUES(?,?,?,?,?,?)',
                    (symbol.upper(), qty, entry, stop_loss, target, note))
        con.commit()


def load_positions() -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM positions WHERE is_open=1 ORDER BY created_at DESC', con)


def close_position(position_id):
    with connect() as con:
        con.execute('UPDATE positions SET is_open=0 WHERE id=?', (position_id,))
        con.commit()


def watch(symbol, note=''):
    with connect() as con:
        con.execute('INSERT OR REPLACE INTO watchlist(symbol,note) VALUES(?,?)', (symbol.upper(), note))
        con.commit()


def unwatch(symbol):
    with connect() as con:
        con.execute('DELETE FROM watchlist WHERE symbol=?', (symbol.upper(),))
        con.commit()


def load_watchlist() -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM watchlist ORDER BY created_at DESC', con)


def replace_news_events(df: pd.DataFrame):
    init_db()
    with connect() as con:
        con.execute('DELETE FROM news_events')
        if df is not None and not df.empty:
            cols = ['symbol','event_date','source','category','headline','summary','url','event_score','event_risk','item_type','news_type','news_sentiment_score','event_caution_score','sentiment','materiality','source_weight','confidence','matched_keyword','classification_reason','fetched_at']
            rows = df.copy()
            for c in cols:
                if c not in rows.columns:
                    rows[c] = None
            rows[cols].to_sql('news_events', con, if_exists='append', index=False)
        con.commit()


def load_news_events(symbols=None) -> pd.DataFrame:
    init_db()
    with connect() as con:
        if symbols:
            placeholders = ','.join(['?'] * len(symbols))
            q = f'SELECT * FROM news_events WHERE symbol IN ({placeholders}) ORDER BY fetched_at DESC, id DESC'
            return pd.read_sql_query(q, con, params=list(symbols))
        return pd.read_sql_query('SELECT * FROM news_events ORDER BY fetched_at DESC, id DESC', con)


def replace_macro_snapshot(df: pd.DataFrame):
    init_db()
    with connect() as con:
        con.execute('DELETE FROM macro_snapshot')
        if df is not None and not df.empty:
            cols = ['as_of','metric','value','macro_score','risk','note','source']
            rows = df.copy()
            for c in cols:
                if c not in rows.columns:
                    rows[c] = None
            rows[cols].to_sql('macro_snapshot', con, if_exists='append', index=False)
        con.commit()


def load_macro_snapshot() -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query('SELECT * FROM macro_snapshot ORDER BY id DESC', con)
