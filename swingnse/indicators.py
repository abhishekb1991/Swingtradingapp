import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series):
    fast = ema(close, 12)
    slow = ema(close, 26)
    line = fast - slow
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['symbol', 'date']).copy()
    out = []
    for symbol, g in df.groupby('symbol', sort=False):
        g = g.sort_values('date').copy()
        g['rsi14'] = rsi(g['close'])
        m_line, m_sig, m_hist = macd(g['close'])
        g['macd'] = m_line
        g['macd_signal'] = m_sig
        g['macd_hist'] = m_hist
        g['sma20'] = g['close'].rolling(20).mean()
        g['sma50'] = g['close'].rolling(50).mean()
        g['sma200'] = g['close'].rolling(200).mean()
        g['atr14'] = atr(g)
        g['adx14'] = adx(g)
        rolling_low = g['low'].rolling(14).min()
        rolling_high = g['high'].rolling(14).max()
        g['stoch14'] = 100 * (g['close'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        ma20 = g['close'].rolling(20).mean()
        sd20 = g['close'].rolling(20).std()
        g['bb_upper'] = ma20 + 2 * sd20
        g['bb_lower'] = ma20 - 2 * sd20
        g['bb_pos'] = (g['close'] - g['bb_lower']) / (g['bb_upper'] - g['bb_lower']).replace(0, np.nan)
        g['vol20'] = g['volume'].rolling(20).mean()
        g['vol_ratio'] = g['volume'] / g['vol20'].replace(0, np.nan)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score_latest(row: pd.Series) -> dict:
    score = 0
    reasons = []
    close = row.get('close')
    rsi_v = row.get('rsi14')
    adx_v = row.get('adx14')
    macd_h = row.get('macd_hist')
    bb_pos = row.get('bb_pos')
    stoch = row.get('stoch14')
    vol_ratio = row.get('vol_ratio')

    if pd.notna(rsi_v):
        if 40 <= rsi_v <= 62:
            score += 2; reasons.append('RSI in swing zone')
        elif rsi_v < 35:
            score += 1; reasons.append('RSI oversold rebound candidate')
        elif rsi_v > 72:
            score -= 2; reasons.append('RSI overbought')
    if pd.notna(macd_h):
        if macd_h > 0:
            score += 2; reasons.append('MACD bullish')
        else:
            score -= 1; reasons.append('MACD weak')
    if pd.notna(adx_v):
        if adx_v >= 22:
            score += 1; reasons.append('Trend strength acceptable')
        elif adx_v < 15:
            score -= 1; reasons.append('Weak trend')
    if pd.notna(close):
        if pd.notna(row.get('sma20')) and close > row['sma20']:
            score += 1; reasons.append('Above SMA20')
        else:
            score -= 1
        if pd.notna(row.get('sma50')) and close > row['sma50']:
            score += 1; reasons.append('Above SMA50')
        else:
            score -= 1
        if pd.notna(row.get('sma200')) and close > row['sma200']:
            score += 1; reasons.append('Long-term trend positive')
    if pd.notna(bb_pos):
        if 0.2 <= bb_pos <= 0.75:
            score += 1; reasons.append('Bollinger position healthy')
        elif bb_pos > 0.9:
            score -= 1; reasons.append('Near upper Bollinger band')
    if pd.notna(stoch):
        if 20 <= stoch <= 80:
            score += 1
        elif stoch > 90:
            score -= 1
    if pd.notna(vol_ratio) and vol_ratio >= 1.2:
        score += 1; reasons.append('Volume expansion')

    if score >= 7:
        action = 'BUY'
    elif score >= 4:
        action = 'WATCH'
    elif score <= -3:
        action = 'SELL/AVOID'
    else:
        action = 'NEUTRAL'

    atr = row.get('atr14')
    if pd.notna(atr) and pd.notna(close):
        sl = round(close - 1.5 * atr, 2)
        tp = round(close + 2.5 * atr, 2)
    else:
        sl = round(close * 0.97, 2) if pd.notna(close) else None
        tp = round(close * 1.06, 2) if pd.notna(close) else None
    rr = round((tp - close) / max(close - sl, 0.01), 2) if sl and tp and close else None

    return {'score': int(score), 'action': action, 'stop_loss': sl, 'target': tp, 'rr': rr, 'reason': '; '.join(reasons[:5])}
