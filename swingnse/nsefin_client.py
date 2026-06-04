"""Small nsefin wrapper using the syntax the user confirmed works.

Do not change these calls unless nsefin changes its API:
    nse = nsefin.NSEClient()
    nse.get_equity_bhav_copy(target_date)
    nse.get_quote(symbol)
"""
import datetime as dt
from functools import lru_cache
from typing import Any, Optional

import pandas as pd


@lru_cache(maxsize=1)
def client():
    import nsefin
    return nsefin.NSEClient()


def get_equity_bhav_copy_for_date(target_date: dt.date | dt.datetime) -> pd.DataFrame:
    """Fetch one daily equity bhavcopy using nsefin's confirmed syntax."""
    if isinstance(target_date, dt.date) and not isinstance(target_date, dt.datetime):
        target_date = dt.datetime.combine(target_date, dt.time.min)
    nse = client()
    df = nse.get_equity_bhav_copy(target_date)
    if df is None:
        return pd.DataFrame()
    return df


def get_quote(symbol: str) -> dict[str, Any]:
    """Fetch live-ish NSE quote using nsefin's confirmed quote syntax."""
    nse = client()
    return nse.get_quote(str(symbol).upper().strip())


def get_ltp(symbol: str) -> Optional[float]:
    """Return last traded price from nsefin quote with common fallback keys."""
    try:
        quote = get_quote(symbol)
        for key in ["lastPrice", "last_price", "ltp", "LTP", "last"]:
            if key in quote and quote[key] is not None:
                return float(str(quote[key]).replace(",", ""))
    except Exception:
        return None
    return None
