import datetime as dt
from swingnse.nsefin_client import get_equity_bhav_copy_for_date, get_quote, get_ltp

print('Testing nsefin Bhavcopy syntax...')
for back in range(1, 8):
    d = dt.datetime.now() - dt.timedelta(days=back)
    try:
        df = get_equity_bhav_copy_for_date(d)
        print(f'{d:%Y-%m-%d}: rows={0 if df is None else len(df)}')
        if df is not None and not df.empty:
            print('Columns:', list(df.columns))
            print(df.head(3))
            break
    except Exception as e:
        print(f'{d:%Y-%m-%d}: {type(e).__name__}: {e}')

print('\nTesting nsefin quote syntax...')
for sym in ['RELIANCE', 'TCS', 'HDFCBANK']:
    try:
        q = get_quote(sym)
        print(sym, 'lastPrice=', q.get('lastPrice'), 'ltp helper=', get_ltp(sym))
    except Exception as e:
        print(sym, type(e).__name__, e)
