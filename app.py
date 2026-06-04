import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from swingnse.pipeline import refresh, generate_recommendations, refresh_news_only, refresh_macro_only
from swingnse.nsefin_client import get_ltp
from swingnse.storage import (
    init_db, load_recommendations, load_positions, add_position, close_position,
    watch, unwatch, load_watchlist, load_prices, load_news_events, load_macro_snapshot
)

st.set_page_config(page_title='SwingNSE Desktop', page_icon='📈', layout='wide')
init_db()


def safe_numeric_col(df, col, default=0):
    # Return a numeric Series for col, even when the column is missing.
    if df is None or getattr(df, 'empty', True) or col not in df.columns:
        if df is None:
            return pd.Series(dtype='float64')
        return pd.Series([default] * len(df), index=df.index, dtype='float64')
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


st.markdown('''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {border:1px solid #ddd;border-radius:12px;padding:14px;background:#fafafa;}
.buy {color:#087f5b;font-weight:700}.watch{color:#0b63ce;font-weight:700}.sell{color:#c92a2a;font-weight:700}.neutral{color:#666;font-weight:700}
.small {font-size: 0.88rem; color:#666}
</style>
''', unsafe_allow_html=True)

st.title('📈 SwingNSE Desktop v10.1 — NSEFIN Daily Swing Scanner')
st.caption('Uses nsefin daily equity Bhavcopy pull + get_quote LTP syntax, local SQLite storage, deterministic technical scoring, structured news taxonomy, macro overlay, sector breadth and risk analytics. Educational/research use only.')

with st.sidebar:
    st.header('Daily Refresh')
    mode = st.radio('Universe mode', ['top500', 'symbols', 'all'], index=0,
                    help='top500 stores daily top 500 by volume; symbols uses symbols.txt; all stores full daily bhavcopy.')
    days = st.slider('Trading days to refresh', 30, 250, 120, 10)
    include_news = st.checkbox('Also refresh news/results for top symbols', value=False, help='Optional. Uses free/public NSE announcement and Google News RSS lookups; may be slower or blocked by source sites.')
    include_macro = st.checkbox('Also refresh macro regime', value=True, help='Adds Nifty trend, India VIX, USDINR, crude, and optional manual macro inputs to adjusted score.')
    news_limit = st.slider('News symbols limit', 20, 250, 100, 10)
    if st.button('🔄 Run nsefin Refresh', type='primary'):
        with st.spinner('Fetching Bhavcopies with nsefin and recalculating indicators...'):
            info, recs = refresh(days=days, symbols_path='symbols.txt', mode=mode, with_news=include_news, news_limit_symbols=news_limit, with_macro=include_macro)
        st.success(f"Fetched {info['days_collected']} days and generated {len(recs)} recommendations.")
        with st.expander('Refresh log'):
            st.write('\n'.join(info['logs']))
    if st.button('Recalculate from local DB'):
        recs = generate_recommendations()
        st.success(f'Regenerated {len(recs)} recommendations.')
    if st.button('📰 Refresh News & Results only'):
        with st.spinner('Fetching latest company announcements/news for existing recommendations...'):
            recs = refresh_news_only(limit_symbols=news_limit)
        st.success(f'News/results refreshed for existing recommendations. Rows: {len(recs)}')
    if st.button('🌐 Refresh Macro only'):
        with st.spinner('Fetching macro proxies and updating macro-adjusted recommendations...'):
            recs = refresh_macro_only()
        st.success(f'Macro refreshed. Rows updated: {len(recs)}')
    st.divider()
    st.markdown('**Files**')
    st.caption('DB: data/swingnse.db')
    st.caption('CSV: exports/recommendations.csv')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['Scanner', 'Positions', 'Watchlist', 'News & Results', 'Macro Regime', 'Sector Analytics', 'History', 'Setup'])

with tab1:
    recs = load_recommendations()
    if recs.empty:
        st.info('No recommendations yet. Run “Run nsefin Refresh” from the sidebar.')
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('Stocks scanned', len(recs))
        c2.metric('Final BUY/Strong', int(recs.get('final_recommendation', pd.Series(dtype=str)).astype(str).str.contains('BUY', na=False).sum()))
        c3.metric('Watch', int(recs.get('final_recommendation', recs['action']).astype(str).str.contains('WATCH', na=False).sum()))
        c4.metric('Latest date', str(recs['date'].max()))
        c5.metric('Avg sector score', round(safe_numeric_col(recs, 'sector_score').mean(), 2))
        actions = ['ALL'] + sorted(recs['action'].dropna().unique().tolist())
        cols = st.columns([2, 1, 1, 1])
        q = cols[0].text_input('Search symbol')
        action_filter = cols[1].selectbox('Action', actions)
        min_score = cols[2].number_input('Min score', value=int(recs['score'].min()), step=1)
        topn = cols[3].number_input('Show top N', value=100, min_value=10, max_value=1000, step=10)
        view = recs.copy()
        if q:
            view = view[view['symbol'].str.contains(q.upper(), na=False)]
        if action_filter != 'ALL':
            view = view[view['action'] == action_filter]
        view = view[view['score'] >= min_score].head(int(topn))
        display_cols = [c for c in ['symbol','date','close','sector','setup_grade','technical_score','analytics_score','trend_score','momentum_score','volume_score','liquidity_score','volatility_score','risk_quality_score','sector_score','news_score','event_caution_score','macro_score','sector_macro_adj','combined_signal_score','suggested_risk_pct','action','final_recommendation','news_adjusted_action','macro_adjusted_action','event_risk','top_news_type','top_news_confidence','top_news_materiality','macro_risk','macro_regime','news_summary','upcoming_event','event_summary','latest_event','avg_traded_value_20_cr','atr_pct','volatility_risk','ret_20d','ret_60d','rsi14','macd_hist','adx14','sma20','sma50','sma200','atr14','vol_ratio','stop_loss','target','rr','analytics_reason','news_audit_reason','final_signal_reason','reason'] if c in view.columns]
        st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
        st.download_button('Download recommendations CSV', view.to_csv(index=False), 'recommendations_filtered.csv', 'text/csv')
        st.subheader('Quick actions')
        with st.form('quick_actions'):
            symbol = st.selectbox('Symbol', view['symbol'].tolist() if not view.empty else recs['symbol'].tolist())
            col_a, col_b = st.columns(2)
            add_w = col_a.form_submit_button('⭐ Add to watchlist')
            add_p = col_b.form_submit_button('➕ Use in position form')
            if add_w:
                watch(symbol)
                st.success(f'{symbol} added to watchlist.')
            if add_p:
                st.session_state['position_symbol'] = symbol
                st.success('Open the Positions tab to add details.')

with tab2:
    st.subheader('Manual positions')
    st.caption('Position tracking uses latest stored close by default. Enable live LTP to use nsefin.get_quote(symbol)["lastPrice"], matching your working script.')
    recs = load_recommendations()
    latest_price = dict(zip(recs['symbol'], recs['close'])) if not recs.empty else {}
    use_live_ltp = st.checkbox('Use live LTP from nsefin.get_quote for positions', value=True)

    with st.form('add_position'):
        c1, c2, c3, c4, c5 = st.columns(5)
        default_symbol = st.session_state.get('position_symbol', '')
        symbol = c1.text_input('Symbol', value=default_symbol).upper()
        qty = c2.number_input('Qty', min_value=0.0, step=1.0)
        entry_default = float(latest_price.get(symbol, 0.0)) if symbol else 0.0
        entry = c3.number_input('Entry', min_value=0.0, value=entry_default, step=0.05)
        sl = c4.number_input('Stop loss', min_value=0.0, value=0.0, step=0.05)
        tgt = c5.number_input('Target', min_value=0.0, value=0.0, step=0.05)
        note = st.text_input('Note')
        if st.form_submit_button('Save position'):
            if symbol and qty > 0 and entry > 0:
                add_position(symbol, qty, entry, sl or None, tgt or None, note)
                st.success(f'Position saved for {symbol}.')
            else:
                st.error('Symbol, qty and entry are required.')

    positions = load_positions()
    if positions.empty:
        st.info('No open positions yet.')
    else:
        rows = []
        total_invested = 0.0
        total_current = 0.0
        wins = 0
        losses = 0
        risk_rows = []
        for _, p in positions.iterrows():
            live = get_ltp(p['symbol']) if use_live_ltp else None
            cur = float(live if live is not None else latest_price.get(p['symbol'], p['entry']))
            invested = p['entry'] * p['qty']
            current_val = cur * p['qty']
            pnl = (cur - p['entry']) * p['qty']
            pnl_pct = ((cur - p['entry']) / p['entry']) * 100 if p['entry'] else 0
            total_invested += invested
            total_current += current_val
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1
            sl_hit = pd.notna(p['stop_loss']) and p['stop_loss'] and cur <= p['stop_loss']
            tgt_hit = pd.notna(p['target']) and p['target'] and cur >= p['target']
            status = 'SL HIT' if sl_hit else 'TARGET HIT' if tgt_hit else 'OPEN'
            row = {**p.to_dict(), 'current': round(cur,2), 'invested': round(invested,2), 'current_value': round(current_val,2), 'pnl': round(pnl,2), 'pnl_pct': round(pnl_pct,2), 'status': status}
            rows.append(row)
            if sl_hit:
                risk_rows.append({'symbol': p['symbol'], 'current': round(cur,2), 'stop_loss': p['stop_loss'], 'status': 'EXIT REQUIRED'})

        posdf = pd.DataFrame(rows)
        total_pnl = total_current - total_invested
        ret = (total_pnl / total_invested * 100) if total_invested else 0
        win_rate = (wins / max(wins + losses, 1) * 100) if (wins + losses) else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Invested', f'₹{total_invested:,.0f}')
        c2.metric('Current Value', f'₹{total_current:,.0f}')
        c3.metric('Unrealized P&L', f'₹{total_pnl:,.0f}', f'{ret:.2f}%')
        c4.metric('Win Rate', f'{win_rate:.2f}%')

        if risk_rows:
            st.error('Stop loss breach detected')
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
        else:
            st.success('All positions within risk limits')

        st.dataframe(posdf[['id','symbol','qty','entry','current','invested','current_value','pnl','pnl_pct','stop_loss','target','status','note','created_at']], use_container_width=True, hide_index=True)
        st.line_chart(pd.DataFrame({'Equity':[total_invested, total_current]}, index=['Invested','Now']))
        close_id = st.number_input('Close position ID', min_value=0, step=1)
        if st.button('Close selected position') and close_id:
            close_position(int(close_id))
            st.success('Position closed. Refresh the tab to update view.')

with tab3:
    st.subheader('Watchlist')
    wl = load_watchlist()
    recs = load_recommendations()
    if wl.empty:
        st.info('No watchlist items yet.')
    else:
        merged = wl.merge(recs, on='symbol', how='left')
        st.dataframe(merged, use_container_width=True, hide_index=True)
        rem = st.text_input('Remove symbol').upper()
        if st.button('Remove from watchlist') and rem:
            unwatch(rem)
            st.success(f'{rem} removed.')

with tab4:
    st.subheader('Structured News Taxonomy & Event Caution')
    st.caption('v10 classifies each item by source quality, news type, materiality, sentiment, confidence and event caution before scoring. Low-value shareholder-return/price-move articles are neutralized.')
    recs = load_recommendations()
    events = load_news_events()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Rows stored', 0 if events.empty else len(events))
    c2.metric('Symbols covered', 0 if events.empty else events['symbol'].nunique())
    if not events.empty and 'item_type' in events.columns:
        c3.metric('Scored news', int(safe_numeric_col(events, 'news_sentiment_score').ne(0).sum()))
        c4.metric('Upcoming events', int(safe_numeric_col(events, 'event_caution_score').lt(0).sum()))
        c5.metric('Material catalysts', int(safe_numeric_col(events, 'materiality').gt(0).sum()))
    else:
        c3.metric('News items', 0)
        c4.metric('Upcoming events', 0)
        c5.metric('Material catalysts', 0)

    if st.button('📰 Fetch/refresh events now'):
        with st.spinner('Fetching NSE announcements/news and updating adjusted recommendations...'):
            recs = refresh_news_only(limit_symbols=100)
        st.success('News & Results refreshed. Go back to Scanner to see adjusted actions.')
        events = load_news_events()

    if not recs.empty:
        show_cols = [c for c in ['symbol','technical_score','news_score','event_caution_score','macro_score','sector_macro_adj','combined_signal_score','action','final_recommendation','news_adjusted_action','event_risk','top_news_type','top_news_confidence','top_news_materiality','news_event_count','upcoming_event_count','news_summary','upcoming_event','event_source','event_url','news_audit_reason','news_reason'] if c in recs.columns]
        st.markdown('**Recommendations with structured news classification**')
        sort_cols = [c for c in ['event_risk', 'combined_signal_score', 'event_adjusted_score', 'score'] if c in recs.columns]
        ascending = [False] * len(sort_cols)
        display_recs = recs[show_cols].copy()
        if sort_cols:
            # Use only sort columns that exist in the displayed frame; prevents KeyError on older DB/CSV schemas.
            sort_cols_display = [c for c in sort_cols if c in display_recs.columns]
            if sort_cols_display:
                display_recs = display_recs.sort_values(sort_cols_display, ascending=[False]*len(sort_cols_display))
        st.dataframe(display_recs, use_container_width=True, hide_index=True)

    if events.empty:
        st.info('No news/events stored yet. Click “Fetch/refresh events now” or enable news during daily refresh.')
    else:
        qsym = st.text_input('Filter events by symbol').upper()
        ev = events.copy()
        if qsym:
            ev = ev[ev['symbol'].str.contains(qsym, na=False)]
        ev_cols = [c for c in ['symbol','event_date','news_type','sentiment','news_sentiment_score','event_caution_score','event_risk','materiality','source_weight','confidence','matched_keyword','classification_reason','summary','headline','source','url','fetched_at'] if c in ev.columns]
        st.markdown('**News classification audit table**')
        st.dataframe(ev[ev_cols].head(500), use_container_width=True, hide_index=True)
        st.download_button('Download news_events.csv', ev.to_csv(index=False), 'news_events.csv', 'text/csv')

with tab5:
    st.subheader('Macro Regime')
    st.caption('Macro overlay uses free/public proxies: Nifty trend, India VIX, USDINR, crude, plus optional manual inputs from macro_inputs.csv for FII/FPI flows, RBI bias, inflation bias and results-season risk.')
    macro = load_macro_snapshot()
    recs = load_recommendations()
    if st.button('🌐 Fetch/refresh macro now'):
        with st.spinner('Fetching macro proxies and updating adjusted recommendations...'):
            recs = refresh_macro_only()
        macro = load_macro_snapshot()
        st.success('Macro regime refreshed. Go back to Scanner to see macro-adjusted actions.')
    if macro.empty:
        st.info('No macro snapshot yet. Click refresh macro or run daily refresh with macro enabled.')
    else:
        c1, c2, c3 = st.columns(3)
        total_macro = macro['macro_score'].fillna(0).sum()
        high_risk = int((macro['risk'] == 'HIGH').sum())
        latest_asof = macro['as_of'].iloc[0] if 'as_of' in macro.columns else ''
        c1.metric('Macro score', int(total_macro))
        c2.metric('High-risk metrics', high_risk)
        c3.metric('As of', str(latest_asof)[:19])
        st.dataframe(macro[['metric','value','macro_score','risk','note','source','as_of']], use_container_width=True, hide_index=True)
        st.download_button('Download macro_snapshot.csv', macro.to_csv(index=False), 'macro_snapshot.csv', 'text/csv')
    st.markdown('**Manual macro inputs**')
    st.caption('Create/edit macro_inputs.csv in the app folder with columns: metric,value,note. Supported metrics: fii_5d_cr, rbi_policy_bias, inflation_bias, results_season_risk. Use +1 positive, 0 neutral, -1 negative for bias fields.')
    sample = 'metric,value,note\nfii_5d_cr,0,manual FII/FPI 5-day net flow in crore\nrbi_policy_bias,0,dovish +1 neutral 0 hawkish -1\ninflation_bias,0,easing +1 neutral 0 high inflation -1\nresults_season_risk,0,0 normal -1 during uncertain results season\n'
    st.download_button('Download sample macro_inputs.csv', sample, 'macro_inputs.csv', 'text/csv')
    if not recs.empty:
        show_cols = [c for c in ['symbol','sector','sector_score','technical_score','analytics_score','news_score','event_caution_score','macro_score','sector_macro_adj','macro_sector_adj','combined_signal_score','final_recommendation','action','news_adjusted_action','macro_adjusted_action','macro_risk','macro_regime','news_summary','upcoming_event','macro_reason','analytics_reason'] if c in recs.columns]
        st.markdown('**Recommendations with macro adjustment**')
        st.dataframe(recs[show_cols].sort_values('combined_signal_score' if 'combined_signal_score' in recs.columns else 'macro_adjusted_score' if 'macro_adjusted_score' in recs.columns else 'score', ascending=False).head(500), use_container_width=True, hide_index=True)

with tab6:
    st.subheader('Sector Analytics')
    st.caption('Sector score is calculated from actual stored price data: 20D relative strength, breadth above SMA20/SMA50, and median returns. Extend sector_map.csv to improve coverage.')
    sector_path = Path('sector_snapshot.csv')
    recs = load_recommendations()
    if sector_path.exists():
        sector_df = pd.read_csv(sector_path)
    elif not recs.empty and 'sector' in recs.columns:
        sector_df = recs.groupby('sector', dropna=False).agg(
            symbols=('symbol','count'),
            avg_sector_score=('sector_score','mean'),
            avg_combined_score=('combined_signal_score','mean'),
            buy_watch=('final_recommendation', lambda x: x.astype(str).str.contains('BUY|WATCH', regex=True).sum())
        ).reset_index().rename(columns={'avg_sector_score':'sector_score'})
    else:
        sector_df = pd.DataFrame()
    if sector_df.empty:
        st.info('No sector snapshot yet. Run a refresh first. If many sectors show UNKNOWN, update sector_map.csv.')
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Sectors tracked', len(sector_df))
        c2.metric('Best sector', str(sector_df.sort_values('sector_score', ascending=False)['sector'].iloc[0]))
        c3.metric('Worst sector', str(sector_df.sort_values('sector_score', ascending=True)['sector'].iloc[0]))
        st.dataframe(sector_df.sort_values('sector_score', ascending=False), use_container_width=True, hide_index=True)
        st.download_button('Download sector_snapshot.csv', sector_df.to_csv(index=False), 'sector_snapshot.csv', 'text/csv')
    if not recs.empty and 'sector' in recs.columns:
        selected_sector = st.selectbox('Inspect sector stocks', ['ALL'] + sorted(recs['sector'].fillna('UNKNOWN').unique().tolist()))
        sv = recs.copy()
        if selected_sector != 'ALL':
            sv = sv[sv['sector'] == selected_sector]
        cols = [c for c in ['symbol','sector','setup_grade','final_recommendation','combined_signal_score','sector_score','trend_score','momentum_score','liquidity_score','volatility_risk','ret_20d','ret_60d','news_score','event_caution_score','macro_score','analytics_reason'] if c in sv.columns]
        st.dataframe(sv[cols].sort_values('combined_signal_score' if 'combined_signal_score' in sv.columns else 'score', ascending=False).head(300), use_container_width=True, hide_index=True)

with tab7:
    st.subheader('Historical stored data')
    prices = load_prices()
    if prices.empty:
        st.info('No historical rows yet.')
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Rows', len(prices))
        c2.metric('Symbols', prices['symbol'].nunique())
        c3.metric('Date range', f"{prices['date'].min().date()} → {prices['date'].max().date()}")
        sym = st.selectbox('View symbol', sorted(prices['symbol'].unique()))
        lookback = st.slider('Candlestick lookback days', 30, 250, 120, 10)
        sdata = prices[prices['symbol'] == sym].sort_values('date').tail(int(lookback)).copy()
        if not sdata.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=sdata['date'],
                open=sdata['open'], high=sdata['high'], low=sdata['low'], close=sdata['close'],
                name=sym
            )])
            if 'sma20' not in sdata.columns:
                # indicators are not stored in prices table; add quick moving averages for visual context.
                sdata['sma20'] = sdata['close'].rolling(20).mean()
                sdata['sma50'] = sdata['close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=sdata['date'], y=sdata['sma20'], mode='lines', name='SMA20'))
            fig.add_trace(go.Scatter(x=sdata['date'], y=sdata['sma50'], mode='lines', name='SMA50'))
            fig.update_layout(
                title=f'{sym} Candlestick History',
                xaxis_title='Date', yaxis_title='Price ₹',
                xaxis_rangeslider_visible=False,
                height=520,
                margin=dict(l=20, r=20, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sdata.tail(120), use_container_width=True, hide_index=True)

with tab8:
    st.subheader('Setup')
    st.markdown('''
**Recommended desktop flow**

1. Run `run_app.bat` on Windows or `./run_app.sh` on macOS/Linux.  
2. In the sidebar, choose `top500` and click **Run nsefin Refresh**.  
3. For daily updates, run `daily_refresh.bat` or use Windows Task Scheduler.  
4. Keep `data/swingnse.db` backed up.  

**Universe modes**

- `symbols`: scans only `symbols.txt`.
- `top500`: stores/scans top 500 by traded volume from each daily Bhavcopy.
- `all`: stores/scans the full equity Bhavcopy. Slower but broader.

This app deliberately uses your confirmed syntax: `nsefin.NSEClient().get_equity_bhav_copy(target_date)` for daily data and `nsefin.NSEClient().get_quote(symbol)["lastPrice"]` for live position LTP checks.

**Structured News Taxonomy & Event Caution layer**

- Optional enrichment. Your technical scanner keeps working even if news sources block requests.
- Uses free/public NSE corporate announcements where accessible, plus Google News RSS headlines as a fallback.
- Classifies every item into a taxonomy before scoring: governance risk, regulatory risk, results positive/negative, brokerage positive/negative, order win, upcoming results, capital-action caution, investor communication, price-move commentary, low-value shareholder-return commentary.
- Adds `source_weight`, `materiality`, `confidence`, `matched_keyword`, and `classification_reason` so every news score is auditable.
- Separates `news_score` from `event_caution_score`; upcoming events only add caution and do not create a buy signal.
- High-risk or upcoming events can downgrade a technical BUY into WATCH_EVENT_RISK / WATCH_EVENT_CAUTION.

**Macro Regime layer**

- Optional overlay for broader market context: Nifty trend, India VIX, USDINR, crude oil, and manual macro inputs.

**v10 Analytics layer**

- Adds sector breadth/relative-strength score from stored OHLCV data, not a constant zero.
- Adds liquidity score using average traded value, volatility risk using ATR%, setup grade, analytics score, and suggested risk % per trade.
- Extend `sector_map.csv` to improve coverage for symbols that show UNKNOWN.
- Final recommendation now includes technical + structured news + event caution + macro + sector + liquidity/volatility analytics.
- Adds `macro_score`, `macro_risk`, `macro_regime`, `combined_signal_score`, `final_recommendation`, and `final_signal_reason`.
- This layer should reduce aggressiveness during high volatility/risk-off markets instead of creating blind buy calls.
    ''')
