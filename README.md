# SwingNSE Desktop NSEFIN v10

Desktop swing-trading research dashboard for NSE stocks using your confirmed `nsefin` syntax for daily Bhavcopy refresh.

## What v10 changes

v10 structurally refactors the news layer. It no longer scores headlines directly from random positive/negative keywords. Every item now goes through:

```text
source quality -> news taxonomy -> materiality -> sentiment -> confidence -> impact score
```

New news fields include:

- `news_type`
- `materiality`
- `source_weight`
- `confidence`
- `matched_keyword`
- `classification_reason`
- `top_news_type`
- `top_news_confidence`
- `top_news_materiality`
- `news_audit_reason`

## Taxonomy categories

- `GOVERNANCE_RISK`
- `REGULATORY_RISK`
- `RESULTS_POSITIVE`
- `RESULTS_NEGATIVE`
- `BROKERAGE_POSITIVE`
- `BROKERAGE_NEGATIVE`
- `ORDER_WIN`
- `BUSINESS_POSITIVE`
- `UPCOMING_RESULTS`
- `CAPITAL_ACTION_CAUTION`
- `INVESTOR_COMMUNICATION`
- `PRICE_MOVE_COMMENTARY`
- `LOW_VALUE_COMMENTARY`
- `GENERIC_NEWS`

## Important classification behavior

- Resignations / auditor issues / defaults / penalties are always negative governance/regulatory risk.
- Investor meets / concall updates are informational, not caution by themselves.
- Upcoming results/board meeting for results add event caution, not positive or negative news.
- Historical shareholder-return articles like “investors are sitting on a loss...” score zero.
- Generic price-move articles like “shares fall/rise today” score zero because technicals already capture price action.

## Manual override file

You can maintain classification fixes without editing Python by updating:

```text
news_overrides.csv
```

Example:

```csv
symbol,contains,source_contains,forced_type,forced_sentiment,forced_score,forced_event_caution,forced_risk,notes
*,investors are sitting on a loss,,LOW_VALUE_COMMENTARY,NEUTRAL,0,0,LOW,Historical shareholder-return article; no fresh catalyst
```

## Run the app

```bat
run_app.bat
```

Or manually:

```bat
py -3 -m streamlit run app.py
```

## Full refresh

```bat
daily_refresh_with_news_macro.bat
```

## Run news regression tests

```bat
run_news_tests.bat
```

This checks examples such as Wipro shareholder-loss headline, Equitas decline headline, CFO resignation, investor meet, board meeting for results, COFORGE top pick, order win and SEBI penalty.

## Notes

This is an educational/research tool. It is not investment advice and should not be used as an automated trading system without backtesting, data licensing, risk controls and compliance review.


## v10 analytics upgrade

v10 adds a stronger analytics layer so the final recommendation is no longer just technical + news + macro.

New components:

- `sector` and `sector_score`: calculated from sector breadth and relative strength using stored OHLCV data.
- `trend_score`: structure across SMA20/SMA50/SMA200, ADX and 20-day trend.
- `momentum_score`: RSI, MACD and 20-day return quality.
- `volume_score`: volume expansion/weakness.
- `liquidity_score`: average traded value quality, useful for avoiding illiquid traps.
- `volatility_score` and `volatility_risk`: ATR% based risk assessment.
- `analytics_score`: combined market-structure score.
- `setup_grade`: A/B/C/D/E grade for quick triage.
- `suggested_risk_pct`: conservative risk-per-trade guidance for position sizing.
- `analytics_reason`: explainability string for the score.

Sector scoring uses `sector_map.csv`. If a stock shows `UNKNOWN`, add it to `sector_map.csv` as:

```csv
symbol,sector
ABC,PHARMA
XYZ,IT
```

The final signal now includes:

```text
technical_score + analytics_score + news_score + event_caution_score + macro_score + sector_macro_adj
```

This is still a research/decision-support tool, not investment advice.


## v10.1 patch
- Fixed dashboard metric crash when `sector_score`, `news_sentiment_score`, `event_caution_score`, or `materiality` columns are missing from older CSV/database outputs.
- Added `safe_numeric_col()` helper in `app.py` for backwards-compatible metrics.
