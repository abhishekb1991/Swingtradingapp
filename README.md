# SwingNSE Desktop NSEFIN v11

Personal swing-trading research dashboard for NSE stocks using your confirmed `nsefin` syntax for daily Bhavcopy refresh and live quote checks.

## What v11 adds

v11 adds a controlled learning loop:

1. **Recommendation history**: every refresh stores one snapshot per symbol/date.
2. **Outcome tracker**: after 20 future trading days are available, the app calculates 5D/10D/20D return, max favourable/adverse move, target hit and stop-loss hit.
3. **Mistake review**: failed signals are labelled as false positives, stop-loss failures, weak long calls or missed winners.
4. **AI probability model**: once enough completed outcomes exist, a Random Forest model can be trained manually to generate `ai_success_probability`.
5. **AI overlay**: AI does not replace the rules. It adds probability/risk context through `ai_success_probability`, `ai_confidence`, `ai_adjusted_score` and `ai_recommendation`.

## Run the app

```bat
run_app.bat
```

or manually:

```bat
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

## Daily refresh

```bat
daily_refresh_with_news_macro.bat
```

This fetches Bhavcopy via `nsefin`, recalculates technicals/news/macro/analytics, applies the trained AI model if available, stores recommendations, and records the daily recommendation history.

## Learning workflow

The AI model cannot learn immediately on day 1. It needs completed future outcomes.

Recommended workflow:

1. Run daily refresh for several weeks.
2. Go to the **Learning** tab.
3. Click **Update outcomes from stored OHLCV**.
4. Once you have enough completed outcomes, click **Train / refresh AI probability model**.
5. Future scanner runs will add AI probability columns.

Batch files:

```bat
update_learning.bat
train_ai_model.bat
run_learning_pipeline.bat
```

## Important design choice

The model is intentionally **manual/controlled**, not self-updating after each failed recommendation. This avoids overfitting and unstable behaviour. Train only after enough completed outcomes and review the results.

## Files created

- `data/swingnse.db`: local SQLite database
- `recommendation_history`: table inside SQLite storing all recommendation snapshots
- `models/swing_success_model.pkl`: trained AI probability model, created only after training
- `exports/recommendations.csv`: latest recommendations
- `recommendations.csv`: compatibility CSV

## Notes

This is a research/decision-support tool, not financial advice, not auto-trading software, and not a guarantee of returns.

## Preserving positions from v10 / v10.1

Do not delete your old v10 folder. To migrate positions/watchlist/history into v11:

1. Unzip v11 into a new folder.
2. Run v11 once so the database is initialized.
3. Double-click `migrate_from_v10.bat`.
4. Paste the full path to your old v10/v10.1 folder, for example:
   `C:\Users\Admin\Downloads\SwingNSE_Desktop_NSEFIN_v10_1`

The migration script backs up the v11 database first and does not modify the v10 folder.

If you update GitHub, avoid overwriting your live `data/swingnse.db` unless you have migrated/backed it up.
