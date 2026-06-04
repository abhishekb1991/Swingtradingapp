import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swingnse.news_engine import classify_item

CASES = [
    ("Wipro investors are sitting on a loss of 24% if they invested a year ago", "LOW_VALUE_COMMENTARY", 0, 0, "LOW"),
    ("Equitas Small Finance Bank latest 7.3% decline adds to one-year losses, institutional investors may consider drastic measures", "LOW_VALUE_COMMENTARY", 0, 0, "LOW"),
    ("CFO resigns from the company", "GOVERNANCE_RISK", -3, 0, "HIGH"),
    ("Analysts/Institutional Investor Meet/Con. Call Updates", "INVESTOR_COMMUNICATION", 0, 0, "LOW"),
    ("Board meeting to consider quarterly results", "UPCOMING_RESULTS", 0, -2, "MEDIUM"),
    ("Coforge among Motilal Oswal top picks", "BROKERAGE_POSITIVE", 1, 0, "LOW"),
    ("Company wins large order from Indian Railways", "ORDER_WIN", 1, 0, "LOW"),
    ("SEBI imposes penalty on company", "REGULATORY_RISK", -3, 0, "HIGH"),
]


def test_news_engine_regressions():
    for headline, typ, news, caution, risk in CASES:
        cls = classify_item(headline, source="Google News")
        assert cls["news_type"] == typ, (headline, cls)
        assert cls["news_sentiment_score"] == news, (headline, cls)
        assert cls["event_caution_score"] == caution, (headline, cls)
        assert cls["event_risk"] == risk, (headline, cls)

if __name__ == "__main__":
    test_news_engine_regressions()
    print("All news engine regression checks passed.")
