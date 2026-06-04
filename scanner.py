import argparse
from swingnse.pipeline import refresh, read_symbols, generate_recommendations, fetch_bhavcopies, refresh_news_only, refresh_macro_only
from swingnse.storage import init_db


def main():
    parser = argparse.ArgumentParser(description='SwingNSE desktop scanner using nsefin Bhavcopy daily data')
    parser.add_argument('--days', type=int, default=120, help='Available trading days to collect')
    parser.add_argument('--symbols', default='symbols.txt', help='Symbols file path')
    parser.add_argument('--mode', choices=['symbols','top500','all'], default='top500', help='symbols = symbols.txt only, top500 = daily top 500 by volume, all = full bhavcopy')
    parser.add_argument('--recommend-only', action='store_true', help='Do not fetch data, only regenerate recommendations from local DB')
    parser.add_argument('--with-news', action='store_true', help='Fetch free NSE/Google News event data and adjust scores')
    parser.add_argument('--news-only', action='store_true', help='Only refresh news/events for existing recommendations')
    parser.add_argument('--news-limit', type=int, default=100, help='Maximum top symbols to enrich with news')
    parser.add_argument('--with-macro', action='store_true', help='Fetch macro regime data and adjust scores')
    parser.add_argument('--macro-only', action='store_true', help='Only refresh macro regime and update existing recommendations')
    args = parser.parse_args()
    init_db()
    if args.macro_only:
        recs = refresh_macro_only()
        print(f'Refreshed macro regime and updated {len(recs)} recommendations.')
        print('Saved macro_snapshot.csv and exports/recommendations.csv.')
        return
    if args.news_only:
        recs = refresh_news_only(limit_symbols=args.news_limit)
        print(f'Refreshed news/events and updated {len(recs)} recommendations.')
        print('Saved exports/recommendations.csv and updated News & Results tab data.')
        return
    if args.recommend_only:
        symbols = read_symbols(args.symbols) if args.mode == 'symbols' else None
        recs = generate_recommendations(symbols=symbols, with_news=args.with_news, news_limit_symbols=args.news_limit, with_macro=args.with_macro)
        print(f'Generated {len(recs)} recommendations from local DB.')
        print('Saved exports/recommendations.csv')
        return
    fetch_info, recs = refresh(days=args.days, symbols_path=args.symbols, mode=args.mode, with_news=args.with_news, news_limit_symbols=args.news_limit, with_macro=args.with_macro)
    print(f"Fetched {fetch_info['days_collected']} days; stored/upserted {fetch_info['rows']} rows")
    for line in fetch_info['logs']:
        print(line)
    print(f'Generated {len(recs)} recommendations.')
    print('Saved exports/recommendations.csv')


if __name__ == '__main__':
    main()
