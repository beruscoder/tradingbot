import praw
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import re
import argparse
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extracter.log'),
        logging.StreamHandler()
    ]
)

# --- Reddit API Setup ---
reddit = praw.Reddit(
    client_id="apikey",
    client_secret="apikey",
    user_agent="apikey"
)

# --- Geography to Subreddit Mapping ---
geography_subreddits = {
    'UK': ['UKInvesting', 'FreetradeApp', 'UKPersonalFinance'],
    'Hong Kong': ['HongKongStocks', 'AsiaMarkets'],
    'India': ['IndianStockMarket', 'NSEIndia', 'IndiaInvestments'],
    'US': ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
}

# --- Load a list of valid stock tickers ---
def load_ticker_list(ticker_file="nas.csv"):
    try:
        tickers_df = pd.read_csv(ticker_file)
        if 'Symbol' not in tickers_df.columns:
            logging.error(f"'Symbol' column not found in {ticker_file}")
            raise ValueError(f"'Symbol' column not found in {ticker_file}")
        tickers = tickers_df['Symbol'].str.strip().tolist()
        return set(tickers)
    except FileNotFoundError:
        logging.error(f"Ticker file {ticker_file} not found")
        raise

# --- List of words that are capitalized but not stock tickers ---
exclude_words = {
    'YOLO', 'MOON', 'FOMO', 'DD', 'WSB', 'HODL', 'BULL', 'BEAR', 'IPO', 'ETF', 'SPAC',
    'FUD', 'LEAPS', 'MACD', 'TA', 'SEC', 'OTC', 'FTSE', 'COMEX', 'WRECK', 'MNQ',
    'BS', 'IF', 'ABOUT', 'NEWS', 'BEEN', 'ONLY', 'AND', 'TRAIN', 'COVID', 'PDF', 'JV', 'Q', 'FLY'
}

# --- Filter posts within specified days ---
def is_within_days(timestamp, days):
    post_time = datetime.fromtimestamp(timestamp)
    days_ago = datetime.now() - timedelta(days=days)
    return post_time >= days_ago

# --- Extract possible stock tickers from text ---
def extract_potential_tickers(text, valid_tickers_set):
    pattern = r'\b[A-Z]{1,5}\b'  # Regex for words of 1 to 5 capital letters
    matches = re.findall(pattern, text)
    return [m for m in matches if m in valid_tickers_set and m not in exclude_words]

# --- Validate ticker using yfinance ---
def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")
        return not info.empty
    except Exception as e:
        logging.warning(f"Error validating ticker {ticker}: {e}")
        return False

# --- Main data collection function ---
def collect_data(days, ticker_file, output_file, geography):
    try:
        # Validate geography
        if geography not in geography_subreddits:
            valid_geographies = list(geography_subreddits.keys())
            logging.error(f"Invalid geography: {geography}. Valid: {valid_geographies}")
            raise ValueError(f"Invalid geography: {geography}. Valid: {valid_geographies}")
        
        # Get subreddits for the selected geography
        subreddits = geography_subreddits[geography]
        logging.info(f"Collecting data for geography: {geography}, subreddits: {subreddits}")

        valid_tickers_set = load_ticker_list(ticker_file)
        data = []
        all_tickers = set()
        
        for subreddit_name in subreddits:
            subreddit = reddit.subreddit(subreddit_name)
            try:
                for post in subreddit.new(limit=1000):
                    if not is_within_days(post.created_utc, days):
                        continue
                    text = post.title + " " + (post.selftext or "")
                    potential_tickers = extract_potential_tickers(text, valid_tickers_set)
                    if potential_tickers:
                        for ticker in potential_tickers:
                            all_tickers.add(ticker)
                            data.append({
                                'date': datetime.fromtimestamp(post.created_utc).date(),
                                'ticker': ticker,
                                'subreddit': subreddit_name,
                                'title': post.title,
                                'text': text,
                                'geography': geography
                            })
            except Exception as e:
                logging.error(f"Error fetching posts from r/{subreddit_name}: {e}")
            time.sleep(1)  # Delay to avoid rate limits

        # --- Validate found tickers ---
        confirmed_tickers = []
        for ticker in all_tickers:
            if validate_ticker(ticker):
                confirmed_tickers.append(ticker)
            time.sleep(0.5)

        # --- Filter data to keep only valid tickers ---
        data = [d for d in data if d['ticker'] in confirmed_tickers]

        # --- Create Reddit data DataFrame ---
        reddit_df = pd.DataFrame(data)

        # --- Aggregate Reddit data by ticker, date, and geography ---
        reddit_summary = reddit_df.groupby(['ticker', 'date', 'geography']).agg({
            'text': lambda x: ' || '.join(x),
            'subreddit': lambda x: ', '.join(set(x)),
            'title': 'count'
        }).rename(columns={'title': 'post_count'}).reset_index()

        # --- Fetch historical stock prices for confirmed tickers ---
        stock_data = []
        history_period = f"{days + 1}d"  # Fetch one extra day to ensure coverage
        for ticker in confirmed_tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=history_period)
                if hist.empty:
                    continue
                hist['50_day_MA'] = hist['Close'].rolling(window=50, min_periods=1).mean()
                hist['200_day_MA'] = hist['Close'].rolling(window=200, min_periods=1).mean()
                for date, row in hist.iterrows():
                    if (datetime.now().date() - date.date()).days <= days:
                        stock_data.append({
                            'ticker': ticker,
                            'date': date.date(),
                            'close_price': row['Close'],
                            '50_day_MA': row['50_day_MA'],
                            '200_day_MA': row['200_day_MA'],
                            'geography': geography
                        })
            except Exception as e:
                logging.error(f"Error fetching yfinance data for {ticker}: {e}")
            time.sleep(0.5)

        # --- Create stock price DataFrame ---
        stock_df = pd.DataFrame(stock_data)

        # --- Merge Reddit and stock data ---
        merged_df = pd.merge(
            reddit_summary,
            stock_df,
            on=['ticker', 'date', 'geography'],
            how='outer'
        ).fillna({'text': '', 'subreddit': '', 'post_count': 0})

        # --- Save to CSV ---
        merged_df.to_csv(output_file, index=False)

        # --- Log summary ---
        logging.info(f"Data collection complete for {geography}. Output saved to {output_file}")
        logging.info(f"Valid tickers found: {confirmed_tickers}")
        logging.info(f"DataFrame head:\n{merged_df.head().to_string()}")

        return merged_df, confirmed_tickers

    except Exception as e:
        logging.error(f"Data collection failed: {e}")
        raise

def main(days=None, ticker_file="nas.csv", output_file="stock_data_for_valid_tickers.csv", geography=None):
    """
    Main function to run the data extraction process.
    
    Args:
        days (int, optional): Number of days to look back for Reddit posts.
        ticker_file (str, optional): Path to the ticker list CSV file.
        output_file (str, optional): Path to save the output CSV file.
        geography (str, optional): Geography for subreddit selection ('UK', 'Hong Kong', 'India', 'US').
    """
    try:
        # If arguments are not provided (e.g., for standalone execution), use argparse
        if days is None or geography is None:
            parser = argparse.ArgumentParser(description="Extract stock data from Reddit and yfinance")
            parser.add_argument('--days', type=int, default=None, help='Number of days to look back for Reddit posts')
            parser.add_argument('--ticker-file', default='nas.csv', help='Path to the ticker list CSV file')
            parser.add_argument('--output-file', default='stock_data_for_valid_tickers.csv', help='Path to save the output CSV')
            parser.add_argument('--geography', default=None, choices=geography_subreddits.keys(),
                               help=f"Geography to extract data from. Options: {', '.join(geography_subreddits.keys())}")
            args = parser.parse_args()

            # Prompt for days if not provided
            if args.days is None:
                while True:
                    try:
                        days_input = input("Enter the number of days to look back for Reddit posts (e.g., 7): ")
                        days = int(days_input)
                        if days <= 0:
                            print("Please enter a positive number of days.")
                            continue
                        break
                    except ValueError:
                        print("Please enter a valid integer for days.")
            else:
                days = args.days

            # Prompt for geography if not provided
            valid_geographies = list(geography_subreddits.keys())
            if args.geography is None:
                print(f"Available geographies: {', '.join(valid_geographies)}")
                while True:
                    geography = input("Enter the geography to extract data from: ").strip()
                    if geography in valid_geographies:
                        break
                    print(f"Invalid geography. Please choose from: {', '.join(valid_geographies)}")
            else:
                geography = args.geography

            ticker_file = args.ticker_file
            output_file = args.output_file

        # Run data collection
        merged_df, confirmed_tickers = collect_data(days, ticker_file, output_file, geography)

        # Print summary
        print(f"Data collection and preparation complete for {geography}. Output saved to '{output_file}'.")
        print(f"Valid tickers found: {confirmed_tickers}")
        print(merged_df.head())

    except Exception as e:
        logging.error(f"Extracter main failed: {e}")
        raise

if __name__ == "__main__":
    main()