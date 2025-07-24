import pandas as pd
import torch
import sys
import json
import nltk
import logging
import re
from nltk.corpus import stopwords
from fuzzywuzzy import process, fuzz
import yfinance as yf
import time
import os
from datetime import datetime, timedelta
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import argparse

os.environ["HUGGINGFACE_TOKEN"] = "apikey"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.FileHandler('sentiment.log'), logging.StreamHandler()])

# Check PyTorch and dependencies
if torch.__version__ < '2.4.1':
    print(f"Error: PyTorch version {torch.__version__} is too old. Upgrade with: pip3 install --upgrade torch>=2.4.1")
    sys.exit(1)
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    sys.exit(1)
stop_words = set(stopwords.words('english'))

# Load FinBERT model
try:
    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert', use_safetensors=True)
except Exception as e:
    print(f"Error loading FinBERT model: {e}")
    sys.exit(1)
sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Sector keywords
sector_keywords = {
    'Technology': ['AI', 'tech', 'software', 'hardware', 'cloud', 'chip', 'semiconductor', 'digital', 'algorithm'],
    'Financials': ['bank', 'finance', 'investment', 'stock', 'bond', 'yield', 'interest rate', 'federal reserve'],
    'Healthcare': ['medical', 'health', 'pharma', 'biotech', 'hospital', 'fda', 'vaccine', 'treatment'],
    'Energy': ['oil', 'gas', 'energy', 'renewable', 'solar', 'wind', 'drilling', 'OPEC', 'petroleum'],
    'Consumer': ['retail', 'consumer', 'product', 'brand', 'shopping', 'e-commerce', 'fashion', 'apparel'],
    'Industrials': ['manufacturing', 'factory', 'industrial', 'machinery', 'construction', 'supply chain', 'logistics']
}

# Financial events and risk factors
financial_events = {
    'earnings': ['earnings', 'results', 'quarterly', 'Q1', 'Q2', 'Q3', 'Q4', 'profit', 'revenue', 'EPS'],
    'merger': ['merger', 'acquisition', 'takeover', 'buyout', 'M&A', 'consolidation'],
    'product': ['product', 'launch', 'new product', 'innovation', 'release', 'rollout'],
    'regulation': ['regulation', 'compliance', 'legal', 'lawsuit', 'settlement', 'fine', 'investigation'],
    'leadership': ['CEO', 'CFO', 'management', 'executive', 'resign', 'appoint', 'board'],
    'dividend': ['dividend', 'payout', 'stock split', 'share repurchase', 'buyback'],
    'guidance': ['guidance', 'forecast', 'outlook', 'projection', 'estimate'],
    'economic': ['economic', 'inflation', 'interest rate', 'recession', 'macro', 'geopolitical']
}
risk_factors = {
    'regulatory': ['regulation', 'compliance', 'legal', 'lawsuit', 'settlement', 'fine', 'investigation'],
    'market': ['volatility', 'competition', 'market share', 'disruption', 'demand'],
    'financial': ['debt', 'liquidity', 'margin', 'losses', 'bankruptcy', 'cash flow'],
    'operational': ['supply chain', 'production', 'recall', 'disruption', 'outage'],
    'macro': ['recession', 'inflation', 'interest rates', 'geopolitical', 'trade war']
}

# Event weights for final score
event_weights = {
    'earnings': 1.0,
    'merger': 1.0,
    'product': 0.8,
    'regulation': 0.8,
    'leadership': 0.7,
    'dividend': 0.8,
    'guidance': 0.7,
    'economic': 0.5
}

# Load valid ticker list
def load_ticker_list(ticker_file="nas.csv"):
    try:
        tickers_df = pd.read_csv(ticker_file, header=0, encoding='utf-8-sig')
        if 'Symbol' not in tickers_df.columns:
            print(f"Error: 'Symbol' column not found in {ticker_file}. Available columns: {list(tickers_df.columns)}")
            sys.exit(1)
        tickers = tickers_df['Symbol'].str.strip().dropna().tolist()
        return set(tickers)
    except FileNotFoundError:
        print(f"Error: '{ticker_file}' not found. Download from https://www.nasdaq.com/market-activity/stocks/screener.")
        sys.exit(1)

# Load ticker cache
def load_ticker_cache(filename='ticker_corrections_cache.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save ticker cache
def save_ticker_cache(cache, filename='ticker_corrections_cache.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving ticker cache: {e}")

# Extract tickers from text
def extract_tickers_from_text(text):
    if not isinstance(text, str) or pd.isna(text) or len(text.strip()) == 0:
        return []
    return re.findall(r'\b\$?[A-Z]{1,5}\b', text)

# Correct ticker and identify primary stock
def correct_ticker(ticker, valid_tickers, text=''):
    global ticker_cache
    if not isinstance(ticker, str) or pd.isna(ticker) or len(ticker.strip()) == 0:
        logging.warning(f"Invalid ticker input: {ticker}")
        return None, None
    ticker = ticker.strip()
    
    text_tickers = extract_tickers_from_text(text)
    text_tickers = [t.replace('$', '') for t in text_tickers if t.replace('$', '') in valid_tickers]
    primary_stock = text_tickers[0] if text_tickers else ticker
    
    if ticker in ticker_cache:
        corrected_ticker = ticker_cache[ticker]
        return corrected_ticker, primary_stock if primary_stock in valid_tickers else corrected_ticker
    if ticker in valid_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info.get('symbol') == ticker:
                ticker_cache[ticker] = ticker
                save_ticker_cache(ticker_cache)
                return ticker, primary_stock if primary_stock in valid_tickers else ticker
        except Exception as e:
            logging.warning(f"Error validating {ticker} with yfinance: {e}")
    
    match = process.extractOne(ticker.lower(), [t.lower() for t in valid_tickers], scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 85:
        best_match = next(t for t in valid_tickers if t.lower() == match[0])
        ticker_cache[ticker] = best_match
        save_ticker_cache(ticker_cache)
        logging.info(f"Fuzzy matched {ticker} to {best_match}")
        return best_match, primary_stock if primary_stock in valid_tickers else best_match
    
    logging.warning(f"No valid match for ticker {ticker}")
    return ticker, primary_stock

# Infer sector from text
def infer_sector_from_text(text):
    if not isinstance(text, str) or len(text) < 10:
        return 'Unknown'
    text_lower = text.lower()
    sector_scores = {sector: 0 for sector in sector_keywords}
    for sector, keywords in sector_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                sector_scores[sector] += 1
    if sum(sector_scores.values()) == 0:
        return 'Unknown'
    return max(sector_scores, key=sector_scores.get)

# Get sector mapping
def get_sector_mapping(tickers, df):
    sector_mapping = load_ticker_cache('sector_mapping.json')
    ticker_texts = df.groupby('ticker')['text'].apply(lambda x: ' '.join(x.dropna().astype(str))).to_dict()
    missing_tickers = [t for t in tickers if t and t not in sector_mapping]
    
    for ticker in missing_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            sector = info.get('sector', 'Unknown')
            if sector != 'Unknown' and sector != 'N/A':
                sector_mapping[ticker] = sector
            else:
                text = ticker_texts.get(ticker, '')
                sector_mapping[ticker] = infer_sector_from_text(text)
            time.sleep(1)
        except Exception as e:
            logging.warning(f"Failed to fetch sector for {ticker}: {e}")
            text = ticker_texts.get(ticker, '')
            sector_mapping[ticker] = infer_sector_from_text(text)
    
    save_ticker_cache(sector_mapping, 'sector_mapping.json')
    return sector_mapping

# Fetch stock data
def get_stock_data(ticker, date):
    try:
        stock = yf.Ticker(ticker)
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=250)
        hist = stock.history(start=start_date, end=end_date + timedelta(days=1))
        if hist.empty:
            return None, None, None
        hist['50_day_MA'] = hist['Close'].rolling(window=50).mean()
        hist['200_day_MA'] = hist['Close'].rolling(window=200).mean()
        latest = hist.iloc[-1]
        return latest['Close'], latest['50_day_MA'], latest['200_day_MA']
    except Exception as e:
        logging.warning(f"Error fetching stock data for {ticker} on {date}: {e}")
        return None, None, None

# Summarize text
def summarize_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "No text available"
    summary = text[:100].strip()
    if len(text) > 100:
        summary += "..."
    sentences = nltk.sent_tokenize(text)
    key_phrases = []
    for sentence in sentences:
        for keyword in sum([keywords for keywords in risk_factors.values()] + [keywords for keywords in financial_events.values()], []):
            if keyword.lower() in sentence.lower():
                key_phrases.append(keyword)
    if key_phrases:
        summary += f" Keywords: {', '.join(set(key_phrases[:3]))}"
    return summary

# Sentiment analysis
def analyze_sentiment(text):
    if not isinstance(text, str) or len(text) < 10:
        return None, None
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
        sentiment_id = torch.argmax(probs).item()
        return sentiment_map[sentiment_id], torch.max(probs).item()
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        return None, None

# Detect stock-specific events
def detect_stock_specific_events(text, ticker, sector):
    if not isinstance(text, str) or len(text.strip()) == 0:
        logging.info(f"No events detected for {ticker}: Empty text")
        return []
    text_lower = text.lower()
    text_tickers = extract_tickers_from_text(text)
    text_tickers = [t.replace('$', '').lower() for t in text_tickers]
    events = []
    
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for event_type, keywords in financial_events.items():
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                if ticker.lower() in text_lower or any(t in text_lower for t in text_tickers):
                    events.append(event_type)
                    logging.info(f"Event detected for {ticker}: {event_type} in sentence: {sentence[:50]}...")
    
    events = list(set(events))
    if not events:
        logging.info(f"No events detected for {ticker} in text: {text[:50]}...")
    return events

# Extract risk factors
def extract_stock_risk_factors(text, ticker, sector):
    risk_data = []
    if not isinstance(text, str) or len(text.strip()) == 0:
        logging.info(f"No risk factors detected for {ticker}: Empty text")
        return risk_data
    sentences = nltk.sent_tokenize(text)
    text_tickers = extract_tickers_from_text(text)
    text_tickers = [t.replace('$', '') for t in text_tickers]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if not (ticker.lower() in sentence_lower or any(t.lower() in sentence_lower for t in text_tickers)):
            continue
        for risk_type, keywords in risk_factors.items():
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                sentiment, confidence = analyze_sentiment(sentence)
                if sentiment:
                    risk_data.append({
                        'ticker': ticker,
                        'risk_type': risk_type,
                        'sentence': sentence,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'sector_relevant': any(keyword in sector_keywords.get(sector, []) for keyword in keywords)
                    })
                    logging.info(f"Risk factor detected for {ticker}: {risk_type} in sentence: {sentence[:50]}...")
    if not risk_data:
        logging.info(f"No risk factors detected for {ticker} in text: {text[:50]}...")
    return risk_data

# Calculate per-row component scores
def calculate_component_scores(row):
    # Sentiment score
    net_sentiment = row['net_sentiment'] if pd.notna(row['net_sentiment']) else 0
    sentiment_confidence = row['sentiment_confidence'] if pd.notna(row['sentiment_confidence']) else 0
    sentiment_score = (net_sentiment * sentiment_confidence) * 50 + 50
    
    # Event score
    events = row['stock_events']
    num_events = len(events)
    event_weight = sum(event_weights.get(event, 0.5) for event in events)
    sentiment_modifier = {'positive': 1.0, 'neutral': 0.5, 'negative': -0.5}.get(row['sentiment_label'], 0.5)
    event_score = min(100, (num_events * event_weight * sentiment_modifier) * 33.33)
    
    # Technical score
    technical_score = 75 if pd.notna(row['50_day_MA']) and pd.notna(row['200_day_MA']) and row['50_day_MA'] > row['200_day_MA'] else 25
    
    # Risk penalty
    risk_factors = row['stock_risk_factors']
    num_risks = len(risk_factors)
    avg_risk_confidence = sum(r['confidence'] for r in risk_factors) / num_risks if num_risks > 0 else 0
    risk_penalty = min(100, num_risks * 20 * avg_risk_confidence)
    
    return pd.Series({
        'sentiment_score': sentiment_score,
        'event_score': event_score,
        'technical_score': technical_score,
        'risk_penalty': risk_penalty,
        'post_count': row['post_count'],
        'date': row['date']
    })

# Aggregate scores per stock
def aggregate_stock_scores(group, sector_summary):
    # Calculate weights
    max_date = pd.to_datetime(group['date']).max()
    group['days_from_max'] = (max_date - pd.to_datetime(group['date'])).dt.days
    max_days = group['days_from_max'].max() if group['days_from_max'].max() > 0 else 1
    group['recency_weight'] = group['days_from_max'].apply(lambda x: max(0.5, 1 - x / max_days))
    total_post_count = group['post_count'].sum()
    group['post_count_weight'] = group['post_count'] / total_post_count if total_post_count > 0 else 1
    group['weight'] = group['post_count_weight'] * group['recency_weight']
    total_weight = group['weight'].sum()
    group['weight'] = group['weight'] / total_weight if total_weight > 0 else 1
    
    # Aggregate scores
    sentiment_score = (group['sentiment_score'] * group['weight']).sum()
    event_score = (group['event_score'] * group['weight']).sum()
    technical_score = (group['technical_score'] * group['weight']).sum()
    risk_penalty = (group['risk_penalty'] * group['weight']).sum()
    
    # Sector score
    sector = group['sector'].iloc[0]
    sector_avg_sentiment = sector_summary.get(sector, 0) if isinstance(sector_summary, dict) else 0
    sector_score = (sector_avg_sentiment * 0.5 + 0.5) * 100
    
    # Total events
    total_events = sum(len(events) for events in group['stock_events'])
    
    # Final score
    final_score = (0.4 * sentiment_score) + (0.2 * event_score) + (0.2 * sector_score) + (0.1 * technical_score) - (0.1 * risk_penalty)
    final_score = max(0, min(100, final_score))
    
    return pd.Series({
        'primary_stock': group['primary_stock'].iloc[0],
        'final_score': final_score,
        'avg_sentiment_score': sentiment_score,
        'avg_event_score': event_score,
        'sector_score': sector_score,
        'avg_technical_score': technical_score,
        'avg_risk_penalty': risk_penalty,
        'total_events': total_events,
        'sector': sector,
        'total_post_count': total_post_count,
        'last_date': max_date
    })

def main(input_file='stock_data_for_valid_tickers.csv', output_file='stock_final_scores.csv'):
    # Parse command-line arguments if not provided
    if input_file is None or output_file is None:
        parser = argparse.ArgumentParser(description="Perform sentiment analysis on stock data")
        parser.add_argument('--input-file', default='stock_data_for_valid_tickers.csv', 
                            help='Path to the input CSV file (default: stock_data_for_valid_tickers.csv)')
        parser.add_argument('--output-file', default='stock_final_scores.csv',
                            help='Path to the output CSV file for stock scores (default: stock_final_scores.csv)')
        args = parser.parse_args()
        input_file = args.input_file
        output_file = args.output_file

    # Load dataset
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)

    # Validate columns
    required_columns = ['ticker', 'text', 'subreddit', 'post_count', 'date', 'geography']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns: {missing_columns}")
        sys.exit(1)

    # Debug: Inspect data
    logging.info("Ticker distribution:")
    logging.info(df['ticker'].value_counts().to_string())
    logging.info("Rows with single-letter tickers:")
    logging.info(df[df['ticker'].str.len() == 1][['ticker', 'text']].head(10).to_string())
    logging.info(f"NaN text rows: {df['text'].isna().sum()}")

    # Clean text column
    df['text'] = df['text'].fillna('').astype(str).str.strip()

    # Initialize ticker cache
    global ticker_cache
    ticker_cache = load_ticker_cache()
    valid_tickers_set = load_ticker_list()

    # Correct tickers and identify primary stock
    df[['ticker', 'primary_stock']] = df.apply(
        lambda row: pd.Series(correct_ticker(row['ticker'], valid_tickers_set, row['text'])),
        axis=1
    )

    # Filter out empty text rows
    df = df[df['text'] != '']

    # Add date if missing
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime('%Y-%m-%d')

    # Fetch stock data
    df[['close_price', '50_day_MA', '200_day_MA']] = df.apply(
        lambda row: pd.Series(get_stock_data(row['ticker'], row['date'])),
        axis=1
    )

    # Get sector mapping
    unique_tickers = df['ticker'].unique()
    sector_mapping = get_sector_mapping(unique_tickers, df)
    df['sector'] = df['ticker'].map(sector_mapping)

    # Apply sentiment and risk analysis
    df['sentiment_label'], df['sentiment_confidence'] = zip(*df['text'].apply(analyze_sentiment))
    df['stock_events'] = df.apply(
        lambda row: detect_stock_specific_events(row['text'], row['primary_stock'], row['sector']),
        axis=1
    )
    df['stock_risk_factors'] = df.apply(
        lambda row: extract_stock_risk_factors(row['text'], row['primary_stock'], row['sector']),
        axis=1
    )

    # Calculate fear/greed metrics
    def calculate_fear_greed(row):
        if row['sentiment_label'] == 'negative':
            fear = min(1.0, row['sentiment_confidence'] * 1.2) if row['sentiment_confidence'] else 0.0
            greed = max(0.0, 0.5 - (row['sentiment_confidence'] / 2)) if row['sentiment_confidence'] else 0.0
        elif row['sentiment_label'] == 'positive':
            greed = min(1.0, row['sentiment_confidence'] * 1.2) if row['sentiment_confidence'] else 0.0
            fear = max(0.0, 0.5 - (row['sentiment_confidence'] / 2)) if row['sentiment_confidence'] else 0.0
        else:
            fear = row['sentiment_confidence'] * 0.3 if row['sentiment_confidence'] else 0.0
            greed = row['sentiment_confidence'] * 0.3 if row['sentiment_confidence'] else 0.0
        return fear, greed, greed - fear

    df[['fear_score', 'greed_score', 'net_sentiment']] = df.apply(
        lambda row: pd.Series(calculate_fear_greed(row)), axis=1
    )

    # Generate trading signals
    def generate_signal(row):
        if row['net_sentiment'] > 0.6 and row['sentiment_confidence'] and row['sentiment_confidence'] > 0.7:
            return 'strong_buy'
        elif row['net_sentiment'] > 0.3 and row['sentiment_confidence'] and row['sentiment_confidence'] > 0.6:
            return 'buy'
        elif row['net_sentiment'] < -0.6 and row['sentiment_confidence'] and row['sentiment_confidence'] > 0.7:
            return 'strong_sell'
        elif row['net_sentiment'] < -0.3 and row['sentiment_confidence'] and row['sentiment_confidence'] > 0.6:
            return 'sell'
        else:
            return 'hold'

    df['trading_signal'] = df.apply(generate_signal, axis=1)

    # Summarize risk factors
    def summarize_risks(risk_factors):
        if not risk_factors:
            return "None"
        summary = []
        for risk in risk_factors:
            summary.append(f"{risk['risk_type']}: {risk['sentiment']} ({risk['confidence']:.2f})")
        return "; ".join(summary)

    df['risk_summary'] = df['stock_risk_factors'].apply(summarize_risks)

    # Summarize text
    df['text_summary'] = df['text'].apply(summarize_text)

    # Format events
    df['events'] = df['stock_events'].apply(lambda x: ", ".join(x) if x else "None")

    # Calculate component scores
    df[['sentiment_score', 'event_score', 'technical_score', 'risk_penalty', 'post_count', 'date']] = df.apply(
        calculate_component_scores, axis=1
    )

    # Create sector summary
    sector_summary_df = df.groupby(['sector', 'geography']).agg(
        avg_sentiment=('net_sentiment', 'mean'),
        fear_score=('fear_score', 'mean'),
        greed_score=('greed_score', 'mean'),
        buy_signals=('trading_signal', lambda x: (x.isin(['buy', 'strong_buy'])).sum()),
        sell_signals=('trading_signal', lambda x: (x.isin(['sell', 'strong_sell'])).sum())
    ).reset_index()
    sector_summary = sector_summary_df.set_index(['sector', 'geography'])['avg_sentiment'].to_dict()

    # Aggregate scores per stock
    stock_scores = df.groupby(['primary_stock', 'geography']).apply(
        lambda group: aggregate_stock_scores(group, sector_summary)
    ).reset_index(drop=True)

    # Save stock-level scores
    stock_scores.to_csv(output_file, index=False)

    # Select final columns for stock-specific sentiment
    output_columns = [
        'ticker', 'primary_stock', 'date', 'subreddit', 'post_count', 'close_price',
        '50_day_MA', '200_day_MA', 'sector', 'geography', 'sentiment_label', 'sentiment_confidence',
        'net_sentiment', 'trading_signal', 'risk_summary', 'text_summary', 'events'
    ]
    df_output = df[output_columns]

    # Save main results
    df_output.to_csv('stock_specific_sentiment.csv', index=False)

    # Create stock-risk summary
    risk_records = []
    for _, row in df.iterrows():
        for risk in row['stock_risk_factors']:
            risk_records.append({
                'ticker': row['primary_stock'],
                'risk_type': risk['risk_type'],
                'sentiment': risk['sentiment'],
                'confidence': risk['confidence'],
                'date': row['date'],
                'sector': row['sector'],
                'geography': row['geography'],
                'sector_relevant': risk['sector_relevant']
            })
    risk_df = pd.DataFrame(risk_records)
    if not risk_df.empty:
        risk_summary = risk_df.groupby(['ticker', 'risk_type', 'geography']).agg(
            dominant_sentiment=('sentiment', lambda x: x.mode()[0] if not x.empty else 'neutral'),
            average_confidence=('confidence', 'mean'),
            occurrences=('risk_type', 'count'),
            sector_relevance=('sector_relevant', 'mean')
        ).reset_index()
    else:
        risk_summary = pd.DataFrame(columns=['ticker', 'risk_type', 'geography', 'dominant_sentiment', 
                                             'average_confidence', 'occurrences', 'sector_relevance'])

    # Save reports
    risk_summary.to_csv('stock_risk_sentiment_report.csv', index=False)
    sector_summary_df.to_csv('sector_sentiment_report.csv', index=False)

    logging.info("Sentiment analysis complete!")
    logging.info(f"Valid tickers processed: {list(unique_tickers)}")
    logging.info(f"Risks identified: {len(risk_summary)}")
    logging.info(f"Sectors covered: {len(sector_summary_df)}")
    logging.info(f"Stocks scored: {len(stock_scores)}")
    logging.info("Output saved to 'stock_specific_sentiment.csv', '%s', "
                "'stock_risk_sentiment_report.csv', and 'sector_sentiment_report.csv'.", output_file)

if __name__ == "__main__":
    main()