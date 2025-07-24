# Stock Sentiment Pipeline: End-to-End Documentation

## Overview

This project is a full-featured pipeline for stock sentiment analysis and signal generation using Reddit posts, stock price data, and state-of-the-art NLP techniques. It consists of several core components:

- **Data Extraction** (`extracter.py`): Scrapes Reddit for posts related to stocks, validates tickers, and fetches historical price data.
- **Sentiment Analysis** (`sentiment.py`): Uses FinBERT to analyze the sentiment of posts and extract events/risk factors.
- **Signal Generation & Backtesting** (`signal_generator.py`): Generates trading signals, calculates technical indicators, and runs a backtest with performance and alpha attribution.
- **Dashboard** (`dashboard.py`): Streamlit-based app to visualize scores, signals, risks, sector analysis, and more.
- **Pipeline Orchestration** (`main.py`): CLI orchestration of the pipeline stages.

This documentation covers setup, usage, code structure, and extensibility.

---

## Table of Contents

1. [Setup & Requirements](#setup--requirements)
2. [Pipeline Structure](#pipeline-structure)
3. [Component Details](#component-details)
    - [1. Data Extraction (`extracter.py`)](#1-data-extraction-extracterpy)
    - [2. Sentiment Analysis (`sentiment.py`)](#2-sentiment-analysis-sentimentpy)
    - [3. Signal Generation & Backtesting (`signal_generator.py`)](#3-signal-generation--backtesting-signal_generatorpy)
    - [4. Pipeline Orchestration (`main.py`)](#4-pipeline-orchestration-mainpy)
    - [5. Streamlit Dashboard (`dashboard.py`)](#5-streamlit-dashboard-dashboardpy)
4. [File Inputs & Outputs](#file-inputs--outputs)
5. [Customization & Extensibility](#customization--extensibility)
6. [Troubleshooting & Logs](#troubleshooting--logs)

---

## Setup & Requirements

### Python Packages

- `praw`, `yfinance`, `pandas`, `numpy`, `plotly`, `streamlit`, `torch`, `transformers`, `nltk`, `fuzzywuzzy`, `joblib`, `backtrader`, `scipy`
- Install via: `pip install -r requirements.txt` or manually as needed.

### Data Files

- **nas.csv**: List of valid stock tickers (with 'Symbol' column).
- The pipeline will produce multiple CSV/Parquet files as outputs.

### API Keys

- **Reddit**: Edit `extracter.py` with your own Reddit API credentials.
- **HuggingFace**: Set up your token for FinBERT access.

### NLTK Setup

- Downloads required NLTK corpora automatically; ensure internet access on first run.

---

## Pipeline Structure

1. **main.py**: Orchestrates the pipeline. Accepts CLI arguments for ticker file, output file, geography, and days.
2. **extracter.py**: Scrapes and processes Reddit and stock price data.
3. **sentiment.py**: Performs NLP sentiment analysis, event/risk extraction, aggregates scores.
4. **signal_generator.py**: Calculates technicals, generates signals, runs backtest, and creates alpha reports.
5. **dashboard.py**: Visualizes all outputs and allows pipeline reruns from the UI.

---

## Component Details

### 1. Data Extraction (`extracter.py`)

**Purpose**: Collects Reddit posts mentioning stocks, validates tickers, and fetches price history.

**Workflow**:
- Maps selected geography (US/UK/HK/India) to subreddits.
- Loads tickers from `nas.csv`.
- For each subreddit:
    - Fetches latest posts within specified days.
    - Extracts tickers mentioned in posts, filters out non-stock uppercase words.
    - Validates tickers via `yfinance`.
- For each confirmed ticker:
    - Fetches historical price data and computes moving averages.
- Aggregates and merges Reddit/post and price data.
- Output: **`stock_data_for_valid_tickers.csv`**

**Key Functions**:
- `load_ticker_list()`
- `extract_potential_tickers(text, valid_tickers_set)`
- `validate_ticker(ticker)`
- `collect_data(days, ticker_file, output_file, geography)`
- Logging to `extracter.log`

---

### 2. Sentiment Analysis (`sentiment.py`)

**Purpose**: Analyzes Reddit texts for stock sentiment using FinBERT, identifies events/risk factors, and scores stocks.

**Workflow**:
- Loads data from CSV.
- For each row/text:
    - Cleans and preprocesses text.
    - Uses FinBERT for sentiment (`positive`, `neutral`, `negative`) and confidence.
    - Detects financial events (earnings, product, merger, etc.).
    - Extracts risk factors (regulatory, financial, market, operational).
    - Infers sector (via yfinance or keyword analysis).
    - Calculates component scores (sentiment, event, technical, risk).
    - Aggregates scores per stock.
    - Generates trading signal (`buy`, `sell`, `hold`, etc.).
- Outputs:
    - **`stock_final_scores.csv`**: Stock-level scores.
    - **`stock_specific_sentiment.csv`**: Row-level sentiment details.
    - **`stock_risk_sentiment_report.csv`**: Risk summary per stock.
    - **`sector_sentiment_report.csv`**: Sector summary.

**Key Functions**:
- `analyze_sentiment(text)`
- `detect_stock_specific_events(text, ticker, sector)`
- `extract_stock_risk_factors(text, ticker, sector)`
- `calculate_component_scores(row)`
- `aggregate_stock_scores(group, sector_summary)`
- Logging to `sentiment.log`

---

### 3. Signal Generation & Backtesting (`signal_generator.py`)

**Purpose**: Generates trading signals based on sentiment and technicals, runs a backtest, and analyzes performance.

**Workflow**:
- Loads `stock_final_scores.csv`.
- Validates tickers.
- Computes confidence scores and position sizes.
- Generates mock price/volume data (for demo/backtest).
- Merges sentiment and price data, computes price/volume changes.
- Signal generation via rules (strong_buy, buy, sell, strong_sell, hold).
- Runs `backtrader` simulation strategy:
    - Executes trades based on signals and position sizes.
    - Tracks portfolio value and trades.
- Calculates performance metrics (Sharpe, alpha, beta, win rate).
- Generates alpha attribution report.
- Outputs:
    - **`signal_output.parquet`**: All signals and technicals.
    - **`alpha_generation_report.csv`**
- Visualizes results with `plotly`.

**Key Functions**:
- `compute_overall_confidence(row)`
- `generate_signal(df)`
- `SentimentStrategy` (Backtrader strategy)
- `calculate_performance_metrics(returns_df, trades_df, benchmark_df)`
- `generate_alpha_report(trades_df, merged_df, benchmark_df)`

---

### 4. Pipeline Orchestration (`main.py`)

**Purpose**: CLI entrypoint that coordinates all pipeline steps.

**Workflow**:
- Parses arguments for ticker file, output file, geography, days.
- Sets up config dictionary.
- Runs extraction, sentiment, and signal generation in order.
- Prints completion status and output file locations.
- Logging to `pipeline.log`.

---

### 5. Streamlit Dashboard (`dashboard.py`)

**Purpose**: Interactive visualization of pipeline outputs.

**Features**:
- Sidebar: Pipeline parameters (days, geography), run button, ticker/date filters.
- Tabs:
    - **Home**: Pipeline summary metrics.
    - **Stock Scores**: Table and scatter plot (sentiment vs price change).
    - **Signals**: Signal table, top signals, price chart with indicators.
    - **Bullish/Bearish**: Signal breakdowns and scatter plots.
    - **Risks**: Risk table and distribution by ticker.
    - **Events**: Event summary and bar chart.
    - **Sector Analysis**: Sector metrics and bar charts.
- Auto-reloads data after pipeline run.
- Error handling for missing files.
- Footer shows last updated time.

---

## File Inputs & Outputs

| File                          | Description                             | Produced By         |
|-------------------------------|-----------------------------------------|---------------------|
| `nas.csv`                     | Stock ticker list (input)               | user/download       |
| `stock_data_for_valid_tickers.csv` | Raw Reddit & price data                 | extracter.py        |
| `stock_final_scores.csv`      | Aggregated stock scores                 | sentiment.py        |
| `stock_specific_sentiment.csv`| Row-level sentiment analysis            | sentiment.py        |
| `stock_risk_sentiment_report.csv` | Risk summary per stock                   | sentiment.py        |
| `sector_sentiment_report.csv` | Sector-level summary                    | sentiment.py        |
| `signal_output.parquet`       | Signals with technicals                 | signal_generator.py |
| `alpha_generation_report.csv` | Alpha attribution per ticker            | signal_generator.py |
| `pipeline.log`, `extracter.log`, `sentiment.log` | Logs | respective stages |

---

## Customization & Extensibility

- **Add Geographies/Subreddits**: Edit `geography_subreddits` in `extracter.py`.
- **Modify Sentiment/Event Detection**: Update keyword dictionaries in `sentiment.py`.
- **Change Trading Logic**: Edit signal generation rules and backtrader strategy in `signal_generator.py`.
- **Dashboard Customization**: Extend tabs/visuals in `dashboard.py`.

---

## Troubleshooting & Logs

- All major scripts log errors and info to `.log` files.
- Check logs for detailed error traces.
- Common issues:
    - Missing or misnamed columns in input files (`nas.csv` must have 'Symbol').
    - API credential issues (Reddit, HuggingFace).
    - Missing NLTK corpora (auto-downloads).
    - Dependency version mismatches (check torch, transformers).
- For dashboard errors, check Streamlit UI and logs.

---

## Example CLI Usage

```bash
python main.py --ticker-file nas.csv --output-file stock_final_scores.csv --geography US --days 7
```

## Example Dashboard Usage

1. Run: `streamlit run dashboard.py`
2. Set parameters, click "Run Pipeline", explore tabs.

---

## Contact & Contribution

- For issues, use the project GitHub Issues.
- Contributions welcome! Please document any new components or features.
