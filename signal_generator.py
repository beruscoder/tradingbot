import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
import backtrader as bt
import warnings
import json
import os
from datetime import datetime, timedelta
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')

# --- Ticker Validation ---
def validate_tickers(tickers, cache_file="ticker_corrections_cache.json"):
    ticker_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            ticker_cache = json.load(f)
    
    valid_tickers = []
    for ticker in tickers:
        if ticker in ticker_cache and ticker_cache[ticker]:
            valid_tickers.append(ticker)
            continue
        ticker_cache[ticker] = f"Mock {ticker}"
        valid_tickers.append(ticker)
    
    with open(cache_file, 'w') as f:
        json.dump(ticker_cache, f, indent=4)
    
    return valid_tickers

# --- Optimized Mock Stock History ---
def generate_mock_stock_history(ticker, dates, base_price=100, volatility=0.02, days_range=7):
    np.random.seed(42)
    all_dates = sorted(list(set([pd.to_datetime(date) + timedelta(days=x) for date in dates for x in range(-days_range, days_range + 1)])))
    
    df = pd.DataFrame({'Date': all_dates, 'ticker': ticker})
    final_score = np.round(summary_df[summary_df['primary_stock'] == ticker]['final_score'].mean() if 'summary_df' in globals() else 50, 2)
    trend = np.round(0.0015 * (final_score - 50) / 50, 2)
    df['Close'] = np.round(np.float32(base_price * (1 + np.random.normal(trend, volatility, len(all_dates)).cumsum())), 2)
    df['Volume'] = np.int32(np.random.randint(1000000, 5000000, len(all_dates)))
    df['open'] = np.round(np.float32(df['Close'] * (1 - np.random.uniform(0.01, 0.03, len(all_dates)))), 2)
    df['high'] = np.round(np.float32(df['Close'] * (1 + np.random.uniform(0.01, 0.03, len(all_dates)))), 2)
    df['low'] = np.round(np.float32(df['Close'] * (1 - np.random.uniform(0.01, 0.03, len(all_dates)))), 2)
    
    # Technical indicators
    df['SMA_20'] = np.round(np.float32(df['Close'].rolling(window=20).mean()), 2)
    df['SMA_30'] = np.round(np.float32(df['Close'].rolling(window=30).mean()), 2)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    sentiment_factor = np.round((final_score - 50) / 50 if final_score != 50 else 0, 2)
    df['RSI'] = np.round(np.float32((100 - (100 / (1 + rs)) + sentiment_factor * 10).clip(0, 100)), 2)
    ema_short = df['Close'].ewm(span=12, adjust=False).mean()
    ema_long = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = np.round(np.float32(ema_short - ema_long + sentiment_factor * 0.3), 2)
    df['MACD_signal'] = np.round(np.float32(df['MACD'].ewm(span=9, adjust=False).mean()), 2)
    df['MACD_histogram'] = np.round(np.float32(df['MACD'] - df['MACD_signal']), 2)
    df['BB_mid'] = np.round(np.float32(df['Close'].rolling(window=20).mean()), 2)
    df['BB_std'] = np.round(np.float32(df['Close'].rolling(window=20).std()), 2)
    df['BB_upper'] = np.round(np.float32(df['BB_mid'] + 2 * df['BB_std']), 2)
    df['BB_lower'] = np.round(np.float32(df['BB_mid'] - 2 * df['BB_std']), 2)
    df['tr'] = np.round(np.float32(df[['high', 'Close', 'low']].max(axis=1) - df[['high', 'Close', 'low']].min(axis=1)), 2)
    df['atr'] = np.round(np.float32(df['tr'].rolling(window=14).mean()), 2)
    return df[['Date', 'ticker', 'open', 'high', 'low', 'Close', 'Volume', 'SMA_20', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'BB_upper', 'BB_lower', 'atr']]

# --- Dynamic Sentiment Weighting ---
def compute_overall_confidence(row, recency_weight=0.4, sentiment_weight=0.4, event_weight=0.1, volume_weight=0.1):
    try:
        base_conf = row.get('avg_sentiment_score', 0) / 100
        event_bonus = min(1.0, row.get('total_events', 0) / 10.0)
        volume_bonus = min(1.0, row.get('total_post_count', 0) / 20.0)
        days_old = (pd.to_datetime('2025-07-14') - pd.to_datetime(row.get('last_date'))).days
        recency_factor = np.round(np.exp(-0.1 * max(0, days_old)), 2)
        return np.round(np.float32(recency_weight * recency_factor + sentiment_weight * base_conf +
                                  event_weight * event_bonus + volume_weight * volume_bonus), 2)
    except Exception as e:
        print(f"Error computing confidence for row: {e}")
        return np.float32(0.0)

# --- Risk-Adjusted Position Sizing ---
def calculate_position_size(row, max_allocation_percent=10.0):
    confidence = row['confidence_score'] * 100
    risk_penalty = row.get('avg_risk_penalty', 0)
    volatility_adjustment = np.round(min(1.0, 10.0 / (row['atr'] * 100)), 2) if 'atr' in row and not pd.isna(row['atr']) else 1.0
    raw_score = (confidence - risk_penalty) * volatility_adjustment
    return np.round(np.float32(max(0, min(raw_score, max_allocation_percent))), 2)

# --- Vectorized Signal Generation ---
def generate_signal(df):
    conditions = [
        (df['final_score'] > 40) & (df['confidence_score'] > 0.6) & (df['RSI'] < 75) & (df['MACD_histogram'] > 0),
        (df['final_score'] > 10) & (df['confidence_score'] > 0.4) & (df['RSI'] < 75),
        (df['final_score'] < -40) & (df['confidence_score'] > 0.6) & (df['RSI'] > 25) & (df['MACD_histogram'] < 0),
        (df['final_score'] < -10) & (df['confidence_score'] > 0.4) & (df['RSI'] > 25)
    ]
    choices = ['strong_buy', 'buy', 'strong_sell', 'sell']
    return np.select(conditions, choices, default='hold')

# --- Backtrader Strategy ---
class SentimentStrategy(bt.Strategy):
    params = (('signals', None), ('max_allocation', 0.1), ('commission', 0.001))

    def __init__(self):
        self.signals = self.params.signals.set_index(['Date', 'ticker'])
        self.returns = []
        self.trades = []

    def next(self):
        date = self.datas[0].datetime.date(0)
        for data in self.datas:
            ticker = data._name
            signal_key = (pd.Timestamp(date), ticker)
            if signal_key in self.signals.index:
                signal = self.signals.loc[signal_key, 'signal']
                position_size = self.signals.loc[signal_key, 'position_size_percent'] / 100
                if signal in ['buy', 'strong_buy'] and not self.getposition(data).size:
                    size = (self.broker.getvalue() * position_size) // data.close[0]
                    if size > 0:
                        self.buy(data=data, size=size)
                elif signal in ['sell', 'strong_sell'] and self.getposition(data).size:
                    self.close(data=data)
        self.returns.append((date, self.broker.getvalue()))

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'date': pd.Timestamp(trade.dtclose, unit='D', origin='1970-01-01'),
                'ticker': trade.data._name,
                'trade_profit': np.round(trade.pnlcomm, 2)
            })

# --- Performance Metrics ---
def calculate_performance_metrics(returns_df, trades_df, benchmark_df, risk_free_rate=0.02):
    returns_df['daily_return'] = np.round(returns_df['portfolio_value'].pct_change().fillna(0), 2)
    benchmark_df['daily_return'] = np.round(benchmark_df['Close'].pct_change().fillna(0), 2)
    avg_return = np.round(returns_df['daily_return'].mean() * 252, 2)
    std_return = np.round(returns_df['daily_return'].std() * np.sqrt(252), 2)
    sharpe_ratio = np.round((avg_return - risk_free_rate) / std_return, 2) if std_return != 0 else np.nan
    cov = np.round(returns_df['daily_return'].cov(benchmark_df['daily_return']) * 252, 2)
    var_benchmark = np.round(benchmark_df['daily_return'].var() * 252, 2)
    beta = np.round(cov / var_benchmark, 2) if var_benchmark != 0 else np.nan
    benchmark_return = np.round(benchmark_df['daily_return'].mean() * 252, 2)
    alpha = np.round(avg_return - risk_free_rate - beta * (benchmark_return - risk_free_rate), 2)
    win_rate = np.round(len(trades_df[trades_df['trade_profit'] > 0]) / len(trades_df), 2) if len(trades_df) > 0 else np.nan
    return {
        'sharpe_ratio': sharpe_ratio,
        'alpha': alpha,
        'win_rate': win_rate,
        'beta': beta,
        'portfolio_return': avg_return,
        'benchmark_return': benchmark_return
    }

# --- Alpha Generation Report ---
def generate_alpha_report(trades_df, merged_df, benchmark_df, risk_free_rate=0.02):
    if trades_df.empty or merged_df.empty or benchmark_df.empty:
        print("Warning: Empty trades_df, merged_df, or benchmark_df. Skipping alpha report.")
        return pd.DataFrame()

    # Calculate daily portfolio and benchmark returns
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
    trades_df['date'] = pd.to_datetime(trades_df['date'])

    # Merge trade data with sentiment and technical data
    alpha_df = trades_df.merge(
        merged_df[['Date', 'ticker', 'final_score', 'confidence_score', 'RSI', 'MACD_histogram', 'avg_risk_penalty', 'signal']],
        left_on=['date', 'ticker'],
        right_on=['Date', 'ticker'],
        how='left'
    )

    # Calculate trade-level alpha (trade profit adjusted for benchmark return)
    benchmark_returns = benchmark_df.set_index('Date')['Close'].pct_change().fillna(0)
    alpha_df['benchmark_return'] = alpha_df['date'].map(benchmark_returns)
    alpha_df['trade_alpha'] = np.round(alpha_df['trade_profit'] - alpha_df['benchmark_return'] * alpha_df['trade_profit'], 2)

    # Attribute alpha contributions
    alpha_df['sentiment_contribution'] = np.round(alpha_df['final_score'] * alpha_df['confidence_score'] * 0.01, 2)
    alpha_df['technical_contribution'] = np.round((alpha_df['RSI'].clip(0, 100) / 100 + alpha_df['MACD_histogram']) * 0.5, 2)
    alpha_df['risk_adjustment'] = np.round(-alpha_df['avg_risk_penalty'] * 0.01, 2)

    # Aggregate by ticker
    ticker_alpha = alpha_df.groupby('ticker').agg({
        'trade_alpha': 'sum',
        'sentiment_contribution': 'mean',
        'technical_contribution': 'mean',
        'risk_adjustment': 'mean',
        'signal': lambda x: x.mode()[0] if not x.empty else 'hold',
        'trade_profit': 'count'
    }).rename(columns={'trade_profit': 'trade_count'}).reset_index()

    # Calculate correlations
    corr_data = alpha_df[['trade_alpha', 'final_score', 'confidence_score', 'RSI', 'MACD_histogram', 'avg_risk_penalty']].dropna()
    correlations = {}
    for metric in ['final_score', 'confidence_score', 'RSI', 'MACD_histogram', 'avg_risk_penalty']:
        if len(corr_data) > 1:
            corr, _ = pearsonr(corr_data['trade_alpha'], corr_data[metric])
            correlations[metric] = np.round(corr, 2)
        else:
            correlations[metric] = np.nan

    # Add correlations to report
    ticker_alpha['alpha_correlation_sentiment'] = correlations.get('final_score', np.nan)
    ticker_alpha['alpha_correlation_confidence'] = correlations.get('confidence_score', np.nan)
    ticker_alpha['alpha_correlation_RSI'] = correlations.get('RSI', np.nan)
    ticker_alpha['alpha_correlation_MACD'] = correlations.get('MACD_histogram', np.nan)
    ticker_alpha['alpha_correlation_risk'] = correlations.get('avg_risk_penalty', np.nan)

    # Sort by total alpha
    ticker_alpha = ticker_alpha.sort_values('trade_alpha', ascending=False)

    # Save report
    ticker_alpha.to_csv('alpha_generation_report.csv', index=False)

    # Visualize alpha contributions
    if not ticker_alpha.empty:
        fig = px.bar(
            ticker_alpha.head(10),
            x='ticker',
            y=['trade_alpha', 'sentiment_contribution', 'technical_contribution', 'risk_adjustment'],
            title="Top 10 Tickers by Alpha Contribution",
            labels={'value': 'Contribution', 'variable': 'Component'},
            barmode='group'
        )
        fig.update_layout(xaxis_title="Ticker", yaxis_title="Alpha Contribution", showlegend=True)
        fig.update_yaxes(tickformat=".2f")
        fig.show()

    return ticker_alpha

# --- Main Execution ---
def main():
    global summary_df
    try:
        summary_df = pd.read_csv("stock_final_scores.csv")
        numeric_cols = ['final_score', 'avg_sentiment_score', 'avg_event_score', 'sector_score', 'avg_technical_score', 'avg_risk_penalty']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
        required_columns = ['primary_stock', 'last_date', 'avg_sentiment_score', 'total_events', 'total_post_count', 'final_score']
        missing_cols = [col for col in required_columns if col not in summary_df.columns]
        if missing_cols:
            print(f"Error: Missing columns in stock_final_scores.csv: {missing_cols}")
            return
    except FileNotFoundError:
        print("Error: stock_final_scores.csv not found")
        return

    # Validate tickers
    tickers = summary_df['primary_stock'].unique()
    valid_tickers = validate_tickers(tickers)
    if not valid_tickers:
        print("Error: No valid tickers found")
        return
    summary_df = summary_df[summary_df['primary_stock'].isin(valid_tickers)].copy()
    summary_df['primary_stock'] = summary_df['primary_stock'].astype('category')
    print(f"Valid tickers: {valid_tickers}")
    print(f"Filtered summary_df shape: {summary_df.shape}")
    print("Sample summary_df:\n", summary_df[['primary_stock', 'last_date', 'final_score', 'avg_sentiment_score']].head())

    # Compute confidence and position sizes
    summary_df['confidence_score'] = summary_df.apply(compute_overall_confidence, axis=1)
    summary_df['position_size_percent'] = summary_df.apply(calculate_position_size, axis=1)

    # Generate mock stock data
    unique_dates = pd.to_datetime(summary_df['last_date']).unique()
    price_volume_dfs = Parallel(n_jobs=-1)(delayed(generate_mock_stock_history)(ticker, unique_dates) for ticker in valid_tickers)
    price_volume_df = pd.concat(price_volume_dfs, ignore_index=True)
    price_volume_df['ticker'] = price_volume_df['ticker'].astype('category')
    price_volume_df['Date'] = pd.to_datetime(price_volume_df['Date'])
    if price_volume_df.empty:
        print("Error: No mock price data generated")
        return
    print(f"price_volume_df shape: {price_volume_df.shape}")
    print("Sample price_volume_df:\n", price_volume_df[['ticker', 'Date', 'Close']].head())

    # Merge data
    summary_df['date'] = pd.to_datetime(summary_df['last_date'])
    merged_df = pd.merge(
        summary_df.sort_values('primary_stock'),
        price_volume_df.sort_values('ticker'),
        left_on='primary_stock',
        right_on='ticker',
        how='inner'
    )
    print(f"merged_df shape: {merged_df.shape}")
    if merged_df.empty:
        print("Error: Merged dataframe is empty")
        print("Sample summary_df:\n", summary_df[['primary_stock', 'date', 'final_score']].head())
        print("Sample price_volume_df:\n", price_volume_df[['ticker', 'Date', 'Close']].head())
        return

    # Calculate price/volume changes
    merged_df['price_change'] = np.round(np.float32(merged_df.groupby('ticker')['Close'].pct_change() * 100), 2)
    merged_df['volume_change'] = np.round(np.float32(merged_df.groupby('ticker')['Volume'].pct_change() * 100), 2)
    merged_df['price_change_lag1'] = np.round(np.float32(merged_df.groupby('ticker')['Close'].pct_change().shift(1) * 100), 2)
    merged_df['price_change_lag3'] = np.round(np.float32(merged_df.groupby('ticker')['Close'].pct_change().shift(3) * 100), 2)

    # Generate signals
    merged_df['signal'] = generate_signal(merged_df)
    # Round all numerical columns
    numeric_cols = ['final_score', 'avg_sentiment_score', 'avg_event_score', 'sector_score', 'avg_technical_score', 
                    'avg_risk_penalty', 'confidence_score', 'position_size_percent', 'Close', 'open', 'high', 'low', 
                    'SMA_20', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'BB_upper', 'BB_lower', 'atr', 
                    'price_change', 'volume_change', 'price_change_lag1', 'price_change_lag3']
    merged_df[numeric_cols] = merged_df[numeric_cols].round(2)
    print("merged_df sample after changes:\n", merged_df[['ticker', 'Date', 'final_score', 'confidence_score', 'RSI', 'MACD_histogram', 'price_change', 'volume_change', 'signal']].head())
    print("Signal distribution:\n", merged_df['signal'].value_counts())
    print("NaN counts in merged_df:\n", merged_df[['price_change', 'volume_change', 'price_change_lag1', 'price_change_lag3']].isna().sum())
    print("Signal condition check (first 5 rows):\n", merged_df[['ticker', 'final_score', 'confidence_score', 'RSI', 'MACD_histogram', 'signal']].head())

    # Calculate correlations
    corr_data = merged_df[['final_score', 'price_change', 'volume_change', 'price_change_lag1', 'price_change_lag3']].dropna()
    correlations = {}
    for metric in ['price_change', 'volume_change', 'price_change_lag1', 'price_change_lag3']:
        print(f"Valid data for {metric} correlation: {len(corr_data)} rows")
        correlations[metric] = np.round(corr_data['final_score'].corr(corr_data[metric]), 2) if len(corr_data) > 1 else np.nan
    print("Correlations:")
    for metric, corr in correlations.items():
        print(f"  {metric}: {corr}")

    # Backtrader Setup
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)

    # Add benchmark data
    sp500 = generate_mock_stock_history('^GSPC', unique_dates)
    if not sp500.empty:
        sp500_data = bt.feeds.PandasData(dataname=sp500, datetime='Date')
        cerebro.adddata(sp500_data, name='^GSPC')

    # Add stock data for non-hold signals
    active_tickers = merged_df[merged_df['signal'].isin(['buy', 'strong_buy', 'sell', 'strong_sell'])]['ticker'].unique()
    for ticker in active_tickers:
        stock_data = price_volume_df[price_volume_df['ticker'] == ticker]
        if not stock_data.empty:
            data = bt.feeds.PandasData(dataname=stock_data, datetime='Date')
            cerebro.adddata(data, name=ticker)

    # Add strategy
    cerebro.addstrategy(SentimentStrategy, signals=merged_df[['Date', 'ticker', 'signal', 'position_size_percent']])

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    # Extract performance metrics
    returns_df = pd.DataFrame(strategy.returns, columns=['Date', 'portfolio_value'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    trades_df = pd.DataFrame(strategy.trades)
    print(f"trades_df shape: {trades_df.shape}")

    # Calculate performance metrics
    benchmark_df = sp500[['Date', 'Close']] if not sp500.empty else pd.DataFrame()
    if not returns_df.empty and not benchmark_df.empty:
        performance_metrics = calculate_performance_metrics(returns_df, trades_df, benchmark_df)
        print("\nPerformance Metrics:")
        print(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']}")
        print(f"  Alpha: {performance_metrics['alpha']}")
        print(f"  Beta: {performance_metrics['beta']}")
        print(f"  Portfolio Return: {performance_metrics['portfolio_return']}")
        print(f"  Benchmark Return: {performance_metrics['benchmark_return']}")
        print(f"  Win Rate: {performance_metrics['win_rate']}")

    # Generate alpha report
    alpha_report = generate_alpha_report(trades_df, merged_df, benchmark_df)
    if not alpha_report.empty:
        print("\nAlpha Generation Report:")
        print(alpha_report[['ticker', 'trade_alpha', 'sentiment_contribution', 'technical_contribution', 'risk_adjustment', 'trade_count']].head(10))

    # Signal Validation
    if not trades_df.empty:
        print("\nSignal Validation:")
        print(f"  Total Trades: {len(trades_df)}")
        print(f"  Sample Trades:\n{trades_df.round(2).head()}")

    # Visualizations (downsampled)
    if not merged_df.empty and len(merged_df) > 1:
        plot_df = merged_df.groupby('ticker').head(15).round(2).copy()
        fig1 = px.scatter(plot_df, x='final_score', y='price_change', color='signal',
                          hover_data=['ticker'], title="Sentiment vs Price Change",
                          hover_name='ticker', text='ticker')
        fig1.update_traces(textposition='top center', hovertemplate='Ticker: %{hovertext}<br>Sentiment: %{x:.2f}<br>Price Change: %{y:.2f}%')
        fig1.update_layout(xaxis_title="Sentiment Score", yaxis_title="Price Change (%)", showlegend=True)
        fig1.update_xaxes(tickformat=".2f")
        fig1.update_yaxes(tickformat=".2f")
        fig1.show()

        fig2 = px.scatter(plot_df, x='final_score', y='volume_change', color='signal',
                          hover_data=['ticker'], title="Sentiment vs Volume Change",
                          hover_name='ticker', text='ticker')
        fig2.update_traces(textposition='top center', hovertemplate='Ticker: %{hovertext}<br>Sentiment: %{x:.2f}<br>Volume Change: %{y:.2f}%')
        fig2.update_layout(xaxis_title="Sentiment Score", yaxis_title="Volume Change (%)", showlegend=True)
        fig2.update_xaxes(tickformat=".2f")
        fig2.update_yaxes(tickformat=".2f")
        fig2.show()

        sample_ticker = valid_tickers[0] if valid_tickers else None
        if sample_ticker:
            ticker_df = merged_df[merged_df['ticker'] == sample_ticker].round(2)
            if not ticker_df.empty:
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['Close'], name='Close Price'))
                fig3.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['SMA_20'], name='SMA 20'))
                fig3.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_upper'], name='BB Upper', line=dict(dash='dash')))
                fig3.add_trace(go.Scatter(x=ticker_df['Date'], y=ticker_df['BB_lower'], name='BB Lower', line=dict(dash='dash')))
                fig3.update_layout(title=f"{sample_ticker} Price and Indicators", xaxis_title="Date", yaxis_title="Price")
                fig3.update_yaxes(tickformat=".2f")
                fig3.show()

    # Save results as Parquet
    merged_df = merged_df.round(2)
    merged_df.to_parquet("signal_output.parquet", index=False)

    # Print leaderboard
    if not merged_df.empty:
        leaderboard = merged_df.sort_values(by='position_size_percent', ascending=False)
        print("\nTop Signal Leaderboard:")
        print(leaderboard[['ticker', 'signal', 'confidence_score', 'position_size_percent', 'RSI', 'MACD_histogram']].head(10))

if __name__ == "__main__":
    main()