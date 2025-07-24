import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import subprocess
import os

# Set page configuration
st.set_page_config(page_title="Stock Sentiment Dashboard", layout="wide")

# Function to load data with error handling
@st.cache_data
def load_data(file_path, file_type='csv'):
    try:
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'parquet':
            return pd.read_parquet(file_path)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

# Function to run the pipeline
def run_pipeline(days, geography):
    try:
        cmd = [
            "python", "main.py",
            "--ticker-file", "nas.csv",
            "--output-file", "stock_final_scores.csv",  # Use output-file as expected by main.py
            "--geography", geography,  # Pass geography directly
            "--days", str(days)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        st.success("Pipeline executed successfully!")
        st.text(result.stdout)
        if result.stderr:
            st.warning(f"Pipeline warnings:\n{result.stderr}")
        # Clear cache to reload data
        st.cache_data.clear()
    except subprocess.CalledProcessError as e:
        st.error(f"Pipeline failed:\n{e.stderr}")
        return False
    except Exception as e:
        st.error(f"Error running pipeline: {str(e)}")
        return False
    return True

# Load data files
scores_df = load_data('stock_final_scores.csv', 'csv')
signals_df = load_data('signal_output.parquet', 'parquet')
risk_df = load_data('stock_risk_sentiment_report.csv', 'csv')
sector_df = load_data('sector_sentiment_report.csv', 'csv')

# Check if all data is loaded
if any(df is None for df in [scores_df, signals_df, risk_df, sector_df]):
    st.error("Some data failed to load. Please check the file paths.")
    st.stop()

# Convert date columns to datetime
date_columns = [
    (scores_df, 'last_date'),
    (signals_df, 'Date'),
    (risk_df, 'date')
]

for df, date_col in date_columns:
    if df is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])

# Sidebar for filters
st.sidebar.header("Pipeline Parameters")
days = st.sidebar.number_input("Days for Analysis", min_value=1, max_value=30, value=7, step=1)
geography = st.sidebar.selectbox("Select Country", ["US", "UK", "Hong Kong", "India"], index=0)

if st.sidebar.button("Run Pipeline"):
    with st.spinner("Running pipeline..."):
        if run_pipeline(days, geography):
            # Reload data after pipeline run
            scores_df = load_data('stock_final_scores.csv', 'csv')
            signals_df = load_data('signal_output.parquet', 'parquet')
            risk_df = load_data('stock_risk_sentiment_report.csv', 'csv')
            sector_df = load_data('sector_sentiment_report.csv', 'csv')
            if any(df is None for df in [scores_df, signals_df, risk_df, sector_df]):
                st.error("Some data failed to load after pipeline run.")
                st.stop()
            # Update date columns again
            for df, date_col in date_columns:
                if df is not None and date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col])

# Sidebar: Ticker and Date Filters
st.sidebar.header("Data Filters")
ticker_col = 'primary_stock'
if ticker_col in scores_df.columns:
    tickers = sorted(scores_df[ticker_col].dropna().unique())
    default_tickers = tickers[:5] if len(tickers) > 5 else tickers
    selected_tickers = st.sidebar.multiselect("Select Tickers", tickers, default=default_tickers)
else:
    st.sidebar.error("No ticker column found in scores data")
    st.stop()

if signals_df is not None and 'Date' in signals_df.columns:
    min_date = signals_df['Date'].min().date()
    max_date = signals_df['Date'].max().date()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
else:
    st.sidebar.warning("Using default date range")
    date_range = [datetime.now().date() - pd.Timedelta(days=7), datetime.now().date()]

# Filter dataframes
def filter_df(df, date_col=None):
    if df is None:
        return None
    filtered = df.copy()
    if ticker_col in filtered.columns:
        filtered = filtered[filtered[ticker_col].isin(selected_tickers)]
    elif 'ticker' in filtered.columns:
        filtered = filtered[filtered['ticker'].isin(selected_tickers)]
    if date_col and date_col in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered[date_col].dt.date >= start_date) &
            (filtered[date_col].dt.date <= end_date)
        ]
    return filtered

filtered_scores = filter_df(scores_df, 'last_date')
filtered_signals = filter_df(signals_df, 'Date')
filtered_risk = filter_df(risk_df, 'date')
filtered_sector = filter_df(sector_df)

# Tabs for different views
tabs = st.tabs(["Home", "Stock Scores", "Signals", "Bullish", "Bearish", "Risks", "Events", "Sector Analysis"])

# Home Tab
with tabs[0]:
    st.header("Stock Sentiment Dashboard")
    st.write("This dashboard visualizes the results of the stock sentiment and signal generation pipeline.")
    if scores_df is not None:
        st.subheader("Pipeline Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Stocks Processed", len(scores_df[ticker_col].unique()))
        signal_count = 0
        if filtered_signals is not None and 'signal' in filtered_signals.columns:
            signal_count = len(filtered_signals[filtered_signals['signal'] != 'hold'])
        col2.metric("Total Signals", signal_count)
        risk_count = 0
        if filtered_risk is not None and 'risk_type' in filtered_risk.columns:
            risk_count = len(filtered_risk)
        col3.metric("Risk Factors", risk_count)

# Stock Scores Tab
with tabs[1]:
    st.header("Stock Scores")
    if filtered_scores is not None and not filtered_scores.empty:
        display_cols = [ticker_col, 'final_score', 'avg_sentiment_score', 
                       'avg_event_score', 'sector_score', 'avg_technical_score',
                       'avg_risk_penalty', 'total_post_count', 'last_date']
        display_cols = [col for col in display_cols if col in filtered_scores.columns and col != 'geography']
        st.dataframe(filtered_scores[display_cols], use_container_width=True)
        if (filtered_signals is not None and not filtered_signals.empty and 
            'final_score' in filtered_signals.columns and 
            'price_change' in filtered_signals.columns):
            valid_signals = filtered_signals.dropna(subset=['final_score', 'price_change'])
            if not valid_signals.empty:
                fig = px.scatter(
                    valid_signals,
                    x='final_score',
                    y='price_change',
                    color='signal',
                    hover_data=[ticker_col, 'confidence_score', 'Date'],
                    title="Sentiment Score vs. Price Change",
                    labels={'final_score': 'Sentiment Score', 'price_change': 'Price Change (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid data for scatter plot after removing NaN values")
        else:
            st.warning("Required columns for scatter plot not found in signals data")
    else:
        st.warning("No stock scores available for the selected filters.")

# Signals Tab
with tabs[2]:
    st.header("Trading Signals")
    if filtered_signals is not None and not filtered_signals.empty:
        signal_cols = [ticker_col, 'Date', 'signal', 'confidence_score', 
                      'position_size_percent', 'RSI', 'MACD_histogram', 'Close']
        signal_cols = [col for col in signal_cols if col in filtered_signals.columns]
        st.dataframe(filtered_signals[signal_cols], use_container_width=True)
        if 'confidence_score' in filtered_signals.columns:
            st.subheader("Top Signals")
            leaderboard = filtered_signals[filtered_signals['signal'] != 'hold']
            leaderboard = leaderboard.sort_values(by='confidence_score', ascending=False).head(10)
            st.dataframe(leaderboard[signal_cols], use_container_width=True)
        if ticker_col in filtered_signals.columns:
            st.subheader("Price Chart")
            selected_ticker = st.selectbox("Select Ticker", selected_tickers, key="price_chart_signals")
            ticker_data = filtered_signals[filtered_signals[ticker_col] == selected_ticker]
            if not ticker_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'], name='Close Price'))
                indicator_map = {
                    'SMA_20': 'SMA 20',
                    'BB_upper': 'BB Upper',
                    'BB_lower': 'BB Lower'
                }
                for col, name in indicator_map.items():
                    if col in ticker_data.columns:
                        line_style = dict(dash='dash') if 'BB' in col else {}
                        fig.add_trace(go.Scatter(
                            x=ticker_data['Date'], 
                            y=ticker_data[col], 
                            name=name,
                            line=line_style
                        ))
                fig.update_layout(title=f"{selected_ticker} Price and Indicators", 
                                xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data found for {selected_ticker}")
        else:
            st.warning("Ticker column not found in signals data")
    else:
        st.warning("No signals available for the selected filters.")

# Bullish Tab
with tabs[3]:
    st.header("Bullish Signals")
    if filtered_signals is not None and not filtered_signals.empty and 'signal' in filtered_signals.columns:
        bullish_signals = filtered_signals[filtered_signals['signal'].isin(['buy', 'strong_buy'])]
        if not bullish_signals.empty:
            signal_cols = [ticker_col, 'Date', 'signal', 'confidence_score', 
                          'position_size_percent', 'RSI', 'MACD_histogram', 'Close']
            signal_cols = [col for col in signal_cols if col in bullish_signals.columns]
            st.dataframe(bullish_signals[signal_cols], use_container_width=True)
            if 'final_score' in bullish_signals.columns and 'price_change' in bullish_signals.columns:
                valid_bullish = bullish_signals.dropna(subset=['final_score', 'price_change'])
                if not valid_bullish.empty:
                    fig = px.scatter(
                        valid_bullish,
                        x='final_score',
                        y='price_change',
                        color='confidence_score',
                        hover_data=[ticker_col, 'Date'],
                        title="Bullish Signals: Sentiment vs. Price Change",
                        labels={'final_score': 'Sentiment Score', 'price_change': 'Price Change (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid data for bullish scatter plot after removing NaN values")
            else:
                st.warning("Required columns for scatter plot not found in bullish signals")
        else:
            st.warning("No bullish signals available for the selected filters.")
    else:
        st.warning("No signals data available for bullish analysis.")

# Bearish Tab
with tabs[4]:
    st.header("Bearish Signals")
    if filtered_signals is not None and not filtered_signals.empty and 'signal' in filtered_signals.columns:
        bearish_signals = filtered_signals[filtered_signals['signal'] == 'sell']
        if not bearish_signals.empty:
            signal_cols = [ticker_col, 'Date', 'signal', 'confidence_score', 
                          'position_size_percent', 'RSI', 'MACD_histogram', 'Close']
            signal_cols = [col for col in signal_cols if col in bearish_signals.columns]
            st.dataframe(bearish_signals[signal_cols], use_container_width=True)
            if 'final_score' in bearish_signals.columns and 'price_change' in bearish_signals.columns:
                valid_bearish = bearish_signals.dropna(subset=['final_score', 'price_change'])
                if not valid_bearish.empty:
                    fig = px.scatter(
                        valid_bearish,
                        x='final_score',
                        y='price_change',
                        color='confidence_score',
                        hover_data=[ticker_col, 'Date'],
                        title="Bearish Signals: Sentiment vs. Price Change",
                        labels={'final_score': 'Sentiment Score', 'price_change': 'Price Change (%)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid data for bearish scatter plot after removing NaN values")
            else:
                st.warning("Required columns for scatter plot not found in bearish signals")
        else:
            st.warning("No bearish signals available for the selected filters.")
    else:
        st.warning("No signals data available for bearish analysis.")

# Risks Tab
with tabs[5]:
    st.header("Risk Factors")
    if filtered_risk is not None and not filtered_risk.empty:
        risk_cols = [ticker_col, 'risk_type', 'dominant_sentiment', 
                    'average_confidence', 'occurrences', 'sector_relevance']
        risk_cols = [col for col in risk_cols if col in filtered_risk.columns and col != 'geography']
        st.dataframe(filtered_risk[risk_cols], use_container_width=True)
        if 'risk_type' in filtered_risk.columns:
            st.subheader("Risk Factor Distribution")
            risk_summary = filtered_risk.groupby(['ticker', 'risk_type']).size().reset_index(name='count')
            fig = px.bar(
                risk_summary,
                x='ticker',
                y='count',
                color='risk_type',
                title="Risk Factors by Ticker",
                labels={'count': 'Number of Occurrences', 'ticker': 'Ticker'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Risk type column not found in risk data")
    else:
        st.warning("No risk factors available for the selected filters.")

# Events Tab
with tabs[6]:
    st.header("Event Summary")
    if filtered_risk is not None and not filtered_risk.empty and 'risk_type' in filtered_risk.columns:
        # Infer events from risk_df, assuming events are related to risk_type or dominant_sentiment
        event_summary = filtered_risk.groupby(['ticker', 'dominant_sentiment']).size().reset_index(name='event_count')
        event_summary = event_summary[event_summary['dominant_sentiment'].isin(
            ['earnings', 'leadership', 'product', 'guidance', 'economic', 'dividend']
        )]
        if not event_summary.empty:
            st.dataframe(event_summary, use_container_width=True)
            fig = px.bar(
                event_summary,
                x='ticker',
                y='event_count',
                color='dominant_sentiment',
                title="Events by Ticker",
                labels={'event_count': 'Number of Events', 'dominant_sentiment': 'Event Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No event-related data available for the selected filters.")
    else:
        st.warning("No risk or event data available for analysis.")

# Sector Analysis Tab
with tabs[7]:
    st.header("Sector Analysis")
    if filtered_sector is not None and not filtered_sector.empty:
        sector_cols = [col for col in filtered_sector.columns if col != 'geography']
        st.dataframe(filtered_sector[sector_cols], use_container_width=True)
        if 'sector' in filtered_sector.columns:
            metrics = [col for col in ['avg_sentiment', 'fear_score', 'greed_score'] 
                      if col in filtered_sector.columns]
            if metrics:
                fig = px.bar(
                    filtered_sector,
                    x='sector',
                    y=metrics,
                    barmode='group',
                    title="Sector Sentiment Metrics",
                    labels={'value': 'Score', 'variable': 'Metric'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No sentiment metrics found for sector analysis")
        else:
            st.warning("Sector column not found in sector data")
    else:
        st.warning("No sector data available for the selected filters.")

# Footer
st.write("Built with Streamlit | Data from Stock Sentiment Pipeline | Last updated: July 18, 2025, 2:40 PM IST")