import sys
import logging
import argparse
from datetime import datetime
import extracter
import sentiment
import signal_generator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_pipeline(config):
    """
    Orchestrates the execution of the data extraction, sentiment analysis, and signal generation pipeline.
    
    Args:
        config (dict): Configuration dictionary with file paths and parameters.
    
    Returns:
        bool: True if pipeline completes successfully, False otherwise.
    """
    try:
        logging.info("Starting pipeline execution")
        
        # Step 1: Run data extraction
        logging.info("Running extracter.py")
        extracter.main(
            days=config['days_lookback'],
            ticker_file=config['ticker_list_file'],
            output_file=config['extracted_data_file'],
            geography=config['geography']
        )
        logging.info(f"Data extraction complete. Output saved to {config['extracted_data_file']}")
        
        # Step 2: Run sentiment analysis
        logging.info("Running sentiment.py")
        sentiment.main(
            input_file=config['extracted_data_file'],
            output_file=config['sentiment_output_files'][0]
        )
        logging.info(f"Sentiment analysis complete. Outputs saved to {config['sentiment_output_files']}")
        
        # Step 3: Run signal generation
        logging.info("Running signal_generator.py")
        signal_generator.main()
        logging.info(f"Signal generation complete. Output saved to {config['signal_output_file']}")
        
        logging.info("Pipeline execution completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        return False

def main(ticker_file="nas.csv", output_file="stock_final_scores.csv", geography="US", days=7):
    # Define default configuration
    config = {
        'extracted_data_file': 'stock_data_for_valid_tickers.csv',
        'sentiment_output_files': [
            output_file,
            'stock_risk_sentiment_report.csv',
            'sector_sentiment_report.csv'
        ],
        'signal_output_file': 'signal_output.parquet',
        'ticker_list_file': ticker_file,
        'days_lookback': days,
        'geography': geography
    }
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stock Sentiment and Signal Generation Pipeline")
    parser.add_argument('--ticker-file', default=config['ticker_list_file'],
                        help='Path to the ticker list CSV file')
    parser.add_argument('--output-file', default=config['sentiment_output_files'][0],
                        help='Path to the final output file (used by sentiment.py)')
    parser.add_argument('--geography', default=config['geography'],
                        help='Geography for subreddit selection', choices=['UK', 'Hong Kong', 'India', 'US'])
    parser.add_argument('--days', type=int, default=config['days_lookback'],
                        help='Number of days to look back for Reddit posts')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config['ticker_list_file'] = args.ticker_file
    config['extracted_data_file'] = 'stock_data_for_valid_tickers.csv'
    config['sentiment_output_files'] = [
        args.output_file,
        'stock_risk_sentiment_report.csv',
        'sector_sentiment_report.csv'
    ]
    config['geography'] = args.geography
    config['days_lookback'] = args.days
    
    # Run the pipeline
    success = run_pipeline(config)
    
    if success:
        print(f"Pipeline completed successfully at {datetime.now()}")
        print(f"Outputs saved to {args.output_file}")
    else:
        print("Pipeline failed. Check pipeline.log for details.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()