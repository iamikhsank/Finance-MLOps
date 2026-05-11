import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from db_helper import MongoDBHelper
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

load_dotenv()

TICKER = os.getenv("TICKER", "BBCA.JK")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "stock_data")

def fetch_data(ticker, start_date=None, end_date=None):
    """Fetches stock data from yfinance"""
    # Keep track of actual requested start date
    requested_start_date = start_date
    
    # Define the start date to FETCH, which includes a historical buffer (e.g., 60 days) 
    # to calculate indicators correctly (otherwise they return NaN on incremental data)
    fetch_start = None
    if start_date is not None:
        buffer_date = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
        fetch_start = buffer_date.strftime('%Y-%m-%d')
        print(f"Fetching with history buffer: requested={start_date}, fetching_from={fetch_start}")
    
    print(f"Fetching data for {ticker} from {fetch_start if fetch_start else '10y'} to {end_date}...")
    stock = yf.Ticker(ticker)
    
    if start_date is None:
        # If no start date, fetch max history (e.g., last 10 years)
        df = stock.history(period="10y")
    else:
        # Use our extended buffer window
        df = stock.history(start=fetch_start, end=end_date)
        
    if df.empty:
        print("No data fetched.")
        return pd.DataFrame()
        
    df.reset_index(inplace=True)
    # Ensure Date column is string or datetime
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['Ticker'] = ticker
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    # SMA 14
    indicator_sma = SMAIndicator(close=df["Close"], window=14, fillna=False)
    df['SMA_14'] = indicator_sma.sma_indicator()
    
    # RSI 14
    indicator_rsi = RSIIndicator(close=df["Close"], window=14, fillna=False)
    df['RSI_14'] = indicator_rsi.rsi()
    
    # MACD
    indicator_macd = MACD(close=df["Close"], fillna=False)
    df['MACD'] = indicator_macd.macd()
    
    # Bollinger Bands
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()
    
    # Drop rows with NaN values created by moving windows (e.g. first 33 rows)
    df.dropna(inplace=True)
    
    # FILTER BACK to only the requested dates after calculations
    if requested_start_date is not None:
        before_filter = len(df)
        df = df[df['Date'] >= requested_start_date]
        print(f"Filtered data to include only new rows starting from {requested_start_date}. Found {len(df)} new rows.")
    
    return df

def run_ingestion():
    db = MongoDBHelper()
    db.connect()
    
    # 1. Check latest date in DB
    latest_date_str = db.get_latest_date(COLLECTION_NAME, TICKER)
    
    if latest_date_str:
        print(f"Latest data in DB for {TICKER} is up to {latest_date_str}")
        # Next date to fetch
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
        start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Check if start_date is today or in future
        today_str = datetime.now().strftime('%Y-%m-%d')
        if start_date > today_str:
            print("Data is already up to date.")
            db.close()
            return
            
        df = fetch_data(TICKER, start_date=start_date)
    else:
        print(f"No existing data for {TICKER}. Fetching historical data...")
        df = fetch_data(TICKER)
        
    if not df.empty:
        # Convert DataFrame to list of dicts for MongoDB
        records = df.to_dict('records')
        db.insert_many(COLLECTION_NAME, records)
        print(f"Successfully ingested {len(records)} records.")
    else:
        print("No new data to ingest.")
        
    db.close()

if __name__ == "__main__":
    run_ingestion()
