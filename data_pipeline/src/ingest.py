import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from db_helper import MongoDBHelper

load_dotenv()

TICKER = os.getenv("TICKER", "BBCA.JK")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "stock_data")

def fetch_data(ticker, start_date=None, end_date=None):
    """Fetches stock data from yfinance"""
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock = yf.Ticker(ticker)
    
    if start_date is None:
        # If no start date, fetch max history (e.g., last 10 years)
        df = stock.history(period="10y")
    else:
        df = stock.history(start=start_date, end=end_date)
        
    if df.empty:
        print("No data fetched.")
        return pd.DataFrame()
        
    df.reset_index(inplace=True)
    # Ensure Date column is string or datetime
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['Ticker'] = ticker
    
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
