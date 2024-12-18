# downloader.py

import ccxt
import sqlite3
import time
import numpy as np
import pandas as pd
import calendar
from datetime import datetime, timedelta
import logging
from ccxt_credentials import APIKEY, SECRET

# Set up logging
logging.basicConfig(filename='crypto_downloader.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize Bitget API
bitget = ccxt.bitget({
    'apiKey': APIKEY,
    'secret': SECRET
})

# Define database connection function
def connect_to_sqlite():
    conn = sqlite3.connect("crypto_data.sqlite", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads
    return conn

# Function to check if table exists
def check_table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

# Function to fetch the latest timestamp from the SQLite table
def get_latest_timestamp(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(Time) FROM {table_name}")
    result = cursor.fetchone()
    return datetime.fromisoformat(result[0]) if result and result[0] else None

# Function to fetch OHLCV data
def fetch_ohlcv_data(pair, start_time):
    limit = 1000
    since = calendar.timegm(start_time.utctimetuple()) * 1000
    ohlcv1 = bitget.fetch_ohlcv(symbol=pair, timeframe='1m', since=since, limit=limit)
    ohlcv2 = bitget.fetch_ohlcv(symbol=pair, timeframe='1m', since=since, limit=limit)
    ohlcv = ohlcv1 + ohlcv2
    df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    return df

# Main function to handle the update process
def update_history():
    table_name = 'ethusd'
    symbol = 'ETH/USDT'
    conn = connect_to_sqlite()

    # Check if the table exists, if not create it
    if not check_table_exists(conn, table_name):
        logging.info(f"Table {table_name} does not exist, creating a new table.")
        df = fetch_ohlcv_data(symbol, datetime.now() - timedelta(days=120))
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    else:
        latest_timestamp = get_latest_timestamp(conn, table_name)
        if latest_timestamp:
            logging.info(f"Latest timestamp in the table: {latest_timestamp}")
            print(f"Latest timestamp in the table: {latest_timestamp}")
            next_minute = latest_timestamp + timedelta(minutes=1)
            df = fetch_ohlcv_data(symbol, next_minute)
            if not df.empty:
                df = df[df['Time'] > latest_timestamp]  # Ensure no duplicates
                if not df.empty:
                    logging.info(f"Inserting {len(df)} new records into {table_name}.")
                    print(f"Inserting {len(df)} new records into {table_name}.")
                    df.to_sql(table_name, conn, if_exists='append', index=False)
                else:
                    logging.info("No new data to insert.")
                    print("No new data to insert.")
            else:
                logging.info("No new data available from API.")
                print("No new data available from API.")
        else:
            logging.warning("Failed to retrieve the latest timestamp from the table.")
            print("Failed to retrieve the latest timestamp from the table.")
    
    conn.close()
