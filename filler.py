# filler.py

import sqlite3
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='crypto_data_filler.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to connect to SQLite
def connect_to_sqlite():
    conn = sqlite3.connect("crypto_data.sqlite", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads and writes
    return conn

# Function to read data from the SQLite table
def read_data_from_table(table_name):
    conn = connect_to_sqlite()
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to fill missing timestamps
def fill_missing_data(df):
    # Ensure dataframe is sorted by time
    df = df.sort_values('Time').reset_index(drop=True)

    # Drop duplicate timestamps, keeping the first occurrence
    df = df.drop_duplicates(subset='Time', keep='first')

    # Set the 'Time' column as the index
    df['Time'] = pd.to_datetime(df['Time'])  # Ensure 'Time' is datetime
    df.set_index('Time', inplace=True)

    # Reindex to fill missing timestamps
    idx = pd.date_range(df.index[0], df.index[-1], freq='1min')
    df = df.reindex(idx, method='ffill').reset_index()

    # Rename the index back to 'Time'
    df.rename(columns={'index': 'Time'}, inplace=True)

    return df

# Function to update the SQLite table with filled data
def update_sqlite_table(table_name, df):
    conn = connect_to_sqlite()
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    conn.commit()
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

# Main function to handle the filling process
def fill_missing():
    table_name = 'ethusd'
    logging.info("Starting to fill missing data...")
    df = read_data_from_table(table_name)
    if not df.empty:
        filled_df = fill_missing_data(df)
        update_sqlite_table(table_name, filled_df)
        logging.info("Missing data filling process completed successfully.")
        print("Missing data filling process completed successfully.")
    else:
        logging.warning("No data to fill.")
        print("No data to fill.")
