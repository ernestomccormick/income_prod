# filler.py

import duckdb
import pandas as pd
import logging

# Set up logging
logging.basicConfig(filename='crypto_data_filler.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to connect to DuckDB
def connect_to_duckdb():
    return duckdb.connect("crypto_data.duckdb")

# Function to read data from the DuckDB table
def read_data_from_table(table_name):
    conn = connect_to_duckdb()
    df = conn.execute(f"SELECT * FROM {table_name}").df()
    conn.close()
    return df

# Function to fill missing timestamps
def fill_missing_data(df):
    # Ensure dataframe is sorted by time
    df = df.sort_values('Time').reset_index(drop=True)
    df.set_index('Time', inplace=True)
    # Reindex to fill missing timestamps
    idx = pd.date_range(df.index[0], df.index[-1], freq='1min')
    df = df.reindex(idx, method='ffill').reset_index()
    df.rename(columns={'index': 'Time'}, inplace=True)
    return df

# Function to update the DuckDB table with filled data
def update_duckdb_table(table_name, df):
    conn = connect_to_duckdb()
    conn.execute(f"DELETE FROM {table_name}")
    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
    conn.close()

# Main function to handle the filling process
def fill_missing():
    table_name = 'crypto_data'
    logging.info("Starting to fill missing data...")
    df = read_data_from_table(table_name)
    if not df.empty:
        filled_df = fill_missing_data(df)
        update_duckdb_table(table_name, filled_df)
        logging.info("Missing data filling process completed successfully.")
        print("Missing data filling process completed successfully.")
    else:
        logging.warning("No data to fill.")
        print("No data to fill.")
