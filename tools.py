# tools.py

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import logging

# ----------------------------------------
# 1. Database Utilities
# ----------------------------------------
def connect_to_sqlite(db_name="crypto_data.sqlite"):
    """
    Connect to SQLite database with WAL (Write-Ahead Logging).
    """
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  
    return conn

def read_data_from_table(table_name, db_name="crypto_data.sqlite"):
    """
    Read the entire table from the given SQLite DB into a pandas DataFrame.
    """
    conn = connect_to_sqlite(db_name)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def write_prediction_to_table(
        db_name,
        table_name,
        timestamp,
        current_close_price,
        predicted_next_absolute_max,
        predicted_percentage_change
    ):
    """
    Write a single prediction record to a given table. 
    Creates the table if it doesn't exist.
    """
    conn = connect_to_sqlite(db_name)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TEXT,
            current_close_price REAL,
            predicted_next_absolute_max REAL,
            predicted_percentage_change REAL
        )
    """)

    insert_query = f"""
        INSERT INTO {table_name}
        VALUES (?, ?, ?, ?)
    """

    data_tuple = (
        timestamp,
        current_close_price,
        predicted_next_absolute_max,
        predicted_percentage_change
    )

    cursor.execute(insert_query, data_tuple)
    conn.commit()
    conn.close()

# ----------------------------------------
# 2. Feature Engineering
# ----------------------------------------
def feature_augmentation(
    df,
    previous_periods,
    future_periods=None,
    is_training=True
):
    """
    Perform feature augmentation. If is_training=True, we also add
    future close shifts (for target creation). Otherwise, we skip it.
    """
    df['date'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['date'].dt.hour
    df['Minute'] = df['date'].dt.minute

    new_columns = {}

    # Add previous periods for Close and Volume
    for period in previous_periods:
        df_shifted = df.shift(periods=period)
        new_columns[f'Shift{period}'] = df_shifted['Close']
        new_columns[f'VShift{period}'] = df_shifted['Volume']
        del df_shifted
        gc.collect()

    # If training, add future periods of Close
    if is_training and future_periods is not None:
        for period in future_periods:
            df_shifted = df.shift(periods=period)
            new_columns[f'Shift{period}'] = df_shifted['Close']
            del df_shifted
            gc.collect()

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    df.dropna(inplace=True)
    return df

def calculate_target_variable(df, future_periods):
    """
    Calculate the Next_absolute_max column as the maximum absolute 
    difference from the current Close among the future Shift columns.
    """
    shift_columns = [f'Shift{period}' for period in future_periods]

    # Calculate absolute differences
    diff_df = df[shift_columns].sub(df['Close'], axis=0).abs()

    # Find index of max difference
    max_diff_idx = diff_df.values.argmax(axis=1)

    # Assign the value from the future Shift column with the max difference
    df['Next_absolute_max'] = df[shift_columns].values[np.arange(len(df)), max_diff_idx]

    del diff_df
    gc.collect()

    return df
