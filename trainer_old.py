# trainer.py

import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import gc
import logging

# Set up logging
logging.basicConfig(filename='crypto_trainer.log', level=logging.INFO, 
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

# Feature augmentation for training
def feature_augmentation_train(df, previous_periods, future_periods):
    df['date'] = pd.to_datetime(df['Time'])
    df['Hour'] = df['date'].dt.hour
    df['Minute'] = df['date'].dt.minute

    new_columns = {}

    # Adding previous periods of close and volume
    for period in previous_periods:
        df_shifted = df.shift(periods=period)
        new_columns[f'Shift{period}'] = df_shifted['Close']
        new_columns[f'VShift{period}'] = df_shifted['Volume']
        del df_shifted
        gc.collect()

    # Adding future periods of close
    for period in future_periods:
        df_shifted = df.shift(periods=period)
        new_columns[f'Shift{period}'] = df_shifted['Close']
        del df_shifted
        gc.collect()

    # Concatenate all new columns to the original DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    del new_columns
    gc.collect()

    df.dropna(inplace=True)

    return df

# Calculate target variable
def calculate_target_variable(df, future_periods):
    shift_columns = [f'Shift{period}' for period in future_periods]

    # Calculate the absolute differences between future Shift- columns and the current 'Close' price
    diff_df = df[shift_columns].sub(df['Close'], axis=0).abs()

    # For each row, find the index of the column with the maximum absolute difference
    max_diff_idx = diff_df.values.argmax(axis=1)

    # Get the values from df[shift_columns] at the positions of maximum absolute difference
    df['Next_absolute_max'] = df[shift_columns].values[np.arange(len(df)), max_diff_idx]

    # Clean up
    del diff_df
    gc.collect()

    return df

# Training function
def train_model(hyperparameters=None):
    logging.info("Starting model training...")
    print("Starting model training...")
    data = read_data_from_table('ethusd')
    previous_periods = list(range(60, 0, -1))
    future_periods = list(range(-1, -60, -1))

    data_augmented = feature_augmentation_train(data, previous_periods, future_periods)
    data_augmented = calculate_target_variable(data_augmented, future_periods)

    # Define features and targets
    targets = ['Next_absolute_max']
    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{period}' for period in previous_periods]
    previous_volume_shift_features = [f'VShift{period}' for period in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    # Split data into training and validation sets
    div_test_training = -300
    training_data = data_augmented.iloc[:div_test_training]
    validation_data = data_augmented.iloc[div_test_training:]

    X_train = training_data[features]
    Y_train = training_data[targets]

    X_val = validation_data[features]
    Y_val = validation_data[targets]

    # Default hyperparameters
    default_hyperparameters = {
        "n_estimators": 600,
        "max_depth": 2,
        "learning_rate": 0.2,
        "min_child_weight": 1,
        "colsample_bytree": 0.5,
        "tree_method": 'hist',
    }

    # Use provided hyperparameters or fall back to defaults
    if hyperparameters is None:
        hyperparameters = default_hyperparameters
    else:
        # Fill in missing hyperparameters with defaults
        for key, value in default_hyperparameters.items():
            hyperparameters.setdefault(key, value)

    # Define and train the model
    model = xgb.XGBRegressor(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        min_child_weight=hyperparameters["min_child_weight"],
        colsample_bytree=hyperparameters["colsample_bytree"],
        learning_rate=hyperparameters["learning_rate"],
        tree_method=hyperparameters["tree_method"],
        n_jobs=4,
    )

    model.fit(X_train, Y_train)
    model.save_model("xgboost_model.json")
    logging.info("Model training completed and model saved.")
    print("Model training completed and model saved.")
