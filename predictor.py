# predictor.py

import logging
import pandas as pd
import xgboost as xgb

from tools import (
    connect_to_sqlite,
    feature_augmentation,
    write_prediction_to_table
)

logging.basicConfig(
    filename='crypto_predictor.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_latest_data_for_prediction():
    """
    Load the last 61 rows (for 60 shifts) from 'ethusd' table.
    """
    conn = connect_to_sqlite()
    query = "SELECT * FROM ethusd ORDER BY Time DESC LIMIT 61"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Return in chronological order
    return df.sort_values('Time')

def predict_latest_price():
    """
    Predict next_absolute_max for the most recent record in 'ethusd'.
    """
    # Load data
    df = load_latest_data_for_prediction()

    # Feature augmentation (no future periods needed here)
    previous_periods = list(range(60, 0, -1))
    df_augmented = feature_augmentation(
        df=df,
        previous_periods=previous_periods,
        future_periods=None,
        is_training=False
    )

    # The latest row for prediction
    X_test = df_augmented.iloc[[-1]]
    current_time = X_test['Time'].iloc[0]

    # Load trained model
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")

    # Construct feature list
    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{p}' for p in previous_periods]
    previous_volume_shift_features = [f'VShift{p}' for p in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    X_test = X_test[features]

    # Predict
    y_pred = model.predict(X_test)
    predicted_next_absolute_max = float(y_pred[0])

    # Compute extra info
    current_close_price = float(X_test['Close'].iloc[0])
    predicted_percentage_change = 100 * (predicted_next_absolute_max / current_close_price - 1)

    # Logging
    print(f"Prediction: {predicted_next_absolute_max:.3f} at {current_time} "
          f"with Close {current_close_price:.3f}: "
          f"Predicted change {predicted_percentage_change:.2f}%")
    logging.info(
        f"Prediction {predicted_next_absolute_max:.3f} made at {current_time} "
        f"with Close {current_close_price:.3f}: "
        f"Predicted change {predicted_percentage_change:.2f}%"
    )

    # Save the prediction in a table named 'prediction_seq'
    write_prediction_to_table(
        db_name="crypto_data.sqlite",
        table_name="prediction_seq",
        timestamp=current_time,
        current_close_price=current_close_price,
        predicted_next_absolute_max=predicted_next_absolute_max,
        predicted_percentage_change=predicted_percentage_change
    )

def predict_range(start_time: str, end_time: str):
    """
    Generate a prediction of the next_absolute_max for each 1-minute data point
    between start_time and end_time, writing results to 'prediction_seq'.

    :param start_time: Start of the time range, e.g. '2023-01-01 00:00:00'
    :param end_time: End of the time range, e.g. '2023-01-02 00:00:00'
    """
    logging.info(f"Starting predict_range from {start_time} to {end_time}")

    # 1. Query rows including 60 minutes before start_time for feature augmentation
    query = f"""
        SELECT * 
        FROM ethusd
        WHERE Time <= '{end_time}'
          AND Time >= (
            SELECT MIN(Time)
            FROM (
              SELECT Time
              FROM ethusd
              WHERE Time <= '{start_time}'
              ORDER BY Time DESC
              LIMIT 60
            )
          )
        ORDER BY Time
    """

    # Fetch data from the database
    conn = connect_to_sqlite()
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert Time column to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # 2. Feature augmentation with 60 previous periods
    previous_periods = list(range(60, 0, -1))
    df_augmented = feature_augmentation(
        df=df,
        previous_periods=previous_periods,
        future_periods=None,
        is_training=False
    )

    # 3. Load the trained model
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")

    # Define the feature columns
    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{p}' for p in previous_periods]
    previous_volume_shift_features = [f'VShift{p}' for p in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    # 4. Loop through df_augmented, skipping the first 60 rows
    for i in range(60, len(df_augmented)):
        row = df_augmented.iloc[i]
        row_time = row['Time']

        # Skip rows outside of the [start_time, end_time] range
        if row_time < pd.to_datetime(start_time) or row_time > pd.to_datetime(end_time):
            continue

        # Prepare feature vector
        X_test = df_augmented.iloc[[i]][features]

        # Predict
        y_pred = model.predict(X_test)
        predicted_next_absolute_max = float(y_pred[0])

        # Compute additional metrics
        current_close_price = float(X_test['Close'].iloc[0])
        predicted_percentage_change = 100 * (predicted_next_absolute_max / current_close_price - 1)

        # Logging / Printing
        logging.info(
            f"Predict_range => Time: {row_time}, "
            f"Close: {current_close_price}, "
            f"Pred: {predicted_next_absolute_max}, "
            f"Change: {predicted_percentage_change:.2f}%"
        )
        print(
            f"Prediction: {predicted_next_absolute_max:.3f} at {row_time} | "
            f"Close: {current_close_price:.3f} | "
            f"Change: {predicted_percentage_change:.2f}%"
        )

        # 5. Write the prediction to 'prediction_seq'
        write_prediction_to_table(
            db_name="crypto_data.sqlite",
            table_name="prediction_seq",
            timestamp=str(row_time),
            current_close_price=current_close_price,
            predicted_next_absolute_max=predicted_next_absolute_max,
            predicted_percentage_change=predicted_percentage_change
        )

    logging.info(f"Completed predict_range from {start_time} to {end_time}")