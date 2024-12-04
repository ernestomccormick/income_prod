# predictor.py

import duckdb
import pandas as pd
import xgboost as xgb
import gc
import threading
import trainer  # Importing the trainer module
import logging
import time

# Set up logging
logging.basicConfig(filename='crypto_predictor.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to connect to DuckDB
def connect_to_duckdb():
    return duckdb.connect("crypto_data.duckdb")

# Load the latest data for prediction
def load_latest_data():
    conn = connect_to_duckdb()
    df = conn.execute("SELECT * FROM crypto_data ORDER BY Time DESC LIMIT 61").df()
    conn.close()
    return df.sort_values('Time')

# Feature augmentation for prediction
def feature_augmentation_predict(df, previous_periods):
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

    # Concatenate all new columns to the original DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    del new_columns
    gc.collect()

    df.dropna(inplace=True)

    return df

# Prediction function
def predict_price():
    previous_periods = list(range(60, 0, -1))

    data = load_latest_data()
    data_augmented = feature_augmentation_predict(data, previous_periods)

    # Use the latest row for prediction
    X_test = data_augmented.iloc[[-1]]
    current_time = X_test['Time'].iloc[0]

    # Load the model
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")

    # Features used in training
    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{period}' for period in previous_periods]
    previous_volume_shift_features = [f'VShift{period}' for period in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    X_test = X_test[features]

    # Predict
    y_pred = model.predict(X_test)

    current_close_price = X_test['Close'].iloc[0]
    predicted_next_absolute_max = y_pred[0]
    predicted_percentage_change = 100 * (predicted_next_absolute_max / current_close_price - 1)

    print(f"Prediction made {y_pred} at {current_time} with Close {current_close_price}: Predicted change {predicted_percentage_change:.2f}%")
    logging.info(f"Prediction {y_pred} made at {current_time} with Close {current_close_price}: Predicted change {predicted_percentage_change:.2f}%")


    # Get XGBoost hyperparameters
    model_params = model.get_xgb_params()
    # Extract the required hyperparameters
    hyperparameters = {
        'n_estimators': model_params.get('n_estimators', None),
        'max_depth': model_params.get('max_depth', None),
        'learning_rate': model_params.get('learning_rate', None),
        'min_child_weight': model_params.get('min_child_weight', None),
        'colsample_bytree': model_params.get('colsample_bytree', None)
    }

    # Save the prediction to DuckDB
    save_prediction_to_duckdb(
    timestamp=X_test.index[0],
    current_close_price=float(current_close_price),
    predicted_next_absolute_max=float(predicted_next_absolute_max),
    predicted_percentage_change=predicted_percentage_change,
    hyperparameters=hyperparameters
)

    # Trigger market transaction
    # transaction_successful = execute_market_transaction()

    # After market transaction is confirmed, retrain the model
    # if transaction_successful:
    #     train_model_thread = threading.Thread(target=trainer.train_model)
    #     train_model_thread.start()

# Function to save prediction to DuckDB
def save_prediction_to_duckdb(timestamp, current_close_price, predicted_next_absolute_max, predicted_percentage_change, hyperparameters):
    conn = connect_to_duckdb()
    cursor = conn.cursor()

    # Check if 'predictions' table exists, create if not
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            timestamp TIMESTAMP,
            current_close_price DOUBLE,
            predicted_next_absolute_max DOUBLE,
            predicted_percentage_change DOUBLE,
            n_estimators INTEGER,
            max_depth INTEGER,
            learning_rate DOUBLE,
            min_child_weight INTEGER,
            colsample_bytree DOUBLE
        )
    """)

    # Prepare the insert statement
    insert_query = """
        INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # Prepare the data
    data_tuple = (
        pd.to_datetime(timestamp),
        current_close_price,
        predicted_next_absolute_max,
        predicted_percentage_change,
        hyperparameters['n_estimators'],
        hyperparameters['max_depth'],
        hyperparameters['learning_rate'],
        hyperparameters['min_child_weight'],
        hyperparameters['colsample_bytree']
    )

    # Insert the data
    cursor.execute(insert_query, data_tuple)
    conn.commit()
    conn.close()
    logging.info("Prediction saved to DuckDB 'predictions' table.")

# Dummy function to simulate market transaction
# def execute_market_transaction():
#     # Implement your market transaction logic here
#     # For this example, we'll simulate a transaction confirmation
#     logging.info("Executing market transaction...")
#     # Simulate transaction processing time
#     logging.info("Market transaction confirmed.")
#     return True
