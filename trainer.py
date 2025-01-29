# trainer.py

import logging
import xgboost as xgb
import pandas as pd

from tools import (
    read_data_from_table,
    feature_augmentation,
    calculate_target_variable
)

logging.basicConfig(
    filename='crypto_trainer.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def train_model(hyperparameters=None):
    """
    Train the XGBoost model on the 'ethusd' table data.
    """
    logging.info("Starting model training...")
    print("Starting model training...")

    # Load data
    data = read_data_from_table('ethusd')

    # Ensure 'timestamp' is a datetime type
    data['timestamp'] = pd.to_datetime(data['Time'], utc=True)

    # Calculate the cutoff date: 3 months ago from now
    three_months_ago = pd.Timestamp.utcnow() - pd.DateOffset(months=3)

    # Keep only the data from the last 3 months
    data = data.loc[data['timestamp'] >= three_months_ago]
    
    # Define shift periods
    previous_periods = list(range(60, 0, -1))    # 60 previous 1-min periods
    future_periods = list(range(-1, -60, -1))   # 60 future 1-min periods

    # Feature engineering
    data_augmented = feature_augmentation(
        df=data, 
        previous_periods=previous_periods,
        future_periods=future_periods,
        is_training=True
    )

    # Calculate target
    data_augmented = calculate_target_variable(
        df=data_augmented,
        future_periods=future_periods
    )

    # Define features and targets
    # targets = ['Next_absolute_max']
    targets = ['Shift-59']
    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{p}' for p in previous_periods]
    previous_volume_shift_features = [f'VShift{p}' for p in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    # Split data into training and validation sets
    # div_test_training = -300
    # training_data = data_augmented.iloc[:div_test_training]
    # validation_data = data_augmented.iloc[div_test_training:]

    # X_train = training_data[features]
    # Y_train = training_data[targets]

    # X_val = validation_data[features]
    # Y_val = validation_data[targets]

    X_train = data_augmented[features]
    Y_train = data_augmented[targets]


    # Default hyperparameters
    default_hyperparameters = {
        "n_estimators": 600,
        "max_depth": 2,
        "learning_rate": 0.2,
        "min_child_weight": 1,
        "colsample_bytree": 0.5,
        "tree_method": 'hist',
        "n_jobs": -1,
    }

    # Merge provided hyperparameters with defaults
    if hyperparameters is None:
        hyperparameters = default_hyperparameters
    else:
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

    # model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
    model.fit(X_train, Y_train, verbose=False)
    model.save_model("xgboost_model.json")

    logging.info("Model training completed and model saved.")
    print("Model training completed and model saved.")

if __name__ == "__main__":
    train_model()