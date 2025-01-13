import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
import gc

# Function to connect to SQLite
def connect_to_sqlite():
    conn = sqlite3.connect("crypto_data.sqlite", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads and writes
    return conn

# Load training data
def load_training_data():
    conn = connect_to_sqlite()
    query = "SELECT * FROM ethusd ORDER BY Time ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Feature augmentation for training
def feature_augmentation_train(df, previous_periods):
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

# Function to find the best hyperparameters
def find_best_hyperparameters():
    print("Starting hyperparameter optimization...")

    # Load and prepare data
    data = load_training_data()
    print("Data loaded successfully.")
    previous_periods = list(range(60, 0, -1))
    data_augmented = feature_augmentation_train(data, previous_periods)
    print("Feature augmentation completed.")

    core_features = ['Close', 'Volume', 'Hour', 'Minute']
    previous_shift_features = [f'Shift{period}' for period in previous_periods]
    previous_volume_shift_features = [f'VShift{period}' for period in previous_periods]
    features = core_features + previous_shift_features + previous_volume_shift_features

    X = data_augmented[features]
    y = data_augmented['Close']  # Target variable

    # Define parameter grid
    param_grid = {
        'n_estimators': [600],
        'max_depth': [2],
        'learning_rate': [0.01, 0.2, 0.1],
        'min_child_weight': [1, 5, 10],
        'colsample_bytree': [0.5, 0.7, 1.0],
    }

    # Initialize model
    model = xgb.XGBRegressor(tree_method='hist', random_state=42)

    # Define GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Fit GridSearchCV
    print("Starting GridSearchCV...")
    grid_search.fit(X, y)
    print("Hyperparameter optimization completed.")

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best score (negative MSE): {best_score}")

    # Save best model
    best_model = grid_search.best_estimator_
    best_model.save_model("xgboost_model_optimized.json")
    print("Optimized model saved as 'xgboost_model_optimized.json'.")

if __name__ == "__main__":
    find_best_hyperparameters()
