# Crypto Data Pipeline

This repository contains a comprehensive cryptocurrency data pipeline that handles data downloading, filling missing values, training models, and predicting prices using machine learning algorithms.

## Project Structure

```
.
├── downloader.py        # Downloads historical crypto data using Bitget API
├── filler.py            # Fills in missing data using interpolation
├── trainer.py           # Trains an XGBoost model for price prediction
├── predictor.py         # Predicts future prices using the trained model
├── prediction_flow.py   # Manages the entire pipeline flow
└── README.md            # Project documentation
```

## Features
- **Data Downloading:** Downloads historical cryptocurrency data from Bitget using `ccxt`.
- **Data Filling:** Interpolates missing data using pandas.
- **Model Training:** Utilizes XGBoost for model training.
- **Price Prediction:** Predicts prices using the latest model.
- **Logging:** Detailed logs are maintained for easy monitoring and debugging.

## Requirements

Make sure you have Python installed and then run the following command to install the necessary packages:

```bash
pip install -r requirements.txt
```

### Sample `requirements.txt`
```text
pandas
numpy
xgboost
ccxt
sqlite3
logging
```

## Usage

1. **Download Data**
    ```bash
    python downloader.py
    ```

2. **Fill Missing Data**
    ```bash
    python filler.py
    ```

3. **Train Model**
    ```bash
    python trainer.py
    ```

4. **Predict Prices**
    ```bash
    python predictor.py
    ```

5. **Run Entire Pipeline**
    ```bash
    python prediction_flow.py
    ```

## Configuration

- Ensure that `ccxt_credentials.py` contains your API key and secret:

```python
APIKEY = 'your_api_key'
SECRET = 'your_api_secret'
```

- Modify hyperparameters in `prediction_flow.py` if needed:

```python
custom_hyperparameters = {
    "n_estimators": 700,
    "max_depth": 2,
    "learning_rate": 0.2,
    "min_child_weight": 1,
    "colsample_bytree": 0.5,
    "tree_method": 'hist',
    "n_jobs": -1,
}
```

## Logging

Logs are stored in respective log files:
- `crypto_downloader.log`
- `crypto_data_filler.log`
- `crypto_trainer.log`
- `crypto_predictor.log`
- `crypto_data_main.log`

## License
This project is licensed under the MIT License.

