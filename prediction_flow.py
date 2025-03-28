# main.py

import time
import downloader
import filler
import predictor
import trainer
import logging
import time

# Configure logging
logging.basicConfig(
    filename='crypto_data_main.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    custom_hyperparameters = {
        "n_estimators": 700,
        "max_depth": 2,
        "learning_rate": 0.2,
        "min_child_weight": 1,
        "colsample_bytree": 0.5,
        "tree_method": 'hist',
        "n_jobs": -1,
    }

    while True:
        try:
            start_time = time.time()
            downloader.update_history()
            filler.fill_missing()
            trainer.train_model(hyperparameters=custom_hyperparameters)
            predictor.predict_latest_price()
            time.sleep(18)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Successful execution. Time: {execution_time:.6f} seconds")
            logging.info(f"Successful execution. Time: {execution_time:.6f} seconds")

            
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}", exc_info=True)
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)
            continue

    
if __name__ == "__main__":
    main()
