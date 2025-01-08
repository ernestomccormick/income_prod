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
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 5,
        "colsample_bytree": 0.5,
        "tree_method": 'hist',
    }

    while True:
        try:
            start_time = time.time()
            downloader.update_history()
            filler.fill_missing()
            trainer.train_model(hyperparameters=custom_hyperparameters)
            predictor.predict_price()
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"Successful execution. Time: {execution_time:.6f} seconds")
            time.sleep(35)
            
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}", exc_info=True)
            logging.info("Retrying in 5 seconds...")
            time.sleep(5)
            continue

    
if __name__ == "__main__":
    main()
