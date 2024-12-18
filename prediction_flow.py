# main.py

import threading
import time
import downloader
import filler
import predictor
import trainer

# def run_downloader():
#     while True:
#         downloader.update_history()
#         filler.fill_missing()


# def run_predictor():
#     while True:
#         predictor.predict_price()

def main():
    # Start downloader and filler in one separated thread
    # downloader_thread = threading.Thread(target=run_downloader)
    # downloader_thread.daemon = True
    # downloader_thread.start()

    # Start predictor (can be in main thread or another thread)
    # run_downloader()
    # print("d")
    # run_predictor()
    # print("p")
    # time.sleep(50)

    # Custom hyperparameters
    custom_hyperparameters = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 5,
        "colsample_bytree": 0.5,
        "tree_method": 'hist',
    }

    while True:
        # Measure execution time
        start_time = time.time()  # Record the start time
        downloader.update_history()
        filler.fill_missing()
        trainer.train_model(hyperparameters=custom_hyperparameters)
        predictor.predict_price()
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        time.sleep(35)

    
if __name__ == "__main__":
    main()
