# main.py

import threading
import time
import downloader
import filler
import predictor

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

    while True:
        downloader.update_history()
        filler.fill_missing()
        predictor.predict_price()
        time.sleep(50)

    
if __name__ == "__main__":
    main()
