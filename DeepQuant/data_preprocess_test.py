from src.bigshitmarketdownloader import BigShitMarketDownloader
from src.bigshitmarketdataloader import BigShitMarketDataLoader
import pandas as pd
import os

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    # TRAIN_START_DATE,
    # TRAIN_END_DATE,
    # TEST_START_DATE,
    # TEST_END_DATE,
    # TRADE_START_DATE,
    # TRADE_END_DATE,
)

MY_TRAIN_START_DATE = '2015-01-01'
MY_TRAIN_END_DATE = '2020-07-01'
MY_TRADE_START_DATE = '2020-07-01'
MY_TRADE_END_DATE = '2025-03-31'

PREDICTION_ROOT='model_predictions'

CONCERNED_TICKET_LIST=['300014', '600011', '000977']

if __name__ == '__main__':
    cache_csv_path = BigShitMarketDataLoader.load(DATA_SAVE_DIR, MY_TRAIN_START_DATE, MY_TRADE_END_DATE, CONCERNED_TICKET_LIST)
    print(cache_csv_path.tic.unique())
