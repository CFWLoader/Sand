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
MY_TRAIN_END_DATE = '2024-07-01'
MY_TRADE_START_DATE = '2024-07-01'
MY_TRADE_END_DATE = '2025-07-10'

PREDICTION_ROOT='model_predictions'

CONCERNED_TICKET_LIST=['300014', '600011', '000977', '000766', '002415', '600036' ]
# CONCERNED_TICKET_LIST=['300014', '600011', '000977']

if __name__ == '__main__':
    bsm_loader = BigShitMarketDataLoader('exp1', MY_TRAIN_START_DATE, MY_TRADE_END_DATE, CONCERNED_TICKET_LIST)
    # raw_df = bsm_loader.load()
    train_df, trade_df = bsm_loader.load_finrl_format_data(MY_TRAIN_END_DATE, MY_TRADE_START_DATE)
    mvo_result = bsm_loader.compute_mvo()
    mvo_result.rename({ 'Mean Var': 'account_value' }, inplace=True)
    mvo_result.to_csv(os.path.join('exp1', 'mvo_result.csv'), encoding='utf-8')
    # print(cache_csv_path.tic.unique())
