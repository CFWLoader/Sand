from src.bigshitmarketdownloader import BigShitMarketDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import pandas as pd
import os
import itertools

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

CONCERNED_TICKET_LIST=['300014', '600011', '000977']

def get_raw_dataset(cache_path):
    cache_csv_path = os.path.join(DATA_SAVE_DIR, "bsm%s_%s.csv" % (TRAIN_START_DATE, TRADE_END_DATE))
    df = None
    if cache_path is not None and os.path.isfile(cache_csv_path):
        df = pd.read_csv(cache_csv_path, encoding='utf-8')
    else:
        bsm_down = BigShitMarketDownloader(TRAIN_START_DATE, TRADE_END_DATE, CONCERNED_TICKET_LIST)
        df = bsm_down.fetch_data()
        df.to_csv(cache_csv_path, encoding='utf-8')
    return df

def get_dataset(use_cache=True):

    TRAIN_START_DATE = '2015-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2025-03-31'
    cache_csv_path = os.path.join(DATA_SAVE_DIR, "bsm%s_%s.csv"%(TRAIN_START_DATE, TRADE_END_DATE))
    cache_fecsv_path = os.path.join(DATA_SAVE_DIR, "bsm_fe%s_%s.csv"%(TRAIN_START_DATE, TRADE_END_DATE))
    processed_full = None
    if use_cache and os.path.isfile(cache_fecsv_path):
        processed_full = pd.read_csv(cache_fecsv_path, encoding='utf-8')
    else:
        df = get_raw_dataset(cache_csv_path)
        df.sort_values(['date', 'tic'], ignore_index=True).head()
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False)

        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])
        processed_full = processed_full.fillna(0)
        if use_cache:
            processed_full.to_csv(cache_fecsv_path, encoding='utf-8')

    processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)

    mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(len(train))
    print(len(trade))

    return train, trade

# res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2023-01-01', end_date='2023-12-31')
# res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2021-01-01', end_date='2021-12-31')
# res_df = adata.stock.market.get_market(stock_code='600001', k_type=1, start_date='2020-01-01')
# res_df = adata.stock.market.get_market(stock_code=stk_code, k_type=1, start_date=start_date)

if __name__ == '__main__':
    train_df, trade_df = get_dataset()

