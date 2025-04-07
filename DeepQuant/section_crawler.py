import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, MODEL_KWARGS
from pandas.core.interchange.dataframe_protocol import DataFrame
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from pprint import pprint
import sys
import itertools
from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
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
sys.path.append("../FinRL")

PREDICTION_ROOT='model_predictions'

def get_dataset():
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-31'

    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRADE_END_DATE,
                         ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
    print(config_tickers.DOW_30_TICKER)

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

    processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)

    mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

    processed_full.to_csv(os.path.join(DATA_SAVE_DIR, "sample1.csv"), encoding='utf-8')

    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(len(train))
    print(len(trade))

    return train, trade

def get_cached_dataset():
    processed_full = pd.read_csv(os.path.join(DATA_SAVE_DIR, "sample1.csv"), encoding='utf-8')
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-31'
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(len(train))
    print(len(trade))
    return train, trade

def get_env_kwargs(input_df):
    stock_dimension = len(input_df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    return env_kwargs

def train_model(train_df) -> A2C:
    env_kwargs = get_env_kwargs(train_df)
    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    if_using_a2c = True

    agent = DRLAgent(env=env_train)
    a2c_lparams = {"device": "cuda", "n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
    model_a2c = agent.get_model("a2c", model_kwargs=a2c_lparams)

    if if_using_a2c:
        # set up logger
        tmp_path = TENSORBOARD_LOG_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000) if if_using_a2c else None

    return trained_a2c

if __name__ == '__main__':
    # train_df, trade_df = get_dataset()
    train_df, trade_df = get_cached_dataset()
    '''
    trained_model = train_model(train_df)
    trained_model.save(os.path.join(TRAINED_MODEL_DIR, 'a2c-sample1.zip'))
    '''
    trained_model = A2C.load(os.path.join(TRAINED_MODEL_DIR, 'a2c-sample1.zip'))
    env_kwargs = get_env_kwargs(train_df)
    e_trade_gym = StockTradingEnv(df=trade_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym)
    check_and_make_directories([os.path.join(PREDICTION_ROOT, "a2c")])
    df_account_value_a2c.to_csv(os.path.join(PREDICTION_ROOT, "a2c", "action_values-1.csv"))
    df_actions_a2c.to_csv(os.path.join(PREDICTION_ROOT, "a2c", "actions-1.csv"))