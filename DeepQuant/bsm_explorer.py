from numpy import dtype
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC

from src.bigshitmarketdownloader import BigShitMarketDownloader
from src.bigshitmarketdataloader import BigShitMarketDataLoader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from finrl.agents.stablebaselines3.models import DRLAgent, MODEL_KWARGS
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt

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
MY_TRADE_END_DATE = '2025-03-31'

PREDICTION_ROOT = 'model_predictions'

CONCERNED_TICKET_LIST = ['300014', '600011', '000977']


# def get_raw_dataset(cache_path):
#     cache_csv_path = os.path.join(DATA_SAVE_DIR, "bsm%s_%s.csv" % (MY_TRAIN_START_DATE, MY_TRADE_END_DATE))
#     df = None
#     if cache_path is not None and os.path.isfile(cache_csv_path):
#         df = pd.read_csv(cache_csv_path, encoding='utf-8', dtype={ 'tic': 'string', 'date': 'string' }, parse_dates=['trade_time'])
#     else:
#         bsm_down = BigShitMarketDownloader(MY_TRAIN_START_DATE, MY_TRADE_END_DATE, CONCERNED_TICKET_LIST)
#         df = bsm_down.fetch_data()
#         df.to_csv(cache_csv_path, encoding='utf-8')
#     return df

def get_dataset(use_cache=True):
    # TRAIN_START_DATE = '2015-01-01'
    # TRAIN_END_DATE = '2020-07-01'
    # TRADE_START_DATE = '2020-07-01'
    # TRADE_END_DATE = '2025-03-31'
    check_and_make_directories([DATA_SAVE_DIR])
    cache_csv_path = os.path.join(DATA_SAVE_DIR, "bsm%s_%s.csv" % (MY_TRAIN_START_DATE, MY_TRADE_END_DATE))
    cache_fecsv_path = os.path.join(DATA_SAVE_DIR, "bsm_fe%s_%s.csv" % (MY_TRAIN_START_DATE, MY_TRADE_END_DATE))
    processed_full = None
    if use_cache and os.path.isfile(cache_fecsv_path):
        processed_full = pd.read_csv(cache_fecsv_path, encoding='utf-8')
    else:
        df = BigShitMarketDataLoader.load(DATA_SAVE_DIR, MY_TRAIN_START_DATE, MY_TRADE_END_DATE, CONCERNED_TICKET_LIST,
                                          print_logs=False)  # get_raw_dataset(cache_csv_path)
        df.sort_values(['date', 'tic'], ignore_index=True).head()
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=False,
            use_turbulence=True,
            user_defined_feature=False)

        processed = fe.preprocess_data(df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"],
                                                                                  how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])
        processed_full = processed_full.fillna(0)
        if use_cache:
            processed_full.to_csv(cache_fecsv_path, encoding='utf-8')

    processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)

    mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

    train = data_split(processed_full, MY_TRAIN_START_DATE, MY_TRAIN_END_DATE)
    trade = data_split(processed_full, MY_TRADE_START_DATE, MY_TRADE_END_DATE)
    print(len(train))
    print(len(trade))

    return train, trade


# res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2023-01-01', end_date='2023-12-31')
# res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2021-01-01', end_date='2021-12-31')
# res_df = adata.stock.market.get_market(stock_code='600001', k_type=1, start_date='2020-01-01')
# res_df = adata.stock.market.get_market(stock_code=stk_code, k_type=1, start_date=start_date)

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


def get_model_train_params(model_name: str) -> dict[str, any]:
    if model_name == 'a2c':
        return {
            "device": "cuda",
            "n_steps": 5,
            "ent_coef": 0.01,
            "learning_rate": 0.0007
        }
    elif model_name == 'ppo':
        return {
            "device": "cuda",
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }
    elif model_name == 'td3':
        return {
            "device": "cuda",
            "batch_size": 100,
            "buffer_size": 1000000,
            "learning_rate": 0.001
        }
    elif model_name == 'sac':
        return {
            "device": "cuda",
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
    return {"device": "cuda"}


def train_models(train_df, model_names: set[str]) -> dict[str, BaseAlgorithm]:
    env_kwargs = get_env_kwargs(train_df)
    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    trained_finrl_models = {}

    for rl_model_name in model_names:
        agent = DRLAgent(env=env_train)
        model_params = get_model_train_params(rl_model_name)
        rl_model = agent.get_model(rl_model_name, model_kwargs=model_params)
        # set up logger
        tmp_path = os.path.join(TENSORBOARD_LOG_DIR, rl_model_name)
        new_logger_rl_model = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        rl_model.set_logger(new_logger_rl_model)

        trained_a2c = agent.train_model(model=rl_model,
                                        tb_log_name=rl_model_name,
                                        total_timesteps=50000)
        trained_finrl_models[rl_model_name] = trained_a2c
    return trained_finrl_models

def load_trained_models(model_names: set[str]) -> dict[str, BaseAlgorithm]:
    trained_finrl_models = {}
    if 'a2c' in model_names:
        trained_finrl_models['a2c'] = A2C.load(os.path.join(TRAINED_MODEL_DIR, 'a2c-sample1.zip'))
    if 'ddpg' in model_names:
        trained_finrl_models['ddpg'] = DDPG.load(os.path.join(TRAINED_MODEL_DIR, 'ddpg-sample1.zip'))
    if 'ppo' in model_names:
        trained_finrl_models['ppo'] = PPO.load(os.path.join(TRAINED_MODEL_DIR, 'ppo-sample1.zip'))
    if 'td3' in model_names:
        trained_finrl_models['td3'] = TD3.load(os.path.join(TRAINED_MODEL_DIR, 'td3-sample1.zip'))
    if 'sac' in model_names:
        trained_finrl_models['sac'] = SAC.load(os.path.join(TRAINED_MODEL_DIR, 'sac-sample1.zip'))
    return trained_finrl_models


if __name__ == '__main__':
    train_df, trade_df = get_dataset()
    print(train_df.tic.unique())
    # trained_models = train_models(train_df, {'a2c', 'ddpg', 'ppo', 'td3', 'sac'})
    # for model_name, trained_model in trained_models.items():
    #     trained_model.save(os.path.join(TRAINED_MODEL_DIR, model_name + '-sample1.zip'))
    trained_models = load_trained_models({'a2c', 'ddpg', 'ppo', 'td3', 'sac'})
    env_kwargs = get_env_kwargs(train_df)
    e_trade_gym = StockTradingEnv(df=trade_df, turbulence_threshold=70, risk_indicator_col='turbulence', **env_kwargs)
    result_df = None
    for model_name, trained_model in trained_models.items():
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=trained_model,
            environment=e_trade_gym)
        check_and_make_directories([os.path.join(PREDICTION_ROOT, model_name)])
        # df_account_value.to_csv(os.path.join(PREDICTION_ROOT, model_name, "action_values-1.csv"))
        # df_actions.to_csv(os.path.join(PREDICTION_ROOT, model_name, "actions-1.csv"))
        if result_df is None:
            result_df = df_account_value.set_index(df_account_value.columns[0])
            result_df = result_df.rename(columns={result_df.columns[0]: model_name})
        else:
            right_df = df_account_value.set_index(df_account_value.columns[0])
            right_df = right_df.rename(columns={right_df.columns[0]: model_name})
            result_df = pd.merge(result_df, right_df, left_index=True, right_index=True)
    result_df.columns = list(trained_models.keys())

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    result_df.plot()
    plt.show()
