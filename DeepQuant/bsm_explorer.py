from stable_baselines3 import A2C

from src.bigshitmarketdownloader import BigShitMarketDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, MODEL_KWARGS
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
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

PREDICTION_ROOT='model_predictions'

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
    train_df, trade_df = get_dataset()
    trained_model = train_model(train_df)
    trained_model.save(os.path.join(TRAINED_MODEL_DIR, 'a2c-sample1.zip'))
    # trained_model = A2C.load(os.path.join(TRAINED_MODEL_DIR, 'a2c-sample1.zip'))
    env_kwargs = get_env_kwargs(train_df)
    e_trade_gym = StockTradingEnv(df=trade_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym)
    check_and_make_directories([os.path.join(PREDICTION_ROOT, "a2c")])
    df_account_value_a2c.to_csv(os.path.join(PREDICTION_ROOT, "a2c", "action_values-1.csv"))
    df_actions_a2c.to_csv(os.path.join(PREDICTION_ROOT, "a2c", "actions-1.csv"))

