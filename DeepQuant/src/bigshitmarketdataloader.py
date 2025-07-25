from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
import pandas as pd
import numpy as np
import adata
import itertools
import os
from scipy import optimize
from scipy.optimize import linprog
from pypfopt.efficient_frontier import EfficientFrontier

class BigShitMarketDataLoader(object):

    default_ticket_indicators = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    def __init__(self,
                 work_dir: str,
                 data_start_date: str,
                 data_end_date: str,
                 tic_list: list[str],
                 use_cache = True,
                 fix_missing_data = True,
                 trade_data_column_name: str = 'date',
                 print_logs = True,
                 ticket_indicators = None,
                 ):
        # input params
        self.work_dir = work_dir
        self.dataset_dir = os.path.join(work_dir, "dataset")
        self.dataset_start_date = data_start_date
        self.dataset_end_date = data_end_date
        self.tic_list = tic_list
        self.use_cache = use_cache
        self.fix_missing_data = fix_missing_data
        self.trade_data_column_name = trade_data_column_name
        self.print_logs = print_logs
        self.ticket_indicators = ticket_indicators if ticket_indicators is not None else BigShitMarketDataLoader.default_ticket_indicators
        # internal variables
        self.raw_data = None
        self.fixed_raw_data = None
        self.processed_full_data = None
        self.finrl_train = None
        self.finrl_trade = None
        self.mvo_result = None
        # init ops
        BigShitMarketDataLoader.check_and_make_directories([self.dataset_dir])

    def load_finrl_format_data(self, train_end_date: str, trade_start_date: str):
        if self.fix_missing_data and self.fixed_raw_data is None or self.raw_data is None:
            self.load()
        self.preprocess_indicators_and_cache()
        self.finrl_train = data_split(self.processed_full_data, self.dataset_start_date, train_end_date)
        self.finrl_trade = data_split(self.processed_full_data, trade_start_date, self.dataset_end_date)
        # print(len(train))
        # print(len(trade))
        return self.finrl_train, self.finrl_trade

    def load(self):
        self.load_raw_data_only()
        if self.raw_data is None:
            print("download data failed!")
            return None
        print('Loaded tickets: ', self.raw_data.tic)
        if self.fix_missing_data:
            fixcache_csv_path = os.path.join(self.dataset_dir, "bsmfix%s_%s.csv" % (self.dataset_start_date, self.dataset_end_date))
            if os.path.exists(fixcache_csv_path):
                cached_df = self.load_cached_dataset(fixcache_csv_path)
            else:
                cached_df = self.fix_missing_dates(self.raw_data)
                cached_df.to_csv(fixcache_csv_path, encoding='utf-8', index=False)
            self.fixed_raw_data = cached_df
        return self.fixed_raw_data if self.fix_missing_data else self.raw_data

    def load_raw_data_only(self):
        cached_df = self.load_cached_dataset()
        if cached_df is None:
            print("Cache not found, download it")
            cached_df = self.download_and_cache(self.dataset_start_date, self.dataset_end_date, self.tic_list)
        self.raw_data = cached_df
        return self.raw_data

    def fix_missing_dates(self, in_df: pd.DataFrame):
        df = in_df.copy()
        # df.drop(columns = ['Unnamed: 0', 'index'], inplace=True)
        # 步骤1：生成所有股票和日期的完整组合
        all_stocks = df['tic'].unique()
        all_dates = df[self.trade_data_column_name].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=[self.trade_data_column_name, 'tic'])

        # 步骤2：创建完整组合的DataFrame
        df_complete = pd.DataFrame(index=full_index).reset_index()

        # 步骤3：左连接原始数据，标记缺失项
        merged = df_complete.merge(df, on=[self.trade_data_column_name, 'tic'], how='left', indicator=True)
        missing_rows = merged[merged['_merge'] == 'left_only'][[self.trade_data_column_name, 'tic']]
        merged.drop(columns=['_merge'], inplace=True)
        # merged.index = merged[trade_data_column_name].factorize()[0]

        return BigShitMarketDataLoader.fill_missing_dates(merged, missing_rows, verbose = self.print_logs)

    def fill_missing_dates(in_df: pd.DataFrame, missing_rows: pd.DataFrame, trade_data_column_name: str = 'date', verbose = True):
        # 一些交易相关的值直接填0
        in_df.fillna({'open': 0, 'high': 0, 'low': 0, 'volume': 0, 'amount': 0, 'change_pct': 0, 'change': 0, 'turnover_ratio': 0, 'pre_close': 0}, inplace=True)
        # 补充时间相关的值
        trade_time_col_idx = in_df.columns.get_loc('trade_time')
        in_df.iloc[missing_rows.index, trade_time_col_idx] = pd.to_datetime(in_df.iloc[missing_rows.index][trade_data_column_name], errors='coerce')
        missing_days = in_df.iloc[missing_rows.index, trade_time_col_idx].dt.dayofweek
        in_df.iloc[missing_rows.index, in_df.columns.get_loc('day')] = missing_days
        last_fetch_row_idx = 0
        close_col_idx = in_df.columns.get_loc('close')
        high_col_idx = in_df.columns.get_loc('high')
        low_col_idx = in_df.columns.get_loc('low')
        open_col_idx = in_df.columns.get_loc('open')
        # 列出需要填值的票
        for row_idx, row_to_fix in missing_rows.iterrows():
            if verbose:
                print(f'Fixing tick {row_to_fix.tic}({row_to_fix[trade_data_column_name]}) NaN values')
            prev_search = in_df[(in_df.index > last_fetch_row_idx) & (in_df.index < row_idx) & (in_df.tic == row_to_fix.tic) & (in_df['close'].notna())]
            # prev_search = in_df[in_df.tic == row_to_fix.tic]
            if len(prev_search) >= 1:
                last_match = prev_search.tail(1)
                last_fetch_row_idx = last_match.index.values[0]
                close_value = last_match['close'].values[0]
                in_df.iloc[row_idx, close_col_idx] = close_value
                in_df.iloc[row_idx, high_col_idx] = close_value
                in_df.iloc[row_idx, low_col_idx] = close_value
                in_df.iloc[row_idx, open_col_idx] = close_value
                if verbose:
                    print(f'Found previous date close value: {last_match[trade_data_column_name].values[0]}={close_value}')
                continue
            if verbose:
                print('No previous date close value found, try looking for next date close value')
            forward_search = in_df[(in_df.index > row_idx) & (in_df.tic == row_to_fix.tic) & (in_df['close'].notna())]
            first_match = forward_search.head(1)
            close_value = first_match['close'].values[0]
            in_df.iloc[row_idx, close_col_idx] = close_value
            in_df.iloc[row_idx, high_col_idx] = close_value
            in_df.iloc[row_idx, low_col_idx] = close_value
            in_df.iloc[row_idx, open_col_idx] = close_value
            if verbose:
                print(f'Found next date close value: {first_match[trade_data_column_name].values[0]}={close_value}')
        return in_df

    def load_cached_dataset(self, specified_path: str = None):
        cache_csv_path = os.path.join(self.dataset_dir, "bsm%s_%s.csv" % (self.dataset_start_date, self.dataset_end_date)) if specified_path is None else specified_path
        if os.path.isfile(cache_csv_path):
            return pd.read_csv(cache_csv_path, encoding='utf-8', parse_dates=['trade_time'],
                             dtype={
                                 'tic': 'string',
                                 'date': 'string',
                                 'open': 'float',
                                 'close': 'float',
                                 'high': 'float',
                                 'low': 'float',
                                 'volume': 'float',
                                 'amount': 'float',
                                 'change_pct': 'float',
                                 'change': 'float',
                                 'turnover_ratio': 'float',
                                 'pre_close': 'float',
                                 'day': 'int'
                             }
                             )
        return None

    def download_and_cache(self, data_start_date: str, data_end_date: str, tic_list: list[str], proxy=None, auto_adjust=False):
        cache_csv_path = os.path.join(self.dataset_dir, "bsm%s_%s.csv" % (self.dataset_start_date, self.dataset_end_date))
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in tic_list:
            temp_df = adata.stock.market.get_market(stock_code=tic, k_type=1, start_date=data_start_date,
                                                    end_date=data_end_date)
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(tic_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index(drop=True)
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "stock_code": "tic",
                    "trade_date": 'date',
                },
                inplace=True,
            )
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        temp_datetime_col = pd.to_datetime(data_df["trade_time"])
        data_df["trade_time"] = temp_datetime_col
        data_df["day"] = temp_datetime_col.dt.dayofweek
        # convert date to standard string format, easy to filter
        # data_df["date"] = temp_datetime_col.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        data_df.to_csv(cache_csv_path, encoding='utf-8', index=False)

        return data_df

    def preprocess_indicators_and_cache(self):
        cache_fecsv_path = os.path.join(self.dataset_dir,
                                        "bsm_fe%s_%s.csv" % (self.dataset_start_date, self.dataset_end_date))
        if os.path.isfile(cache_fecsv_path):
            self.processed_full_data = self.load_cached_dataset(cache_fecsv_path)
            return self.processed_full_data
        local_process_df = self.fixed_raw_data if self.fix_missing_data else self.raw_data
        local_process_df.sort_values(['date', 'tic'], ignore_index=True).head()
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.ticket_indicators,
            use_vix=False,
            use_turbulence=True,
            user_defined_feature=False)

        processed = fe.preprocess_data(local_process_df)
        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"],
                                                                                  how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])
        processed_full = processed_full.fillna(0)
        if self.use_cache:
            processed_full.to_csv(cache_fecsv_path, encoding='utf-8', index=False)

        # processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)
        self.processed_full_data = processed_full
        return self.processed_full_data
        # mvo_df = processed_full.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]

    def compute_mvo(self) -> pd.DataFrame:
        '''
        不知道这段用来做什么的
        mvo_df = self.processed_full_data.sort_values(['date', 'tic'], ignore_index=True)[['date', 'tic', 'close']]
        fst = mvo_df
        fst = fst.iloc[0 * 29:0 * 29 + 29, :]
        tic = fst['tic'].tolist()

        mvo = pd.DataFrame()

        for k in range(len(tic)):
            mvo[tic[k]] = 0

        for i in range(mvo_df.shape[0] // 29):
            n = mvo_df
            n = n.iloc[i * 29:i * 29 + 29, :]
            date = n['date'][i * 29]
            mvo.loc[date] = n['close'].tolist()
        '''

        # extract asset prices
        StockData = self.finrl_train.pivot(index="date", columns="tic", values="close")
        TradeData = self.finrl_trade.pivot(index="date", columns="tic", values="close")
        TradeData.to_numpy()

        # compute asset returns
        arStockPrices = np.asarray(StockData)
        [Rows, Cols] = arStockPrices.shape
        arReturns = BigShitMarketDataLoader.StockReturnsComputing(arStockPrices, Rows, Cols)

        # compute mean returns and variance covariance matrix of returns
        meanReturns = np.mean(arReturns, axis=0)
        covReturns = np.cov(arReturns, rowvar=False)

        # set precision for printing results
        np.set_printoptions(precision=3, suppress=True)

        # display mean returns and variance-covariance matrix of returns
        print('Mean returns of assets in k-portfolio 1\n', meanReturns)
        print('Variance-Covariance matrix of returns\n', covReturns)

        ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
        raw_weights_mean = ef_mean.max_sharpe()
        cleaned_weights_mean = ef_mean.clean_weights()
        mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])

        LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
        Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

        Portfolio_Assets = TradeData @ Initial_Portfolio
        self.mvo_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
        return self.mvo_result

    def check_and_make_directories(directories: list[str], root_dir = "."):
        for directory in directories:
            if not os.path.exists(os.path.join(root_dir, directory)):
                os.makedirs(os.path.join(root_dir, directory))

    def MaximizeReturns(MeanReturns, PortfolioSize):

        # dependencies

        c = (np.multiply(-1, MeanReturns))
        A = np.ones([PortfolioSize, 1]).T
        b = [1]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='simplex')

        return res

    def MinimizeRisk(CovarReturns, PortfolioSize):

        def f(x, CovarReturns):
            func = np.matmul(np.matmul(x, CovarReturns), x.T)
            return func

        def constraintEq(x):
            A = np.ones(x.shape)
            b = 1
            constraintVal = np.matmul(A, x.T) - b
            return constraintVal

        xinit = np.repeat(0.1, PortfolioSize)
        cons = ({'type': 'eq', 'fun': constraintEq})
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        opt = optimize.minimize(f, x0=xinit, args=(CovarReturns), bounds=bnds, constraints=cons, tol=10 ** -3)

        return opt

    def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):

        def f(x, CovarReturns):
            func = np.matmul(np.matmul(x, CovarReturns), x.T)
            return func

        def constraintEq(x):
            AEq = np.ones(x.shape)
            bEq = 1
            EqconstraintVal = np.matmul(AEq, x.T) - bEq
            return EqconstraintVal

        def constraintIneq(x, MeanReturns, R):
            AIneq = np.array(MeanReturns)
            bIneq = R
            IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
            return IneqconstraintVal

        xinit = np.repeat(0.1, PortfolioSize)
        cons = ({'type': 'eq', 'fun': constraintEq},
                {'type': 'ineq', 'fun': constraintIneq, 'args': (MeanReturns, R)})
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        opt = optimize.minimize(f, args=(CovarReturns), method='trust-constr', x0=xinit, bounds=bnds, constraints=cons, tol=10 ** -3)

        return opt

    def StockReturnsComputing(StockPrice, Rows, Columns):
        StockReturn = np.zeros([Rows - 1, Columns])
        for j in range(Columns):  # j: Assets
            for i in range(Rows - 1):  # i: Daily Prices
                StockReturn[i, j] = ((StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]) * 100

        return StockReturn
