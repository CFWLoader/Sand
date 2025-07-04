import pandas as pd
import os

class BigShitMarketDataLoader(object):

    def load(dataset_dir: str, data_start_date: str, data_end_date: str, tic_list: list[str], use_cache = True, fix_missing_dates = True, trade_data_column_name: str = 'date', print_logs = True):
        cache_csv_path = os.path.join(dataset_dir, "bsm%s_%s.csv" % (data_start_date, data_end_date))
        cached_df = BigShitMarketDataLoader.load_cached_dataset(cache_csv_path)
        if cached_df is None:
            print("Cache not found, download will be suppported in future")
            return None
        if fix_missing_dates:
            cached_df = BigShitMarketDataLoader.fix_missing_dates(cached_df, trade_data_column_name, print_logs)
            fixcache_csv_path = os.path.join(dataset_dir, "bsmfix%s_%s.csv" % (data_start_date, data_end_date))
            cached_df.to_csv(fixcache_csv_path, encoding='utf-8')
        return cached_df

    def fix_missing_dates(in_df: pd.DataFrame, trade_data_column_name: str = 'date', verbose = True):
        df = in_df.copy()
        df.drop(columns = ['Unnamed: 0', 'index'], inplace=True)
        # 步骤1：生成所有股票和日期的完整组合
        all_stocks = df['tic'].unique()
        all_dates = df[trade_data_column_name].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=[trade_data_column_name, 'tic'])

        # 步骤2：创建完整组合的DataFrame
        df_complete = pd.DataFrame(index=full_index).reset_index()

        # 步骤3：左连接原始数据，标记缺失项
        merged = df_complete.merge(df, on=[trade_data_column_name, 'tic'], how='left', indicator=True)
        missing_rows = merged[merged['_merge'] == 'left_only'][[trade_data_column_name, 'tic']]
        merged.drop(columns=['_merge'], inplace=True)
        # merged.index = merged[trade_data_column_name].factorize()[0]

        return BigShitMarketDataLoader.fill_missing_dates(merged, missing_rows, verbose = verbose)

    def fill_missing_dates(in_df: pd.DataFrame, missing_rows: pd.DataFrame, trade_data_column_name: str = 'date', verbose = True):
        # 一些交易相关的值直接填0
        in_df.fillna({'open': 0, 'high': 0, 'low': 0, 'volume': 0, 'amount': 0, 'change_pct': 0, 'change': 0, 'turnover_ratio': 0, 'pre_close': 0}, inplace=True)
        # 补充时间相关的值
        in_df.iloc[missing_rows.index, in_df.columns.get_loc('trade_time')] = pd.to_datetime(in_df.iloc[missing_rows.index][trade_data_column_name], errors='coerce')
        missing_days = in_df.iloc[missing_rows.index, in_df.columns.get_loc('trade_time')].dt.dayofweek
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

    def load_cached_dataset(csv_file_path: str):
        if os.path.isfile(csv_file_path):
            return pd.read_csv(csv_file_path, encoding='utf-8', parse_dates=['trade_time'],
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
        # else:
        #     bsm_down = BigShitMarketDownloader(MY_TRAIN_START_DATE, MY_TRADE_END_DATE, CONCERNED_TICKET_LIST)
        #     df = bsm_down.fetch_data()
        #     df.to_csv(cache_csv_path, encoding='utf-8')
        return None