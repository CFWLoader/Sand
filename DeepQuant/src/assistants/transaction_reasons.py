import pandas as pd
from src.bigshitmarketdataloader import BigShitMarketDataLoader
from stockstats import StockDataFrame as Sdf


class TransactionReasonFinder:
    def __init__(self, transaction_records: pd.DataFrame, data_cache_dir: str):
        self.transaction_records = transaction_records
        self.data_cache_dir = data_cache_dir
        self.market_data_loader = None
        self.market_data = None

    def try_justify_transactions(self) -> pd.DataFrame:
        # 提取code列的唯一值
        unique_codes = self.transaction_records['code'].unique()
        
        # 提取trade_datetime列的最大值和最小值
        min_datetime = self.transaction_records['trade_datetime'].min()
        max_datetime = self.transaction_records['trade_datetime'].max()
        
        # 将最小值减去三年并格式化
        min_datetime_minus_three_years = (min_datetime - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        
        # 将最大值格式化
        max_datetime_formatted = max_datetime.strftime('%Y-%m-%d')

        self.market_data_loader = BigShitMarketDataLoader(self.data_cache_dir, min_datetime_minus_three_years, max_datetime_formatted, unique_codes, print_logs=False)
        self.market_data = self.market_data_loader.load()
        for tick_code in unique_codes:
            self.find_transaction_reasons_for_ticket(tick_code)
        return pd.DataFrame()

    def find_transaction_reasons_for_ticket(self, tick_code: str) -> pd.DataFrame:
        print('Finding transaction reasons for %s' % tick_code)
        tick_trans = self.transaction_records[self.transaction_records['code'] == tick_code]
        tick_k_data = self.market_data[self.market_data['tic'] == tick_code]
        if tick_k_data.empty:
            print(f'No data for {tick_code}')
            return pd.DataFrame()
        tick_k_data.sort_values(by=["trade_time"], inplace=True)
        stock_df = Sdf.retype(tick_k_data)
        tick_k_data['boll'] = stock_df['boll']
        tick_k_data['boll_ub'] = stock_df['boll_ub']
        tick_k_data['boll_lb'] = stock_df['boll_lb']
        # 日线操作理由采用先用5日线与20日线作为依据
        self.generate_ma_breakthrough_signals(tick_k_data, stock_df)
        for index, tran_row in tick_trans.iterrows():
            single_trans_reasons: list[str] = []
            transaction_date = tran_row['trade_date']
            transaction_date = pd.to_datetime(transaction_date)
            k_data_row = tick_k_data[tick_k_data['trade_time'] == transaction_date]
            high_price = k_data_row['high'].values[0]
            low_price = k_data_row['low'].values[0]
            boll_ub_value = k_data_row['boll_ub'].values[0]
            boll_lb_value = k_data_row['boll_lb'].values[0]
            if high_price > boll_ub_value:
                single_trans_reasons.append(f'高位价格%f突破上轨%f' % (high_price,  boll_ub_value))
            # 先硬规定突破上轨或下轨，后面再考虑怎么给一个允许区间，否则基本上每笔交易都有正当理由
            # elif high_price < boll_ub_value and high_price > boll_value:
            #     single_trans_reasons.append(f'高位价格%f在上轨区间[%f-%f]' % (high_price, boll_value, boll_ub_value))

            if low_price < boll_lb_value:
                single_trans_reasons.append(f'低位价格%f突破下轨%f' % (low_price,  boll_lb_value))
            # elif low_price > boll_lb_value and low_price < boll_value:
            #     single_trans_reasons.append(f'低位价格%f在下轨区间[%f-%f]' % (low_price, boll_lb_value, boll_value))

            # 从tick_k_data中筛选出trade_time列日期在transaction_date天前到transaction_date的行数据
            date_range_mask = (tick_k_data['trade_time'] <= transaction_date) & (tick_k_data['trade_time'] >= (transaction_date + pd.tseries.offsets.BusinessDay(n=-10)))
            filtered_data = tick_k_data[date_range_mask]

            # 检查ma5_up_break和ma5_low_break列是否有值为1
            up_break_rows = filtered_data[filtered_data['ma5_up_break'] == 1]
            low_break_rows = filtered_data[filtered_data['ma5_low_break'] == 1]

            # 提取ma5和ma20列的值
            if not up_break_rows.empty:
                for _, row in up_break_rows.iterrows():
                    single_trans_reasons.append(f"{row['trade_time']} MA5突破MA20: MA5={row['ma5']}, MA20={row['ma20']}")
            if not low_break_rows.empty:
                for _, row in low_break_rows.iterrows():
                    single_trans_reasons.append(f"{row['trade_time']} MA5跌破MA20: MA5={row['ma5']}, MA20={row['ma20']}")

            print(single_trans_reasons)
        # if tick_trans['direction'] == 'buy':
        #     pass
        # elif tick_trans['direction'] == 'sell':
        #     pass
        return pd.DataFrame()

    def generate_ma_breakthrough_signals(self, input_df: pd.DataFrame, indicator_df: Sdf) -> None:
        input_df['ma5'] = indicator_df['close_5_sma']
        input_df['ma20'] = indicator_df['close_20_sma']
        # 新增buyin_signal列
        input_df['ma5_up_break'] = 0
        buyin_mask = input_df['ma5'].notna() & input_df['ma20'].notna() & (input_df['ma20'] <= input_df['ma5'])
        input_df.loc[buyin_mask, 'ma5_up_break'] = 1
        # 新增sellout_signal列
        input_df['ma5_low_break'] = 0
        sellout_mask = input_df['ma5'].notna() & input_df['ma20'].notna() & (input_df['ma5'] <= input_df['ma20'])
        input_df.loc[sellout_mask, 'ma5_low_break'] = 1

        # 处理连续的1区间，仅保留第一个1
        def process_signal_column(col):
            # 标记连续1区间的变化
            col_diff = col.diff().ne(0).cumsum()
            # 在每个连续区间中，仅保留第一个1
            return col.where(col.groupby(col_diff).cumcount() == 0, 0)

        input_df['ma5_up_break'] = process_signal_column(input_df['ma5_up_break'])
        input_df['ma5_low_break'] = process_signal_column(input_df['ma5_low_break'])