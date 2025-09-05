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
        tick_trans = self.transaction_records[self.transaction_records['code'] == tick_code]
        tick_k_data = self.market_data[self.market_data['tic'] == tick_code]
        stock_df = Sdf.retype(tick_k_data)
        tick_k_data['boll'] = stock_df['boll']
        tick_k_data['boll_ub'] = stock_df['boll_ub']
        tick_k_data['boll_lb'] = stock_df['boll_lb']
        for index, tran_row in tick_trans.iterrows():
            single_trans_reasons: list[str] = []
            transaction_date = tran_row['trade_date']
            k_data_row = tick_k_data[tick_k_data['trade_time'] == transaction_date]
            high_price = k_data_row['high']
            low_price = k_data_row['low']
            boll_ub_value = k_data_row['boll_ub']
            boll_lb_value = k_data_row['boll_lb']
            boll_value = k_data_row['boll']
            if high_price > boll_ub_value:
                single_trans_reasons.append(f'高位价格%f突破上轨%f' % (high_price,  boll_ub_value))
            elif high_price < boll_ub_value and high_price > boll_value:
                single_trans_reasons.append(f'高位价格%f在上轨上轨区间[%f-%f]' % (high_price, boll_value, boll_ub_value))

            if low_price > boll_lb_value:
                single_trans_reasons.append(f'低位价格%f突破下轨%f' % (low_price,  boll_lb_value))
            elif low_price > boll_lb_value and low_price > boll_value:
                single_trans_reasons.append(f'低位价格%f在下轨下轨区间[%f-%f]' % (low_price, boll_lb_value, boll_value))

            print(single_trans_reasons)
        # if tick_trans['direction'] == 'buy':
        #     pass
        # elif tick_trans['direction'] == 'sell':
        #     pass
        return pd.DataFrame()