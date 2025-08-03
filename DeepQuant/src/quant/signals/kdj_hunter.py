from src.quant.signals.bsm_signal_hunter import BSMSignalHunter
import pandas as pd
from stockstats import StockDataFrame as Sdf

class KDJHunter(BSMSignalHunter):
    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.indicated_df = None

    def get_signals(self) -> pd.DataFrame:
        if self.indicated_df is not None:
            return self.indicated_df[['date', 'buyin', 'sellout']]
        stock_df = Sdf.retype(self.data_accessor)
        k_line = stock_df['kdjk']
        d_line = stock_df['kdjd']
        j_line = stock_df['kdjj']
        self.indicated_df = self.data_accessor.copy()
        self.indicated_df['kdjk'] = k_line
        self.indicated_df['kdjd'] = d_line
        self.indicated_df['kdjj'] = j_line
        # 新增buyin_signal列
        self.indicated_df['buyin'] = 0
        buyin_mask = (self.indicated_df['kdjk'] > self.indicated_df['kdjd']) & \
                     (self.indicated_df['kdjk'].shift(1) <= self.indicated_df['kdjd'].shift(1))
        self.indicated_df.loc[buyin_mask, 'buyin'] = 1
        # 新增sellout_signal列
        self.indicated_df['sellout'] = 0
        sellout_mask = (self.indicated_df['kdjk'] < self.indicated_df['kdjd']) & \
                       (self.indicated_df['kdjk'].shift(1) >= self.indicated_df['kdjd'].shift(1))
        self.indicated_df.loc[sellout_mask, 'sellout'] = 1
        self.indicated_df.sort_values(by=["date"], inplace=True)
        self.data_accessor.reset_index(inplace=True)
        self.indicated_df.reset_index(inplace=True)
        return self.indicated_df[['date', 'buyin', 'sellout']]

    def get_buyin_price_col_name(self) -> str:
        return 'close'

    def get_sellout_price_col_name(self) -> str:
        return 'close'