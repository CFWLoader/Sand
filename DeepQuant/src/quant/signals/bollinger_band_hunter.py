from src.quant.signals.bsm_signal_hunter import BSMSignalHunter
import pandas as pd
from stockstats import StockDataFrame as Sdf

class BollingerBandHunter(BSMSignalHunter):
    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.indicated_df = None

    def get_signals(self) -> pd.DataFrame:
        stock_df = Sdf.retype(self.data_accessor)
        boll_ub = stock_df['boll_ub']
        boll_lb = stock_df['boll_lb']
        self.indicated_df = self.data_accessor.copy()
        self.indicated_df['boll_ub'] = boll_ub
        self.indicated_df['boll_lb'] = boll_lb
        # 新增buyin_signal列
        self.indicated_df['buyin'] = 0
        buyin_mask = self.indicated_df['boll_lb'].notna() & (self.indicated_df['low'] <= self.indicated_df['boll_lb'])
        self.indicated_df.loc[buyin_mask, 'buyin'] = 1
        # 新增sellout_signal列
        self.indicated_df['sellout'] = 0
        sellout_mask = self.indicated_df['boll_ub'].notna() & (self.indicated_df['high'] >= self.indicated_df['boll_ub'])
        self.indicated_df.loc[sellout_mask, 'sellout'] = 1
        self.indicated_df.sort_values(by=["date"], inplace=True)
        self.data_accessor.reset_index(inplace=True)
        self.indicated_df.reset_index(inplace=True)
        return self.indicated_df[['date', 'buyin', 'sellout']]