from src.quant.signals.bsm_signal_hunter import BSMSignalHunter
import pandas as pd
from stockstats import StockDataFrame as Sdf

class BollingerBandHunter(BSMSignalHunter):
    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.indicated_df = None

    def get_signals(self) -> (list, list):
        stock_df = Sdf.retype(self.data_accessor)
        boll_ub = stock_df['boll_ub']
        boll_lb = stock_df['boll_lb']
        self.indicated_df = self.data_accessor.copy()
        # self.indicated_df = self.indicated_df.merge(
        #     self.indicated_df[["tic", "date", boll_ub]], on=["tic", "date"], how="left"
        # )
        # self.indicated_df.insert()
        # self.indicated_df.merge(boll_ub[["date", 'boll_ub']], on=["date"], how="left")
        # self.indicated_df.merge(boll_lb[["date", 'boll_lb']], on=["date"], how="left")
        self.indicated_df['boll_ub'] = boll_ub
        self.indicated_df['boll_lb'] = boll_lb
        self.indicated_df.sort_values(by=["date"], inplace=True)
        # self.indicated_df['boll_ub'] = self.indicated_df['boll_ub'].fillna(self.indicated_df['close'])
        # self.indicated_df['boll_lb'] = self.indicated_df['boll_lb'].fillna(self.indicated_df['close'])
        self.data_accessor.reset_index(inplace=True)
        self.indicated_df.reset_index(inplace=True)
        buyin_signals = self.indicated_df[self.indicated_df['boll_lb'].notna() & (self.indicated_df['low'] <= self.indicated_df['boll_lb'])]
        sellout_signals = self.indicated_df[self.indicated_df['boll_ub'].notna() & (self.indicated_df['high'] >= self.indicated_df['boll_ub'])]
        return buyin_signals.index, sellout_signals.index