from src.quant.signals.bsm_signal_hunter import BSMSignalHunter
import pandas as pd
from stockstats import StockDataFrame as Sdf

class MACDHunter(BSMSignalHunter):
    """
    MACDHunter类用于分析MACD指标信号，生成买入和卖出信号。目前仅实现金叉死叉产生的信号。

    Attributes:
        indicated_df (pd.DataFrame): 存储分析结果的DataFrame。
    """

    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.indicated_df = None

    def get_signals(self) -> pd.DataFrame:
        if self.indicated_df is not None:
            return self.indicated_df[['date', 'buyin', 'sellout']]
        stock_df = Sdf.retype(self.data_accessor)
        stock_macd = stock_df['macd']
        stock_macd_signal = stock_df['macds']
        self.indicated_df = self.data_accessor.copy()
        self.indicated_df['macd_dif'] = stock_macd
        self.indicated_df['macd_dea'] = stock_macd_signal
        # 新增buyin_signal列
        self.indicated_df['buyin'] = 0
        # 新增sellout_signal列
        self.indicated_df['sellout'] = 0

        # 计算金叉和死叉条件
        buyin_mask = (
                (self.indicated_df['macd_dif'].shift(1) < self.indicated_df['macd_dea'].shift(1)) &
                (self.indicated_df['macd_dif'] > self.indicated_df['macd_dea'])
        )
        sellout_mask = (
                (self.indicated_df['macd_dif'].shift(1) > self.indicated_df['macd_dea'].shift(1)) &
                (self.indicated_df['macd_dif'] < self.indicated_df['macd_dea'])
        )
        self.indicated_df.loc[buyin_mask, 'buyin'] = 1
        self.indicated_df.loc[sellout_mask, 'sellout'] = 1

        self.indicated_df.sort_values(by=["date"], inplace=True)
        self.data_accessor.reset_index(inplace=True)
        self.indicated_df.reset_index(inplace=True)
        return self.indicated_df[['date', 'buyin', 'sellout']]

    def get_buyin_price_col_name(self) -> str:
        return 'close'

    def get_sellout_price_col_name(self) -> str:
        return 'close'