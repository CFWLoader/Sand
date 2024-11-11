import pandas as pd

class TradeDataframeAccessor:
    def __init__(self, df: pd.DataFrame, use_copy = True):
        self.shadow_df = df.copy() if use_copy else df

    def data_len(self) -> int:
        return len(self.shadow_df)

    def get_close_prices(self) -> pd.Series:
        return self.shadow_df['close']

    def get_trade_dates(self) -> pd.Series:
        return self.shadow_df['trade_date']