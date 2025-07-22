from src.quant.signals.bsm_signal_hunter import BSMSignalHunter
import pandas as pd
from stockstats import StockDataFrame as Sdf

class MA20Hunter(BSMSignalHunter):
    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.indicated_df = None

    # def get_signals(self)-> pd.DataFrame :
    #     self.ma20trend = self.data_accessor.get_close_prices().rolling(20).mean()
    #     # date_col = self.shadow_df['trade_date']
    #     cls_price = self.data_accessor.get_close_prices()
    #     begin_idx = 0
    #     upping_flag = False
    #     downing_flag = False
    #     buyin_idx = []
    #     sellout_idx = []
    #     for iter_idx in range(0, self.data_accessor.data_len()):
    #         if not pd.isna(self.ma20trend[iter_idx]):
    #             begin_idx = iter_idx
    #             break
    #
    #     for iter_idx in range(begin_idx, self.data_accessor.data_len()):
    #         if cls_price[iter_idx] >= self.ma20trend[iter_idx] and not upping_flag:
    #             buyin_idx.append(iter_idx)
    #             upping_flag = True
    #             if downing_flag:
    #                 downing_flag = False
    #             # print('%s up %f %f, buy in time' % (date_col[iter_idx], cls_price[iter_idx], self.ma20trend[iter_idx]))
    #
    #         if cls_price[iter_idx] <= self.ma20trend[iter_idx] and not downing_flag:
    #             sellout_idx.append(iter_idx)
    #             downing_flag = True
    #             if upping_flag:
    #                 upping_flag = False
    #             # print('%s up %f %f, sell out time' % (date_col[iter_idx], cls_price[iter_idx], self.ma20trend[iter_idx]))
    #     # return buyin_idx, sellout_idx

    def get_signals(self) -> pd.DataFrame:
        if self.indicated_df is not None:
            return self.indicated_df[['date', 'buyin', 'sellout']]
        stock_df = Sdf.retype(self.data_accessor)
        close_sma_20 = stock_df['close_20_sma']
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
        return pd.DataFrame()