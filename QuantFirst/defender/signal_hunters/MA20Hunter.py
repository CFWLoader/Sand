from defender.signal_hunters.SignalHunter import SignalHunter
import pandas as pd

class MA20Hunter(SignalHunter):
    def __init__(self, df_to_analysis: pd.DataFrame):
        super().__init__(df_to_analysis)
        self.ma20trend = None

    def is_valid(self):
        return super().is_valid() and (self.data_accessor.data_len() >= 20)

    def begin_analyze(self)->(list, list) :
        self.ma20trend = self.data_accessor.get_close_prices().rolling(20).mean()
        # date_col = self.shadow_df['trade_date']
        cls_price = self.data_accessor.get_close_prices()
        begin_idx = 0
        upping_flag = False
        downing_flag = False
        buyin_idx = []
        sellout_idx = []
        for iter_idx in range(0, self.data_accessor.data_len()):
            if not pd.isna(self.ma20trend[iter_idx]):
                begin_idx = iter_idx
                break

        for iter_idx in range(begin_idx, self.data_accessor.data_len()):
            if cls_price[iter_idx] >= self.ma20trend[iter_idx] and not upping_flag:
                buyin_idx.append(iter_idx)
                upping_flag = True
                if downing_flag:
                    downing_flag = False
                # print('%s up %f %f, buy in time' % (date_col[iter_idx], cls_price[iter_idx], self.ma20trend[iter_idx]))

            if cls_price[iter_idx] <= self.ma20trend[iter_idx] and not downing_flag:
                sellout_idx.append(iter_idx)
                downing_flag = True
                if upping_flag:
                    upping_flag = False
                # print('%s up %f %f, sell out time' % (date_col[iter_idx], cls_price[iter_idx], self.ma20trend[iter_idx]))

        return buyin_idx, sellout_idx