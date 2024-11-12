import pandas as pd
from defender.TradeDataframeAccessor import TradeDataframeAccessor

class SignalHunter(object):
    def __init__(self, df_to_analysis: pd.DataFrame):
        self.data_accessor = TradeDataframeAccessor(df_to_analysis)
        self.cached_result = None

    def begin_analyze(self) -> (list, list):
        pass

    def is_valid(self):
        return self.data_accessor.is_valid()

    @staticmethod
    def calculate_win_rate(day_prices: pd.Series, date_str_pst: pd.Series, buyin_indicies: list,
                           sellout_indices: list) -> (int, int, float, list):
        """
        :param date_str_pst: prices of stock in days
        :param buyin_indicies: buy in day presented by index of day prices
        :param sellout_indices: sell out day presented by index of day prices
        :return: operation times, win times,
        """
        byidx = 0
        slidx = 0
        byopslen = len(buyin_indicies)
        slopslen = len(sellout_indices)
        optimes = 0
        wintimes = 0
        plcross = 0
        op_records = []
        while True:
            if byidx >= byopslen or slidx >= slopslen:
                break
            buydate = buyin_indicies[byidx]
            selldate = sellout_indices[slidx]
            if selldate < buydate:
                slidx += 1
                if slidx >= slopslen:
                    break
                selldate = sellout_indices[slidx]
            optimes += 1
            byidx += 1
            slidx += 1
            prc_diff = day_prices[selldate] - day_prices[buydate]
            plcross += prc_diff
            if prc_diff > 0:
                wintimes += 1
            op_records.append(
                (day_prices[buydate], date_str_pst[buydate], day_prices[selldate], date_str_pst[selldate], prc_diff))
            # print('op B: %f(%s) -> S: %f(%s), PRC diff: %f' % (
            # day_prices[buydate], date_str_pst[buydate], day_prices[selldate], date_str_pst[selldate], prc_diff))
        return optimes, wintimes, plcross, op_records

    def get_trade_summary_of_strategy(self, force_refresh = False) -> ((list, list), (int, int, float, list)):
        """
        get summary of this strategy
        :return: (buy in index of this data, sell out index of this data), (see comment of calculate_win_rate())
        """
        if (not force_refresh) and (self.cached_result is not None):
            return self.cached_result
        buyidx, sellidx = self.begin_analyze()
        date_col = self.data_accessor.get_trade_dates()
        cls_price = self.data_accessor.get_close_prices()
        self.cached_result = (buyidx, sellidx), self.calculate_win_rate(cls_price, date_col, buyidx, sellidx)
        return self.cached_result
