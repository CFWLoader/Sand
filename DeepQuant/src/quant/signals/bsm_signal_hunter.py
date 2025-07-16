import pandas as pd

class BSMSignalHunter(object):
    def __init__(self, df_to_analysis: pd.DataFrame):
        self.data_accessor = df_to_analysis.copy()
        self.cached_result = None

    def get_signals(self) -> pd.DataFrame:
        pass

    @staticmethod
    def calculate_win_rate(day_prices: pd.Series, date_str_pst: pd.Series, op_signals: pd.DataFrame) -> (int, int, float, list):
        """
        :param date_str_pst: prices of stock in days
        :param buyin_indicies: buy in day presented by index of day prices
        :param sellout_indices: sell out day presented by index of day prices
        :return: operation times, win times,
        """
        byidx = 0
        slidx = 0
        optimes = 0
        wintimes = 0
        plcross = 0
        op_records = []
        # for index, row in op_signals.iterrows():
        # while True:
        #     if byidx >= byopslen or slidx >= slopslen:
        #         break
        #     buydate = buyin_indicies[byidx]
        #     selldate = sellout_indices[slidx]
        #     if selldate < buydate:
        #         slidx += 1
        #         if slidx >= slopslen:
        #             break
        #         selldate = sellout_indices[slidx]
        #     optimes += 1
        #     byidx += 1
        #     slidx += 1
        #     prc_diff = day_prices[selldate] - day_prices[buydate]
        #     plcross += prc_diff
        #     if prc_diff > 0:
        #         wintimes += 1
        #     op_records.append(
        #         (day_prices[buydate], date_str_pst[buydate], day_prices[selldate], date_str_pst[selldate], prc_diff))
            # print('op B: %f(%s) -> S: %f(%s), PRC diff: %f' % (
            # day_prices[buydate], date_str_pst[buydate], day_prices[selldate], date_str_pst[selldate], prc_diff))
        return optimes, wintimes, plcross, op_records

    def get_trade_summary_of_strategy(self, force_refresh = False) -> (pd.DataFrame, (int, int, float, list)):
        """
        get summary of this strategy
        :return: (buy in index of this data, sell out index of this data), (see comment of calculate_win_rate())
        """
        if (not force_refresh) and (self.cached_result is not None):
            return self.cached_result
        op_signals = self.get_signals()
        date_col = self.data_accessor['date']
        cls_price = self.data_accessor['close']
        self.cached_result = op_signals, self.calculate_win_rate(cls_price, date_col, op_signals)
        return self.cached_result
