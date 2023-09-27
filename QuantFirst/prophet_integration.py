import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet


def convert_volume_expr(vo_str):
    if '' == vo_str:
        return 0
    split_vo = re.match('([\d.]+)([mMbB]+)', vo_str)
    val = split_vo.group(1)
    unit = split_vo.group(2) if len(split_vo.groups()) == 2 else ''
    if 'm' == unit or 'M' == unit:
        return float(val) * 10 ** 6
    elif 'b' == unit or 'B' == unit:
        return float(val) * 10 ** 9
    return 0


def convert_change_rate_expr(chgrt_str):
    if '' == chgrt_str:
        return 0
    split_vo = re.match('([\-\d.]+)(%)', chgrt_str)
    val = split_vo.group(1)
    return float(val) / 100


class StockProphet:

    def __init__(self):
        self.core_data = None
        self.train_set = None
        self.validation_set = None

    def setup_inspectee(self, csv_path, split_rate=0.8):
        source_data = pd.read_csv(csv_path, keep_default_na=False)
        # print(source_data.dtypes)
        self.core_data = pd.DataFrame()
        self.core_data['date'] = pd.to_datetime(source_data['日期'])
        self.core_data['open'] = source_data['开盘']
        self.core_data['close'] = source_data['收盘']
        self.core_data['low'] = source_data['低']
        self.core_data['high'] = source_data['高']
        self.core_data['volume'] = source_data['交易量'].apply(convert_volume_expr)
        self.core_data['change'] = source_data['涨跌幅'].apply(convert_change_rate_expr)
        self.core_data['ds'] = self.core_data['date']
        self.core_data['y'] = self.core_data['close']
        self.core_data.sort_values('date', inplace=True)
        self.core_data.reset_index(drop=True, inplace=True)
        data_len = len(self.core_data)
        spl_point = int(data_len * split_rate)
        self.train_set = self.core_data[:spl_point]
        self.validation_set = self.core_data[spl_point:]
        # self.data_open = self.train_set['开盘']
        # self.data_close = self.train_set['收盘']
        # self.data_min = self.train_set['低']
        # self.data_max = self.train_set['高']
        # self.data_volume = self.train_set['交易量']
        # self.begin_date = np.min(self.train_set['日期'])
        # self.end_date = np.max(self.train_set['日期'])

    def draw_data(self):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('date')
        ax1.set_ylabel('close')
        ax1.plot(self.core_data['date'], self.core_data['close'], color='blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('volume')
        ax2.bar(self.core_data['date'], self.core_data['volume'], color='red')
        plt.show()

    def simulate_automatic_investment_plan(self, sim_start_date='2015-01-01', sim_end_date='2023-09-25',
                                           invs_per_day=100):
        sim_start_date_dt = pd.to_datetime(sim_start_date)
        sim_end_date_dt = pd.to_datetime(sim_end_date)
        data_min_date = np.min(self.core_data['date'])
        data_max_date = np.max(self.core_data['date'])
        act_begin_date = sim_start_date_dt if data_min_date < sim_start_date_dt else data_min_date
        act_end_date = sim_end_date_dt if sim_end_date_dt < data_max_date else data_max_date
        mask = (self.core_data['date'] >= act_begin_date) & (self.core_data['date'] <= act_end_date)
        sim_ranges = self.core_data[mask]
        sim_ranges.reset_index(drop=True, inplace=True)
        sim_range_len = len(sim_ranges)
        profit_by_dates = np.zeros(sim_range_len)
        acc_profit_by_dates = np.zeros(sim_range_len)
        cost_by_dates = np.zeros(sim_range_len)
        bought_shares = 0
        acc_buy_cost = 0
        open_price = sim_ranges['open']
        close_price = sim_ranges['close']
        for index in range(0, sim_range_len):
            bought_shares += invs_per_day
            cost_by_dates[index] = invs_per_day * open_price[index]
            acc_buy_cost += cost_by_dates[index]
            profit_by_dates[index] = bought_shares * (close_price[index] - open_price[index])
            acc_profit_by_dates[index] = bought_shares * close_price[index] - cost_by_dates.sum()
        last_price = close_price[sim_range_len - 1]
        acc_profit = bought_shares * last_price - acc_buy_cost
        choose_p_color = np.vectorize(lambda prf: 'g' if prf < 0 else 'r')
        positivize_fun = np.vectorize(lambda x: x if x >= 0 else -x)
        profit_by_dates_draw_color = choose_p_color(acc_profit_by_dates)
        positivize_profit = positivize_fun(acc_profit_by_dates)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('date')
        ax1.set_ylabel('close')
        ax1.plot(sim_ranges['date'], sim_ranges['close'], color='blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('profit')
        ax2.bar(sim_ranges['date'], positivize_profit, color=profit_by_dates_draw_color)
        plt.show()

    def simulate_one_shot_investment(self, trigger_date='2015-01-01', shot_amount=100):
        return

    def modelling(self):
        # self.train_set['ds'] = self.train_set['date'].copy()
        # self.train_set['y'] = self.train_set['close'].copy()
        prop_model = Prophet()
        prop_model.fit(self.train_set)
        future = prop_model.make_future_dataframe(periods=0, freq='D')
        future_values = prop_model.predict(future)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.train_set['ds'], self.train_set['y'], 'ko-', linewidth=1.4, alpha=0.8, ms=1.8, label='Observations')
        ax.plot(future_values['ds'], future_values['yhat'], 'forestgreen', linewidth=2.4, label='Modeled')
        ax.fill_between(future_values['ds'].dt.to_pydatetime(), future_values['yhat_upper'], future_values['yhat_lower'], alpha=0.3, facecolor='g', edgecolor='k', linewidth=1.4, label='Confidence Interval')
        plt.show()
