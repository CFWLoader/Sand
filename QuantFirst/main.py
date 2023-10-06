# This is a sample Python script.
from classical_forecaster import arima_automata
from prophet_integration import StockProphet
from series_analyzer import EMA, sliding_window_sum, series_MA, series_dif, series_macd
import pandas as pd
import matplotlib.pyplot as plt

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # df0700 = pd.read_csv('./datasets/0700history.csv')
    # print(df0700)
    # df0700.plot(x='日期', y='收盘', kind='bar')
    # plt.plot(df0700['日期'], df0700['收盘'])
    # plt.show()
    # df0700close = df0700['收盘']
    # fast_ema = df0700close.ewm(span=12).mean()
    # slow_ema = df0700close.ewm(span=26).mean()
    # diff = fast_ema - slow_ema
    # dea = diff.ewm(span = 9).mean()
    # macd = 2 * (diff - dea)
    # print(macd.to_numpy())
    # mymacd = series_macd(df0700close.to_numpy())
    # print(mymacd)
    # arima_automata(df0700['收盘'])
    eve_inspector = StockProphet()
    eve_inspector.setup_inspectee('./QuantFirst/datasets/eve300014.csv')
    # eve_inspector.draw_data()
    # eve_inspector.simulate_automatic_investment_plan('2018-01-01', '2023-09-20')
    eve_inspector.modelling()
    # eve_inspector.plot_change_points()
