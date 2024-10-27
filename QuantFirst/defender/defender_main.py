from shutil import which

import adata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dtt
from defender.signal_hunters.MA20Hunter import MA20Hunter

def manual_way_plot():
    # 手动设置X轴方法
    res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2023-01-01', end_date='2023-12-31')
    date_col = res_df['trade_date']
    xdate = [dtt.datetime.strptime(d, '%Y-%m-%d').date() for d in date_col]
    cls_price = res_df['close']
    sig_hunter = MA20Hunter(res_df)
    buyidx, sellidx = sig_hunter.begin_analyze()
    plt.title('Simplest MA20 Trending Trade')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.plot(xdate, cls_price, label='Close')
    plt.plot(xdate, sig_hunter.ma20trend, label='ma20')
    buydates = [date_col[byidx] for byidx in buyidx]
    selldates = [date_col[slidx] for slidx in sellidx]
    plt.vlines(buydates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='green', label='Buy in')
    plt.vlines(selldates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='red', label='Sell out')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()

def auto_way_plot():
    # 自动设置X轴方法
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2023-01-01', end_date='2023-12-31')
    date_col = pd.to_datetime(res_df['trade_date'])
    cls_price = res_df['close']
    sig_hunter = MA20Hunter(res_df)
    buyidx, sellidx = sig_hunter.begin_analyze()
    buydates = [date_col[byidx] for byidx in buyidx]
    selldates = [date_col[slidx] for slidx in sellidx]
    plt.title('Simplest MA20 Trending Trade')
    plt.plot(date_col, cls_price, label='Close')
    plt.plot(date_col, sig_hunter.ma20trend, label='ma20')
    plt.vlines(buydates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='green', label='Buy in')
    plt.vlines(selldates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='red', label='Sell out')
    plt.gcf().autofmt_xdate()
    plt.legend()
    calculate_win_rate(cls_price, buyidx, sellidx)
    plt.show()

def calculate_win_rate(day_prices: pd.Series, buyin_indicies: list, sellout_indices: list):
    byidx = 0
    slidx = 0
    byopslen = len(buyin_indicies)
    slopslen = len(sellout_indices)
    optimes = 0
    wintimes = 0
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
        if day_prices[buydate] < day_prices[selldate]:
            wintimes += 1
    print("Op times: %d, Win times: %d, Win rate: %f" % (optimes, wintimes, wintimes / optimes))

if __name__ == '__main__':
    # res_df = adata.stock.info.all_code()
    # manual_way_plot()
    auto_way_plot()
