from shutil import which

import adata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from defender.signal_hunters.MA20Hunter import MA20Hunter
from defender.signal_hunters.SignalHunter import SignalHunter


# def manual_way_plot():
#     # 手动设置X轴方法
#     res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2023-01-01', end_date='2023-12-31')
#     date_col = res_df['trade_date']
#     xdate = [dtt.datetime.strptime(d, '%Y-%m-%d').date() for d in date_col]
#     cls_price = res_df['close']
#     sig_hunter = MA20Hunter(res_df)
#     buyidx, sellidx = sig_hunter.begin_analyze()
#     plt.title('Simplest MA20 Trending Trade')
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
#     plt.plot(xdate, cls_price, label='Close')
#     plt.plot(xdate, sig_hunter.ma20trend, label='ma20')
#     buydates = [date_col[byidx] for byidx in buyidx]
#     selldates = [date_col[slidx] for slidx in sellidx]
#     plt.vlines(buydates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='green', label='Buy in')
#     plt.vlines(selldates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='red', label='Sell out')
#     plt.gcf().autofmt_xdate()
#     plt.legend()
#     plt.show()

def auto_way_plot(buyidx: list, sellidx: list, cls_price: pd.Series, date_col: pd.Series, extra_data: dict):
    # 自动设置X轴方法
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    # res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2021-01-01', end_date='2021-12-31')
    # res_df = adata.stock.market.get_market(stock_code='60001', k_type=1, start_date='2020-01-01')
    # date_col = pd.to_datetime(res_df['trade_date'])
    # cls_price = res_df['close']
    # sig_hunter = MA20Hunter(res_df)
    # (buyidx, sellidx), (optimes, wintimes, plcross, op_records)  = sig_hunter.get_trade_summary_of_strategy()
    buydates = [date_col[byidx] for byidx in buyidx]
    selldates = [date_col[slidx] for slidx in sellidx]
    plt.title('Simplest MA20 Trending Trade')
    plt.plot(date_col, cls_price, label='Close')
    for k,v in extra_data.items():
        plt.plot(date_col, v, label=k)
    plt.vlines(buydates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='green', label='Buy in')
    plt.vlines(selldates, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='red', label='Sell out')
    plt.gcf().autofmt_xdate()
    plt.legend()

    plt.show()

def take_test(stk_code: str, start_date: str, plot_diagram: bool = False, print_ops: bool = False) -> SignalHunter:
    res_df = adata.stock.market.get_market(stock_code=stk_code, k_type=1, start_date=start_date)
    sig_hunter = MA20Hunter(res_df)
    if not sig_hunter.is_valid():
        return sig_hunter
    (buyidx, sellidx), (optimes, wintimes, plcross, op_records)  = sig_hunter.get_trade_summary_of_strategy()
    if print_ops:
        for trade_op in op_records:
            print('op B: %f(%s) -> S: %f(%s), PRC diff: %f' % (
            trade_op[0], trade_op[1], trade_op[2], trade_op[3], trade_op[4]))
    # print("Op times: %d, Win times: %d, Win rate: %f, P/C: %f" % (optimes, wintimes, wintimes / optimes, plcross))
    if plot_diagram:
        auto_way_plot(buyidx, sellidx, sig_hunter.data_accessor.get_close_prices(), pd.to_datetime(sig_hunter.data_accessor.get_trade_dates()), {
            'ma20': sig_hunter.ma20trend
        })
    return sig_hunter


if __name__ == '__main__':
    # res_df = adata.stock.info.all_code()
    # manual_way_plot()
    allastock = adata.stock.info.all_code()
    for index, row in allastock.iterrows():
        stk_code = row['stock_code']
        sht_name = row['short_name']
        exchange = row['exchange']
        sig_hunter = take_test(stk_code, '2020-01-01')
        if not sig_hunter.is_valid():
            print('[%s(%s)] has no data, skip' % (sht_name, stk_code))
            continue
        (buyidx, sellidx), (optimes, wintimes, plcross, op_records)  = sig_hunter.get_trade_summary_of_strategy()
        print("[%s(%s)]Op times: %d, Win times: %d, Win rate: %f, P/C: %f" % (sht_name, stk_code, optimes, wintimes, wintimes / optimes, plcross))
    # sig_hunter = take_test('000003', '2020-01-01')

