import adata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from defender.signal_hunters.MA20Hunter import MA20Hunter

if __name__ == '__main__':
    # res_df = adata.stock.info.all_code()
    # print(res_df)

    res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2024-01-01')
    date_col = res_df['trade_date']
    # datetime_col = mdates.datestr2num(date_col)
    # exit(0)
    cls_price = res_df['close']
    sig_hunter = MA20Hunter(res_df)
    buyidx, sellidx = sig_hunter.begin_analyze()
    plt.title('Simplest MA20 Trending Trade')
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.plot(cls_price, label='Close')
    plt.plot(sig_hunter.ma20trend, label='ma20')
    plt.vlines(buyidx, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='green', label='Buy in')
    plt.vlines(sellidx, ymin=cls_price.min(), ymax=cls_price.max(), ls=':', colors='red', label='Sell out')
    # print(res_df['close'])
    # res_df['MA5'] = res_df['close'].rolling(5).mean()
    # res_df['MA20'] = res_df['close'].rolling(20).mean()
    # plt.plot(res_df['close'])
    # plt.title('Price MA5, MA20')
    # plt.ion()
    # plt.plot()
    res_df.plot(x="trade_date", y=["close"])
    # plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
