import adata
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # res_df = adata.stock.info.all_code()
    # print(res_df)

    res_df = adata.stock.market.get_market(stock_code='300014', k_type=1, start_date='2021-01-01')
    # print(res_df['close'])
    plt.plot(res_df['close'])
    plt.show()