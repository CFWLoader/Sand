from enum import Enum
import pandas as pd

class QuantTrader(object):

    BUY_IN_STRATEGIES = Enum('BuyinStrategies', [
        ('ALL', 'all'), # 每次操作都执行最大买入(当前资金)
    ])

    SELL_OUT_STRATEGIES = Enum('SelloutStrategies', [
        ('ALL', 'all'), # 每次操作都执行全部卖出
    ])

    def __init__(self, initial_asset: float, buyin_strategy: BUY_IN_STRATEGIES, sellout_strategy: SELL_OUT_STRATEGIES):
        self.initial_asset = initial_asset
        self.asset_record = None
        self.buyin_strategy = buyin_strategy
        self.sellout_strategy = sellout_strategy

    def simulate(self, signaled_stock_data: pd.DataFrame):
        self.asset_record = pd.DataFrame(index=signaled_stock_data.index)
        self.asset_record['date'] = signaled_stock_data['date']
        self.asset_record['total_asset'] = self.initial_asset