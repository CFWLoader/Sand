from enum import Enum
from typing import Hashable

import pandas as pd

class QuantTrader(object):

    BUY_IN_STRATEGIES = Enum('BuyinStrategies', [
        ('ALL', 'all'), # 每次操作都执行最大买入(当前资金)
    ])

    SELL_OUT_STRATEGIES = Enum('SelloutStrategies', [
        ('ALL', 'all'), # 每次操作都执行全部卖出
    ])

    QUANTITY_STR = 'quantity'
    FUND_STR = 'fund'
    ESTIMATED_ASSET_STR = 'total_asset'

    def __init__(
            self,
            initial_asset: float,
            buyin_strategy: BUY_IN_STRATEGIES,
            sellout_strategy: SELL_OUT_STRATEGIES,
            buyin_price: str = 'close',
            sellout_price: str = 'close',
            lot_size = 100
    ):
        self.initial_asset = initial_asset
        self.asset_record = None
        self.buyin_strategy = buyin_strategy
        self.sellout_strategy = sellout_strategy
        self.buyin_price = buyin_price
        self.sellout_price = sellout_price
        self.lot_size = lot_size

    def simulate(self, signaled_stock_data: pd.DataFrame):
        self.asset_record = pd.DataFrame(index=signaled_stock_data.index)
        self.asset_record['date'] = signaled_stock_data['date']
        self.asset_record[QuantTrader.QUANTITY_STR] = 0
        self.asset_record[QuantTrader.FUND_STR] = self.initial_asset
        self.asset_record[QuantTrader.ESTIMATED_ASSET_STR] = self.initial_asset
        prev_row = None
        for index, row in signaled_stock_data.iterrows():
            if prev_row is not None:
                buyin_sig = row['buyin'] > 0
                sellout_sig = row['sellout'] > 0
                self.asset_record.iloc[index][QuantTrader.QUANTITY_STR] = prev_row[QuantTrader.QUANTITY_STR]
                self.asset_record.iloc[index][QuantTrader.FUND_STR] = prev_row[QuantTrader.FUND_STR]
                if sellout_sig:
                    self.do_sellout(index, row)
                if buyin_sig:
                    self.do_buyin(index, row)
                self.asset_record.iloc[index][QuantTrader.ESTIMATED_ASSET_STR] = self.asset_record.iloc[index][QuantTrader.FUND_STR] + self.asset_record.iloc[index][QuantTrader.QUANTITY_STR] * row['close']
            prev_row = self.asset_record.iloc[index]
        return self.asset_record

    def do_sellout(self, row_idx: Hashable, row_data: pd.Series):
        previous_holds: int = self.asset_record.iloc[row_idx][QuantTrader.QUANTITY_STR]
        fund: float = self.asset_record.iloc[row_idx][QuantTrader.FUND_STR]
        if previous_holds <= 0:
            return
        if self.sellout_strategy == QuantTrader.SELL_OUT_STRATEGIES.ALL:
            sellout_price_value: float = row_data[self.sellout_price]
            sellout_total = sellout_price_value * previous_holds
            self.asset_record.iloc[row_idx][QuantTrader.QUANTITY_STR] = 0
            self.asset_record.iloc[row_idx][QuantTrader.FUND_STR] = fund + sellout_total

    def do_buyin(self, row_idx: Hashable, row_data: pd.Series):
        previous_holds: int = self.asset_record.iloc[row_idx][QuantTrader.QUANTITY_STR]
        fund: float = self.asset_record.iloc[row_idx][QuantTrader.FUND_STR]
        if self.buyin_strategy == QuantTrader.BUY_IN_STRATEGIES.ALL:
            buyin_price_value: float = row_data[self.buyin_price]
            avail_shares = fund // buyin_price_value
            avail_lots = avail_shares // self.lot_size
            final_buyin_shares = avail_lots * self.lot_size
            cost_current = final_buyin_shares * buyin_price_value
            self.asset_record.iloc[row_idx][QuantTrader.QUANTITY_STR] = previous_holds + final_buyin_shares
            self.asset_record.iloc[row_idx][QuantTrader.FUND_STR] = fund - cost_current