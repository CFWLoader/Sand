from enum import Enum
from typing import Hashable

import pandas as pd

class QuantTrader(object):

    # BUY_IN_STRATEGIES = Enum('BuyinStrategies', [
    #     ('ALL', 'all'), # 每次操作都执行最大买入(当前资金)
    # ])

    class BuyInStrategy(Enum):
        ALL = 'all'

    # SELL_OUT_STRATEGIES = Enum('SelloutStrategies', [
    #     ('ALL', 'all'), # 每次操作都执行全部卖出
    # ])

    class SellOutStrategy(Enum):
        ALL = 'all'

    QUANTITY_STR = 'quantity'
    FUND_STR = 'fund'
    ESTIMATED_ASSET_STR = 'total_asset'

    def __init__(
            self,
            initial_asset: float,
            buyin_strategy: BuyInStrategy,
            sellout_strategy: SellOutStrategy,
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
        # 开仓，清仓记录，结构为（开仓对应的dataframe索引，清仓对应的dataframe索引）
        self.last_open_close_record = -1
        self.open_close_records = None

    def simulate(self, signaled_stock_data: pd.DataFrame):
        self.last_open_close_record = -1
        self.open_close_records = []
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
                self.asset_record.loc[index, QuantTrader.QUANTITY_STR] = prev_row[QuantTrader.QUANTITY_STR]
                self.asset_record.loc[index, QuantTrader.FUND_STR] = prev_row[QuantTrader.FUND_STR]
                if sellout_sig:
                    self.do_sellout(index, row)
                if buyin_sig:
                    self.do_buyin(index, row)
                self.asset_record.loc[index, QuantTrader.ESTIMATED_ASSET_STR] = self.asset_record.loc[index, QuantTrader.FUND_STR] + self.asset_record.loc[index, QuantTrader.QUANTITY_STR] * row['close']
            prev_row = self.asset_record.iloc[index]
        return self.asset_record

    def do_sellout(self, row_idx: Hashable, row_data: pd.Series):
        previous_holds: int = self.asset_record.loc[row_idx, QuantTrader.QUANTITY_STR]
        fund: float = self.asset_record.loc[row_idx, QuantTrader.FUND_STR]
        if previous_holds <= 0:
            return
        if self.sellout_strategy == QuantTrader.SellOutStrategy.ALL:
            sellout_price_value: float = row_data[self.sellout_price]
            sellout_total = sellout_price_value * previous_holds
            self.asset_record.loc[row_idx, QuantTrader.QUANTITY_STR] = 0
            self.asset_record.loc[row_idx, QuantTrader.FUND_STR] = fund + sellout_total
            if self.last_open_close_record > 0:
                self.open_close_records.append((self.last_open_close_record, row_idx))
                self.last_open_close_record = -1
            else:
                record_date = row_data['date']
                print(f'No Last open record for {row_idx}=({record_date})')

    def do_buyin(self, row_idx: Hashable, row_data: pd.Series):
        previous_holds: int = self.asset_record.loc[row_idx, QuantTrader.QUANTITY_STR]
        fund: float = self.asset_record.loc[row_idx, QuantTrader.FUND_STR]
        if self.buyin_strategy == QuantTrader.BuyInStrategy.ALL:
            buyin_price_value: float = row_data[self.buyin_price]
            avail_shares = fund // buyin_price_value
            avail_lots = avail_shares // self.lot_size
            final_buyin_shares = avail_lots * self.lot_size
            if final_buyin_shares > 0:
                cost_current = final_buyin_shares * buyin_price_value
                self.asset_record.loc[row_idx, QuantTrader.QUANTITY_STR] = previous_holds + final_buyin_shares
                self.asset_record.loc[row_idx, QuantTrader.FUND_STR] = fund - cost_current
                # self.open_close_records.append((row_idx, 1))
                if self.last_open_close_record < 0:
                    self.last_open_close_record = row_idx

    def calculate_profit_loss_ratio(self, abandon_unclosed=False) -> float:
        """
        计算盈亏比
        根据前面模拟的结果中asset_record即开仓清仓记录open_close_records计算
        算法过程为遍历open_close_records的二元组，第一个参数是asset_record的索引，第二个参数是开仓清仓（1=开仓，0=清仓）
        open_close_records中的记录必然为先开仓再清仓，对于这种成对的记录，取索引访问asset_record的QuantTrader.QUANTITY_STR列，取每次的差值，记录这些差值对
        最后正的差值与负的差值各自求和，正的差值和负的差值（绝对值）的比值即为盈亏比
        :param abandon_unclosed: 是否放弃未清仓的记录，一般模拟最后可能出现未清仓的情况。默认为False即认为假设最后一天清仓，True则放弃最后一次开仓
        :return: 盈亏比
        """
        profit_sum = 0
        loss_sum = 0
        for (op_idx, cls_idx) in self.open_close_records:
            open_quantity = self.asset_record.loc[op_idx, QuantTrader.ESTIMATED_ASSET_STR]
            close_quantity = self.asset_record.loc[cls_idx, QuantTrader.ESTIMATED_ASSET_STR]
            est_asset_diff = close_quantity - open_quantity
            if est_asset_diff > 0:
                profit_sum += est_asset_diff
            else:
                loss_sum += abs(est_asset_diff)
        if self.last_open_close_record > 0 and not abandon_unclosed:
            last_open_asset = self.asset_record.loc[self.last_open_close_record, QuantTrader.ESTIMATED_ASSET_STR]
            last_close_est_asset = self.asset_record.loc[self.asset_record.index[-1], QuantTrader.ESTIMATED_ASSET_STR]
            est_asset_diff = last_close_est_asset - last_open_asset
            if est_asset_diff > 0:
                profit_sum += est_asset_diff
            else:
                loss_sum += abs(est_asset_diff)
        return profit_sum / loss_sum if loss_sum > 0 else profit_sum

    def sum_up_open_close_records(self, abandon_unclosed=False) -> pd.DataFrame:
        """
        计算开仓清仓记录的总和
        :return: 开仓清仓记录的总和
        """
        op_date_col = []
        cls_date_col = []
        est_asset_diffs = []
        for (op_idx, cls_idx) in self.open_close_records:
            open_quantity = self.asset_record.loc[op_idx, QuantTrader.ESTIMATED_ASSET_STR]
            close_quantity = self.asset_record.loc[cls_idx, QuantTrader.ESTIMATED_ASSET_STR]
            est_asset_diff = close_quantity - open_quantity
            op_date_col.append(self.asset_record.loc[op_idx, 'date'])
            cls_date_col.append(self.asset_record.loc[cls_idx, 'date'])
            est_asset_diffs.append(est_asset_diff)
        if self.last_open_close_record > 0 and not abandon_unclosed:
            last_open_asset = self.asset_record.loc[self.last_open_close_record, QuantTrader.ESTIMATED_ASSET_STR]
            last_close_est_asset = self.asset_record.loc[self.asset_record.index[-1], QuantTrader.ESTIMATED_ASSET_STR]
            est_asset_diff = last_close_est_asset - last_open_asset
            op_date_col.append(self.asset_record.loc[self.last_open_close_record, 'date'])
            cls_date_col.append(self.asset_record.loc[self.asset_record.index[-1], 'date'])
            est_asset_diffs.append(est_asset_diff)
        return pd.DataFrame({
            'open_date': op_date_col,
            'close_date': cls_date_col,
            'asset_changes': est_asset_diffs
        })