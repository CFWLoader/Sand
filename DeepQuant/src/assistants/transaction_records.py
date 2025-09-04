from enum import Enum
import pandas as pd

class TransactionDataSources(Enum):
    EAST_MONEY = 1
    FUTU = 2

class TransactionRecordAdministrator(object):
    """ 交易记录管理员
    腾讯文档不支持个人开发者注册，目前只能生成
    """
    def __init__(self, transaction_record: pd.DataFrame, raw_data_source: TransactionDataSources = TransactionDataSources.EAST_MONEY):
        self.transaction_records = self.unify_transaction_data(transaction_record, raw_data_source)

#region 交易数据格式统一
    @staticmethod
    def unify_transaction_data(transaction_records: pd.DataFrame, data_source: TransactionDataSources) -> pd.DataFrame:
        if data_source == TransactionDataSources.EAST_MONEY:
            return TransactionRecordAdministrator.unify_transaction_data_from_east_money(transaction_records)
        elif data_source == TransactionDataSources.FUTU:
            return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def unify_transaction_data_from_east_money(transaction_records: pd.DataFrame) -> pd.DataFrame:
        # 提取需要的列并重命名
        unified_df = transaction_records.copy()
        unified_df.columns = unified_df.columns.str.strip()
        unified_df = unified_df[[
            '成交日期', '成交时间', '证券代码', '证券名称', '成交数量', '委托方向', '成交金额', '成交均价', '股份余额'
        ]].rename(columns={
            '成交日期': 'trade_date',
            '成交时间': 'trade_time',
            '证券代码': 'code',
            '证券名称': 'stock_name',
            '成交数量': 'volume',
            '委托方向': 'direction',
            '成交金额': 'amount',
            '成交均价': 'price',
            '股份余额': 'holds'
        })

        # 对可能为字符串类型的列去除空格
        string_columns = ['code', 'direction']
        for col in string_columns:
            if col in unified_df.columns and unified_df[col].dtype == 'object':
                unified_df[col] = unified_df[col].str.strip()

        # 特殊处理code列，仅保留数字
        if unified_df['code'].dtype == 'object':
            unified_df['code'] = unified_df['code'].str.replace(r'[^0-9]', '', regex=True)

        # 处理direction列，映射为buy或sell
        if unified_df['direction'].dtype == 'object':
            unified_df['direction'] = unified_df['direction'].replace({
                '证券买入': 'buy',
                '本方卖出': 'sell',
                '证券卖出': 'sell'
            })

        # 合并trade_date和trade_time为trade_datetime列
        unified_df['trade_datetime'] = pd.to_datetime(unified_df['trade_date'] + ' ' + unified_df['trade_time'])
        # 按照trade_datetime列递增排序
        unified_df.sort_values(by='trade_datetime', inplace=True)
        unified_df.reset_index(inplace=True, drop=True)
        return unified_df
#endregion