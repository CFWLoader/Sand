import pandas as pd
from src.bigshitmarketdataloader import BigShitMarketDataLoader

class TransactionReasonFinder(object):
    def __init__(self, transaction_records: pd.DataFrame):
        self.transaction_records = transaction_records
        self.market_data_loader = None

    def try_justify_transactions(self) -> pd.DataFrame:
        return pd.DataFrame()