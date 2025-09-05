from src.assistants.transaction_records import TransactionRecordAdministrator, TransactionDataSources
from os import path
import pandas as pd

if __name__ == "__main__":
    east_trans = pd.read_csv("trades_data/Table_7540.csv")
    trans_admin = TransactionRecordAdministrator(east_trans, TransactionDataSources.EAST_MONEY, path.join('.', 'trans_admin_workspace'))
    reasons = trans_admin.try_justify_transactions()