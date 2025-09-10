from src.assistants.transaction_records import TransactionRecordAdministrator, TransactionDataSources
from os import path
import pandas as pd

if __name__ == "__main__":
    east_trans = pd.read_csv("trades_data/Table_7540.csv")
    final_output_str = 'trades_data/Transactions_7540.csv'
    trans_admin = TransactionRecordAdministrator(east_trans, TransactionDataSources.EAST_MONEY, path.join('.', 'trans_admin_workspace'))
    trans_admin.try_justify_transactions()
    reasons_view = trans_admin.get_justified_transactions_view()
    reasons_view.to_csv(final_output_str, encoding='utf-8', index=False)
    print(len(trans_admin.transaction_records))
    print(len(reasons_view))