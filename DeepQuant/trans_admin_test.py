from src.assistants.transaction_records import TransactionRecordAdministrator
import pandas as pd

if __name__ == '__main__':
    east_mon_trans = pd.read_csv('trades_data/Table_7540.csv')
    trans_admin = TransactionRecordAdministrator(east_mon_trans)
    print(trans_admin.transaction_records.head())