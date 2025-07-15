import pandas as pd

if __name__ == '__main__':
    winrate_df = pd.read_csv('eggs.csv')
    winrate_series = winrate_df['wintimes'] / winrate_df['optimes']
    pcr_series = winrate_df['profitcossratio']
    print(winrate_series.mean())
    print(pcr_series.mean())