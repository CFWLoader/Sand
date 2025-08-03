from src.bigshitmarketdataloader import BigShitMarketDataLoader
from src.quant.signals.bollinger_band_hunter import BollingerBandHunter
import matplotlib

from src.quant.strategy_evaluator import StrategyEvaluator

matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

MY_TRAIN_START_DATE = '2015-01-01'
MY_TRAIN_END_DATE = '2025-01-01'
MY_TRADE_START_DATE = '2025-01-01'
MY_TRADE_END_DATE = '2025-07-10'

SHORT_SIM_DATE_START = '2024-07-01'
SHORT_SIM_DATE_END = '2025-07-31'

CONCERNED_TICKET_LIST=['300014', '600011', '000977', '000766', '002415', '600036' ]
# CONCERNED_TICKET_LIST=['300014', '600011', '000977']

if __name__ == '__main__':
    bsm_loader = BigShitMarketDataLoader('exp1', SHORT_SIM_DATE_START, SHORT_SIM_DATE_END, CONCERNED_TICKET_LIST)
    raw_data = bsm_loader.load_raw_data_only()

    strategy_eval = StrategyEvaluator(
        raw_data,
        'rsi',
        StrategyEvaluator.BUY_IN_STRATEGIES.ALL,
        StrategyEvaluator.SELL_OUT_STRATEGIES.ALL,
        buyin_price= 'close',
        sellout_price= 'close'
    )

    strategy_eval.evaluate_all()
    plr_set = strategy_eval.get_profit_and_loss_ratio()
    # for tic, plr in plr_set.items():
    #     print(f'tic: {tic} plr: {plr}')
    oc_reports = strategy_eval.get_open_close_reports()
    for tic, plr in oc_reports.items():
        print(f'tic: {tic}, plr={plr_set[tic]}')
        # print(plr)

    # result_df = strategy_eval.evaluate('300014')
    # print(result_df.tail(10))

    # winrates_sum = 0
    # for tic in CONCERNED_TICKET_LIST:
    #     bollinger_band_hunter = BollingerBandHunter(raw_data[raw_data.tic == tic])
    #     (buyin, sellout), (optimes, wintimes, plcross, op_records) = bollinger_band_hunter.get_trade_summary_of_strategy()
    #     winrates_sum += wintimes/optimes
    #     print(f'tic: {tic} optimes: {optimes}, wintimes: {wintimes}, win rate: {wintimes/optimes}, plcross: {plcross}')
    #
    # print(f'average win rate: {winrates_sum/len(CONCERNED_TICKET_LIST)}')

    # 绘制曲线图
    '''
    df = bollinger_band_hunter.indicated_df
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['open'], label='Open')
    plt.plot(df['date'], df['close'], label='Close')
    plt.plot(df['date'], df['high'], label='High')
    plt.plot(df['date'], df['low'], label='Low')
    plt.plot(df['date'], df['boll_ub'], label='Bollinger Upper Band')
    plt.plot(df['date'], df['boll_lb'], label='Bollinger Lower Band')

    # 调整横坐标显示间隔
    step = max(1, len(df) // 10)  # 每10个数据点显示一个标签
    plt.xticks(range(0, len(df), step), df['date'][::step], rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bollinger Bands and Price Data')
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''