import pandas as pd

from src.quant.signals.bollinger_band_hunter import BollingerBandHunter
from src.quant.trader.quant_trader import QuantTrader


class StrategyEvaluator(object):

    TRADE_SIGNALS = {
        'boll': BollingerBandHunter
    }

    BUY_IN_STRATEGIES = QuantTrader.BuyInStrategy

    SELL_OUT_STRATEGIES = QuantTrader.SellOutStrategy

    def __init__(
            self,
            stock_data: pd.DataFrame,
            signal: str,
            buyin_strategy: BUY_IN_STRATEGIES,
            sellout_strategy: SELL_OUT_STRATEGIES,
            initial_asset: float = 500000.0,
            buyin_price: str = 'close',
            sellout_price: str = 'close',
            lot_size=100
    ):
        signal_clz = StrategyEvaluator.TRADE_SIGNALS[signal]
        if signal_clz is None:
            print(f'Invalid signal: {signal}')
            return
        self.signal_sources = {}
        self.traders = {}
        for tic_code in stock_data.tic:
            signal_source = signal_clz(stock_data[stock_data.tic == tic_code])
            self.signal_sources[tic_code] = signal_source
            self.traders[tic_code] = QuantTrader(
                initial_asset, buyin_strategy, sellout_strategy,
                buyin_price, sellout_price, lot_size
            )
        self.evaluate_report = None

    def evaluate(self, tic_code) -> pd.DataFrame:
        if self.evaluate_report is None:
            self.evaluate_report = {}
        if tic_code in self.signal_sources:
            signal_source = self.signal_sources[tic_code]
            trade_signals = signal_source.get_signals()
            evaluated = self.traders[tic_code].simulate(signal_source.indicated_df)
            self.evaluate_report[tic_code] = evaluated
        return pd.DataFrame()

    def evaluate_all(self):
        self.evaluate_report = {}
        for tic_code, signal_source in self.signal_sources.items():
            trade_signals = signal_source.get_signals()
            evaluated = self.traders[tic_code].simulate(signal_source.indicated_df)
            self.evaluate_report[tic_code] = evaluated

    def get_profit_and_loss_ratio(self) -> dict[str, float]:
        plr_dict = {}
        for tic_code, eval_report in self.evaluate_report.items():
            # print(f'PLR for {tic_code}:')
            trader = self.traders[tic_code]
            plr_dict[tic_code] = trader.calculate_profit_loss_ratio()
        return plr_dict

    def get_open_close_reports(self) -> dict[str, pd.DataFrame]:
        oc_reports = {}
        for tic_code, eval_report in self.evaluate_report.items():
            # print(f'PLR for {tic_code}:')
            trader = self.traders[tic_code]
            oc_reports[tic_code] = trader.sum_up_open_close_records()
        return oc_reports