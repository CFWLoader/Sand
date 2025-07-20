import pandas as pd

from src.quant.signals.bollinger_band_hunter import BollingerBandHunter
from src.quant.trader.quant_trader import QuantTrader


class StrategyEvaluator(object):

    TRADE_SIGNALS = {
        'boll': BollingerBandHunter
    }

    BUY_IN_STRATEGIES = QuantTrader.BUY_IN_STRATEGIES

    SELL_OUT_STRATEGIES = QuantTrader.SELL_OUT_STRATEGIES

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
        self.signal_source = signal_clz(stock_data)
        self.trader = QuantTrader(
            initial_asset, buyin_strategy, sellout_strategy,
            buyin_price, sellout_price, lot_size
        )

    def evaluate(self) -> pd.DataFrame:
        trade_signals = self.signal_source.get_signals()
        return self.trader.simulate(self.signal_source.indicated_df)