"""
Helios Mean Reversion Strategy

Implements a Bollinger Band based mean reversion strategy.
"""

from typing import Optional, Tuple
import pandas as pd

from strategies.base import (
    TechnicalAnalysisStrategy, StrategyConfig, SignalEvent, SignalType
)
from utils.logger import strategy_logger as logger


class MeanReversionStrategy(TechnicalAnalysisStrategy):
    """Bollinger band mean-reversion strategy."""

    async def initialize(self) -> None:
        params = self.config.parameters
        self.bb_period = int(params.get('bb_period', 20))
        self.bb_std = float(params.get('bb_std', 2.0))
        self.signal_threshold = float(params.get('signal_threshold', 0.6))
        logger.info(
            f"MeanReversion initialized period={self.bb_period} std={self.bb_std}"
        )

    async def cleanup(self) -> None:
        pass

    async def update_indicators(self, symbol: str) -> None:
        df = self.market_data.get(symbol)
        if df is None or df.empty:
            return
        bands: Optional[Tuple[pd.Series, pd.Series, pd.Series]] = self.calculate_bollinger_bands(
            symbol, self.bb_period, self.bb_std
        )
        if not bands:
            return
        upper, middle, lower = bands
        self.indicators.setdefault(symbol, {})['bb_upper'] = upper
        self.indicators[symbol]['bb_middle'] = middle
        self.indicators[symbol]['bb_lower'] = lower

    async def generate_signal(self, symbol: str) -> Optional[SignalEvent]:
        ind = self.indicators.get(symbol, {})
        upper: Optional[pd.Series] = ind.get('bb_upper')
        lower: Optional[pd.Series] = ind.get('bb_lower')
        if upper is None or lower is None or len(upper.dropna()) == 0:
            return None
        price = self.get_current_price(symbol) or 0.0
        # Buy when price crosses below lower band, Sell when above upper band
        if price < lower.iloc[-1]:
            return SignalEvent(
                symbol=symbol,
                signal_type=SignalType.BUY.value,
                strength=self.signal_threshold,
                strategy_name=self.config.name,
                target_price=price
            )
        if price > upper.iloc[-1]:
            return SignalEvent(
                symbol=symbol,
                signal_type=SignalType.SELL.value,
                strength=self.signal_threshold,
                strategy_name=self.config.name,
                target_price=price
            )
        return None
