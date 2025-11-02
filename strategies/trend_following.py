"""
Helios Trend Following Strategy

Implements a simple SMA crossover trend following strategy.
"""

from typing import Optional
import pandas as pd

from strategies.base import (
    TechnicalAnalysisStrategy, StrategyConfig, SignalEvent, SignalType
)
from utils.logger import strategy_logger as logger


class TrendFollowingStrategy(TechnicalAnalysisStrategy):
    """SMA crossover trend following strategy."""

    async def initialize(self) -> None:
        params = self.config.parameters
        self.fast_period = int(params.get('fast_period', 20))
        self.slow_period = int(params.get('slow_period', 50))
        self.signal_threshold = float(params.get('signal_threshold', 0.6))
        logger.info(
            f"TrendFollowing initialized fast={self.fast_period} slow={self.slow_period}"
        )

    async def cleanup(self) -> None:
        pass

    async def update_indicators(self, symbol: str) -> None:
        df = self.market_data.get(symbol)
        if df is None or df.empty:
            return
        sma_fast = self.calculate_sma(symbol, self.fast_period)
        sma_slow = self.calculate_sma(symbol, self.slow_period)
        if sma_fast is None or sma_slow is None:
            return
        self.indicators.setdefault(symbol, {})['sma_fast'] = sma_fast
        self.indicators[symbol]['sma_slow'] = sma_slow

    async def generate_signal(self, symbol: str) -> Optional[SignalEvent]:
        ind = self.indicators.get(symbol, {})
        sma_fast: Optional[pd.Series] = ind.get('sma_fast')
        sma_slow: Optional[pd.Series] = ind.get('sma_slow')
        if sma_fast is None or sma_slow is None or len(sma_slow.dropna()) < 2:
            return None
        cross_up = sma_fast.iloc[-2] <= sma_slow.iloc[-2] and sma_fast.iloc[-1] > sma_slow.iloc[-1]
        cross_down = sma_fast.iloc[-2] >= sma_slow.iloc[-2] and sma_fast.iloc[-1] < sma_slow.iloc[-1]
        price = self.get_current_price(symbol) or 0.0
        if cross_up:
            strength = self.signal_threshold
            return SignalEvent(
                symbol=symbol,
                signal_type=SignalType.BUY.value,
                strength=strength,
                strategy_name=self.config.name,
                target_price=price
            )
        if cross_down:
            strength = self.signal_threshold
            return SignalEvent(
                symbol=symbol,
                signal_type=SignalType.SELL.value,
                strength=strength,
                strategy_name=self.config.name,
                target_price=price
            )
        return None
