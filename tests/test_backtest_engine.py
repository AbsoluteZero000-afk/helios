"""
Helios Backtest Engine Tests

Tests for backtest engine basic lifecycle and event processing.
"""

import asyncio
import pytest

from core.engine import TradingEngine, EngineMode
from strategies.base import StrategyConfig
from strategies.trend_following import TrendFollowingStrategy
from data.feeder import HistoricalFeeder


@pytest.mark.asyncio
async def test_backtest_engine_lifecycle():
    engine = TradingEngine(mode=EngineMode.BACKTEST)
    engine.add_symbol("AAPL")
    cfg = StrategyConfig(name="tf", symbols=["AAPL"], parameters={"fast_period":5, "slow_period":20, "min_data_length":25})
    engine.strategies['tf'] = TrendFollowingStrategy(cfg)
    await engine.start()
    feeder = HistoricalFeeder(symbol="AAPL", steps=50)
    await feeder.run()
    await asyncio.sleep(0.2)
    await engine.stop()
    assert engine.state.value == "stopped"
