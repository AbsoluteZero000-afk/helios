"""
Helios Test Suite

Pytest tests for strategies and risk manager.
"""

import asyncio
import pytest

from strategies.base import StrategyConfig
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from data.feeder import HistoricalFeeder
from core.engine import TradingEngine, EngineMode
from risk.manager import RiskManager


@pytest.mark.asyncio
async def test_trend_following_generates_signals():
    engine = TradingEngine(mode=EngineMode.BACKTEST)
    symbol = "AAPL"
    engine.add_symbol(symbol)
    cfg = StrategyConfig(name="tf", symbols=[symbol], parameters={"fast_period":5, "slow_period":20, "min_data_length":25})
    strat = TrendFollowingStrategy(cfg)
    engine.strategies['tf'] = strat
    await engine.start()
    feeder = HistoricalFeeder(symbol=symbol, steps=60)
    await feeder.run()
    await asyncio.sleep(0.2)
    await engine.stop()
    assert strat.metrics.total_signals >= 1


def test_risk_manager_limits():
    rm = RiskManager()
    pv = 100000.0
    max_val = rm.max_position_value(pv)
    assert max_val == pytest.approx(pv * 0.02)
