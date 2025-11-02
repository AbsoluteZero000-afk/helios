"""
Helios Backtest Runner

Command-line entry point to run a simple backtest.
"""

import argparse
import asyncio

from core.engine import TradingEngine, EngineMode
from utils.logger import setup_logging
from strategies.base import StrategyConfig
from strategies.trend_following import TrendFollowingStrategy
from data.feeder import HistoricalFeeder


async def main_async(symbol: str, strategy_name: str):
    setup_logging()
    engine = TradingEngine(mode=EngineMode.BACKTEST)
    engine.add_symbol(symbol)

    # register strategy
    cfg = StrategyConfig(name=strategy_name, symbols=[symbol], parameters={"fast_period": 5, "slow_period": 20, "min_data_length": 25})
    strat = TrendFollowingStrategy(cfg)
    engine.strategies[strategy_name] = strat

    await engine.start()

    feeder = HistoricalFeeder(symbol=symbol, start_price=100.0, steps=300)
    await feeder.run()

    await asyncio.sleep(0.5)
    await engine.stop()


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--strategy', default='trend_following')
    args = parser.parse_args()
    asyncio.run(main_async(args.symbol, args.strategy))


if __name__ == "__main__":
    cli()
