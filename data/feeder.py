"""
Helios Data Feeder

Simple historical feeder to support backtests with mock data.
"""

from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta

from core.events import MarketDataEvent, event_bus


@dataclass
class Bar:
    timestamp: datetime
    price: float
    volume: int


class HistoricalFeeder:
    """Feeds synthetic bars into the event bus (deterministic for tests)."""

    def __init__(self, symbol: str, start_price: float = 100.0, steps: int = 300):
        self.symbol = symbol
        self.start_price = start_price
        self.steps = steps

    async def run(self) -> None:
        now = datetime.now()
        price = self.start_price
        for i in range(self.steps):
            # deterministic walk
            price += (1 if (i % 10) < 6 else -1) * 0.5
            evt = MarketDataEvent(symbol=self.symbol, price=price, volume=100)
            await event_bus.publish(evt)
