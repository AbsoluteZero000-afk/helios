"""
Helios Paper Broker

A minimal broker for paper trading/execution simulation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    symbol: str
    side: str
    qty: float
    price: Optional[float] = None


class PaperBroker:
    """Simulated execution for testing/backtesting."""

    async def place_order(self, order: Order) -> str:
        return f"ord_{order.symbol}_sim"
