"""
Helios Portfolio

Simple in-memory portfolio model for backtests.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Portfolio:
    cash: float = 100000.0
    positions: Dict[str, float] = field(default_factory=dict)

    def update(self, symbol: str, qty_delta: float, price: float) -> None:
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty_delta
        self.cash -= qty_delta * price
