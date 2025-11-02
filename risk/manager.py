"""
Helios Risk Manager

Enforces portfolio-level and position-level risk constraints.
"""

from dataclasses import dataclass
from typing import Dict

from config.settings import get_settings


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float


class RiskManager:
    """Basic risk manager enforcing simple thresholds."""

    def __init__(self):
        self.settings = get_settings()

    def max_position_value(self, portfolio_value: float) -> float:
        return portfolio_value * self.settings.max_portfolio_risk

    def allow_trade(self, symbol: str, qty: float, price: float, portfolio_value: float, open_positions: Dict[str, Position]) -> bool:
        # Enforce max open positions
        if len(open_positions) >= self.settings.max_open_positions and symbol not in open_positions:
            return False
        # Enforce position value threshold
        if qty * price > self.max_position_value(portfolio_value):
            return False
        return True
