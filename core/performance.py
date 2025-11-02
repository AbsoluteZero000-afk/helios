"""
Helios Performance Metrics

Utility functions to compute basic backtest metrics.
"""

from typing import Dict, List


def compute_basic_metrics(pnl_series: List[float]) -> Dict[str, float]:
    total_return = pnl_series[-1] - pnl_series[0] if pnl_series else 0.0
    return {"total_return": total_return}
