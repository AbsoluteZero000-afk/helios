"""
Helios Strategy Base Classes

Abstract base classes and framework for implementing trading strategies.
Provides standardized interface for signal generation and risk management.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from core.events import (
    Event, EventHandler, EventType, MarketDataEvent, SignalEvent, event_bus
)
from config.settings import get_settings
from utils.logger import strategy_logger as logger


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class StrategyState(str, Enum):
    """Strategy execution states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StrategyConfig:
    """Strategy configuration parameters."""
    name: str
    enabled: bool = True
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    position_sizing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics."""
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_holding_period: timedelta = field(default_factory=lambda: timedelta())
    last_signal_time: Optional[datetime] = None


@dataclass
class MarketData:
    """Market data container for strategy analysis."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close
        }


class BaseStrategy(EventHandler, ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides standardized interface for strategy implementation with
    event handling, data management, and signal generation.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize the strategy."""
        self.config = config
        self.settings = get_settings()
        self.state = StrategyState.STOPPED
        self.start_time: Optional[datetime] = None
        
        # Data storage
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, pd.Series]] = {}
        self.signals_history: List[SignalEvent] = []
        
        # Performance tracking
        self.metrics = StrategyMetrics()
        self.positions: Dict[str, float] = {}
        self.unrealized_pnl: float = 0.0
        
        # Configuration
        self.max_data_length = self.config.parameters.get('max_data_length', 1000)
        self.min_data_length = self.config.parameters.get('min_data_length', 50)
        
        logger.info(f"Strategy '{self.config.name}' initialized")
    
    async def start(self) -> None:
        """
        Start the strategy execution.
        """
        if self.state != StrategyState.STOPPED:
            logger.warning(f"Cannot start strategy in state: {self.state}")
            return
        
        self.state = StrategyState.STARTING
        self.start_time = datetime.now()
        
        try:
            # Subscribe to market data events
            event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, self)
            
            # Initialize strategy-specific components
            await self.initialize()
            
            # Validate configuration
            self.validate_config()
            
            self.state = StrategyState.RUNNING
            logger.info(f"Strategy '{self.config.name}' started successfully")
            
        except Exception as e:
            self.state = StrategyState.ERROR
            logger.error(f"Failed to start strategy '{self.config.name}': {e}")
            raise
    
    async def stop(self) -> None:
        """
        Stop the strategy execution.
        """
        if self.state == StrategyState.STOPPED:
            return
        
        self.state = StrategyState.STOPPING
        
        try:
            # Unsubscribe from events
            event_bus.unsubscribe(EventType.MARKET_DATA_RECEIVED, self)
            
            # Cleanup strategy-specific components
            await self.cleanup()
            
            self.state = StrategyState.STOPPED
            logger.info(f"Strategy '{self.config.name}' stopped")
            
        except Exception as e:
            self.state = StrategyState.ERROR
            logger.error(f"Error stopping strategy '{self.config.name}': {e}")
            raise
    
    # EventHandler implementation
    async def handle(self, event: Event) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event: Event to handle
        """
        if not self.is_running():
            return
        
        try:
            if isinstance(event, MarketDataEvent):
                await self.on_market_data(event)
        except Exception as e:
            logger.error(f"Strategy '{self.config.name}' event handling error: {e}")
    
    def can_handle(self, event_type: EventType) -> bool:
        """
        Check if this handler can handle the event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            bool: True if can handle
        """
        return event_type == EventType.MARKET_DATA_RECEIVED
    
    async def on_market_data(self, event: MarketDataEvent) -> None:
        """
        Handle market data events.
        
        Args:
            event: Market data event
        """
        if event.symbol not in self.config.symbols:
            return
        
        # Convert event to MarketData
        market_data = MarketData(
            symbol=event.symbol,
            timestamp=event.timestamp,
            open=event.price,  # Simplified - would need OHLC data
            high=event.price,
            low=event.price,
            close=event.price,
            volume=event.volume,
            adj_close=event.price
        )
        
        # Store market data
        await self.store_market_data(market_data)
        
        # Update indicators
        await self.update_indicators(event.symbol)
        
        # Generate signals if enough data
        if self.has_sufficient_data(event.symbol):
            signal = await self.generate_signal(event.symbol)
            if signal:
                await self.emit_signal(signal)
    
    async def store_market_data(self, data: MarketData) -> None:
        """
        Store market data for analysis.
        
        Args:
            data: Market data to store
        """
        if data.symbol not in self.market_data:
            self.market_data[data.symbol] = pd.DataFrame()
        
        # Convert to DataFrame row
        row_data = data.to_dict()
        new_row = pd.DataFrame([row_data])
        
        # Append data
        self.market_data[data.symbol] = pd.concat(
            [self.market_data[data.symbol], new_row],
            ignore_index=True
        )
        
        # Trim data if too long
        if len(self.market_data[data.symbol]) > self.max_data_length:
            self.market_data[data.symbol] = self.market_data[data.symbol].tail(
                self.max_data_length
            ).reset_index(drop=True)
    
    def has_sufficient_data(self, symbol: str) -> bool:
        """
        Check if we have sufficient data for analysis.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if sufficient data available
        """
        return (
            symbol in self.market_data and
            len(self.market_data[symbol]) >= self.min_data_length
        )
    
    async def emit_signal(self, signal: SignalEvent) -> None:
        """
        Emit a trading signal.
        
        Args:
            signal: Signal to emit
        """
        self.signals_history.append(signal)
        self.metrics.total_signals += 1
        self.metrics.last_signal_time = datetime.now()
        
        logger.info(
            f"Strategy '{self.config.name}' generated {signal.signal_type} "
            f"signal for {signal.symbol} (strength: {signal.strength:.2f})"
        )
        
        # Publish signal to event bus
        await event_bus.publish(signal)
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float
    ) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0.0 to 1.0)
            current_price: Current market price
            
        Returns:
            float: Position size
        """
        # Default position sizing logic
        base_size = self.config.position_sizing.get('base_size', 100.0)
        max_size = self.config.position_sizing.get('max_size', 1000.0)
        
        # Scale by signal strength
        size = base_size * signal_strength
        
        # Apply maximum limit
        return min(size, max_size)
    
    def is_running(self) -> bool:
        """
        Check if strategy is currently running.
        
        Returns:
            bool: True if running
        """
        return self.state == StrategyState.RUNNING
    
    def get_metrics(self) -> StrategyMetrics:
        """
        Get strategy performance metrics.
        
        Returns:
            StrategyMetrics: Current metrics
        """
        # Calculate win rate
        if self.metrics.total_signals > 0:
            self.metrics.win_rate = (
                self.metrics.successful_signals / self.metrics.total_signals
            )
        
        return self.metrics
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status.
        
        Returns:
            Dict[str, Any]: Strategy status information
        """
        metrics = self.get_metrics()
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "enabled": self.config.enabled,
            "symbols": self.config.symbols,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
            "metrics": {
                "total_signals": metrics.total_signals,
                "win_rate": f"{metrics.win_rate:.2%}",
                "total_return": f"{metrics.total_return:.2%}",
                "max_drawdown": f"{metrics.max_drawdown:.2%}",
                "last_signal": metrics.last_signal_time.isoformat() if metrics.last_signal_time else None
            },
            "data_status": {
                symbol: len(df) for symbol, df in self.market_data.items()
            }
        }
    
    def validate_config(self) -> None:
        """
        Validate strategy configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.config.symbols:
            raise ValueError("Strategy must have at least one symbol")
        
        if not self.config.name:
            raise ValueError("Strategy must have a name")
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize strategy-specific components.
        Called during strategy startup.
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup strategy-specific components.
        Called during strategy shutdown.
        """
        pass
    
    @abstractmethod
    async def update_indicators(self, symbol: str) -> None:
        """
        Update technical indicators for a symbol.
        
        Args:
            symbol: Symbol to update indicators for
        """
        pass
    
    @abstractmethod
    async def generate_signal(
        self,
        symbol: str
    ) -> Optional[SignalEvent]:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Symbol to generate signal for
            
        Returns:
            Optional[SignalEvent]: Generated signal or None
        """
        pass


class TechnicalAnalysisStrategy(BaseStrategy):
    """
    Base class for technical analysis strategies.
    
    Provides common technical indicators and analysis methods.
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize technical analysis strategy."""
        super().__init__(config)
        
        # Technical analysis parameters
        self.lookback_period = self.config.parameters.get('lookback_period', 20)
        self.signal_threshold = self.config.parameters.get('signal_threshold', 0.7)
    
    def calculate_sma(self, symbol: str, period: int) -> Optional[pd.Series]:
        """
        Calculate Simple Moving Average.
        
        Args:
            symbol: Symbol to calculate for
            period: Moving average period
            
        Returns:
            Optional[pd.Series]: SMA values or None
        """
        if symbol not in self.market_data or len(self.market_data[symbol]) < period:
            return None
        
        return self.market_data[symbol]['close'].rolling(window=period).mean()
    
    def calculate_ema(self, symbol: str, period: int) -> Optional[pd.Series]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            symbol: Symbol to calculate for
            period: Moving average period
            
        Returns:
            Optional[pd.Series]: EMA values or None
        """
        if symbol not in self.market_data or len(self.market_data[symbol]) < period:
            return None
        
        return self.market_data[symbol]['close'].ewm(span=period).mean()
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[pd.Series]:
        """
        Calculate Relative Strength Index.
        
        Args:
            symbol: Symbol to calculate for
            period: RSI period
            
        Returns:
            Optional[pd.Series]: RSI values or None
        """
        if symbol not in self.market_data or len(self.market_data[symbol]) < period + 1:
            return None
        
        closes = self.market_data[symbol]['close']
        delta = closes.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            symbol: Symbol to calculate for
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Optional[Tuple[pd.Series, pd.Series, pd.Series]]: (upper, middle, lower) or None
        """
        if symbol not in self.market_data or len(self.market_data[symbol]) < period:
            return None
        
        closes = self.market_data[symbol]['close']
        middle = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def calculate_macd(
        self,
        symbol: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            symbol: Symbol to calculate for
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Optional[Tuple[pd.Series, pd.Series, pd.Series]]: (macd, signal, histogram) or None
        """
        if symbol not in self.market_data or len(self.market_data[symbol]) < slow_period:
            return None
        
        closes = self.market_data[symbol]['close']
        ema_fast = closes.ewm(span=fast_period).mean()
        ema_slow = closes.ewm(span=slow_period).mean()
        
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Optional[float]: Current price or None
        """
        if symbol not in self.market_data or self.market_data[symbol].empty:
            return None
        
        return self.market_data[symbol]['close'].iloc[-1]
    
    def get_latest_data(self, symbol: str, count: int = 1) -> Optional[pd.DataFrame]:
        """
        Get latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            count: Number of latest records to return
            
        Returns:
            Optional[pd.DataFrame]: Latest data or None
        """
        if symbol not in self.market_data or self.market_data[symbol].empty:
            return None
        
        return self.market_data[symbol].tail(count)
