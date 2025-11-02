"""
Helios Trading Engine

Main trading engine that coordinates data flow, signal generation,
risk management, and order execution using event-driven architecture.
"""

import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from core.events import (
    Event, EventHandler, EventType, event_bus,
    MarketDataEvent, SignalEvent, OrderEvent, FillEvent,
    SystemEvent, RiskEvent
)
from config.settings import get_settings
from utils.logger import core_logger as logger
from utils.slack import slack_notifier
from utils.sentinel import system_sentinel


class EngineState(str, Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class EngineMode(str, Enum):
    """Trading engine operation modes."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


@dataclass
class EngineStats:
    """Trading engine statistics."""
    start_time: datetime
    uptime: timedelta
    events_processed: int
    signals_generated: int
    orders_placed: int
    fills_received: int
    errors_encountered: int
    active_positions: int
    total_pnl: float


class TradingEngine(EventHandler):
    """
    Main trading engine coordinating all system components.
    
    Manages the flow of market data, signal generation, risk checks,
    and order execution through an event-driven architecture.
    """
    
    def __init__(self, mode: EngineMode = EngineMode.PAPER):
        """Initialize the trading engine."""
        self.settings = get_settings()
        self.mode = mode
        self.state = EngineState.STOPPED
        self.start_time: Optional[datetime] = None
        
        # Component registries
        self.data_feeders: Dict[str, object] = {}
        self.strategies: Dict[str, object] = {}
        self.risk_managers: List[object] = []
        self.brokers: Dict[str, object] = {}
        
        # Runtime tracking
        self.active_symbols: Set[str] = set()
        self.pending_orders: Dict[str, OrderEvent] = {}
        self.active_positions: Dict[str, float] = {}
        
        # Statistics
        self.stats = EngineStats(
            start_time=datetime.now(),
            uptime=timedelta(),
            events_processed=0,
            signals_generated=0,
            orders_placed=0,
            fills_received=0,
            errors_encountered=0,
            active_positions=0,
            total_pnl=0.0
        )
        
        logger.info(f"Trading engine initialized in {mode.value} mode")
    
    async def start(self) -> None:
        """
        Start the trading engine and all components.
        """
        if self.state != EngineState.STOPPED:
            logger.warning(f"Cannot start engine in state: {self.state}")
            return
        
        self.state = EngineState.STARTING
        self.start_time = datetime.now()
        
        try:
            # Start event bus
            await event_bus.start()
            
            # Subscribe to relevant events
            self._subscribe_to_events()
            
            # Start system sentinel if enabled
            if self.settings.sentinel_enabled:
                await system_sentinel.start_monitoring()
            
            # Initialize components
            await self._initialize_components()
            
            # Start data feeds
            await self._start_data_feeds()
            
            # Start strategies
            await self._start_strategies()
            
            # Engine is now running
            self.state = EngineState.RUNNING
            
            logger.info(f"Trading engine started successfully in {self.mode.value} mode")
            
            # Send startup notification
            await slack_notifier.system_alert(
                f"Helios trading engine started in {self.mode.value} mode",
                component="engine",
                severity="success",
                mode=self.mode.value,
                active_symbols=len(self.active_symbols)
            )
            
            # Publish system start event
            await event_bus.publish(SystemEvent(
                system_type=EventType.SYSTEM_STARTED,
                message=f"Trading engine started in {self.mode.value} mode",
                component="trading_engine"
            ))
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.stats.errors_encountered += 1
            logger.error(f"Failed to start trading engine: {e}")
            
            await slack_notifier.system_alert(
                f"Trading engine startup failed: {e}",
                component="engine",
                severity="error"
            )
            
            raise
    
    async def stop(self) -> None:
        """
        Stop the trading engine and all components.
        """
        if self.state == EngineState.STOPPED:
            logger.warning("Engine already stopped")
            return
        
        self.state = EngineState.STOPPING
        
        try:
            logger.info("Stopping trading engine...")
            
            # Stop strategies
            await self._stop_strategies()
            
            # Stop data feeds
            await self._stop_data_feeds()
            
            # Cancel pending orders
            await self._cancel_pending_orders()
            
            # Stop system sentinel
            if self.settings.sentinel_enabled:
                await system_sentinel.stop_monitoring()
            
            # Stop event bus
            await event_bus.stop()
            
            self.state = EngineState.STOPPED
            
            # Calculate final stats
            if self.start_time:
                self.stats.uptime = datetime.now() - self.start_time
            
            logger.info("Trading engine stopped successfully")
            
            # Send shutdown notification
            await slack_notifier.system_alert(
                "Helios trading engine stopped",
                component="engine",
                severity="info",
                uptime=str(self.stats.uptime),
                events_processed=self.stats.events_processed
            )
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.stats.errors_encountered += 1
            logger.error(f"Error stopping trading engine: {e}")
            raise
    
    def _subscribe_to_events(self) -> None:
        """
        Subscribe to relevant events from the event bus.
        """
        # Subscribe to all events that the engine should handle
        event_types = [
            EventType.MARKET_DATA_RECEIVED,
            EventType.SIGNAL_GENERATED,
            EventType.ORDER_FILLED,
            EventType.RISK_THRESHOLD_EXCEEDED,
            EventType.SYSTEM_ERROR
        ]
        
        for event_type in event_types:
            event_bus.subscribe(event_type, self)
        
        logger.debug(f"Subscribed to {len(event_types)} event types")
    
    async def _initialize_components(self) -> None:
        """
        Initialize all trading components.
        """
        # This would initialize data feeders, strategies, risk managers, etc.
        # For now, we'll create placeholder implementations
        
        logger.info("Initializing trading components...")
        
        # Initialize risk managers
        # self.risk_managers = [RiskManager()]
        
        # Initialize brokers based on mode
        if self.mode in [EngineMode.LIVE, EngineMode.PAPER]:
            # self.brokers['alpaca'] = AlpacaBroker()
            pass
        
        logger.info("Trading components initialized")
    
    async def _start_data_feeds(self) -> None:
        """
        Start market data feeds for active symbols.
        """
        logger.info("Starting data feeds...")
        
        # Start data feeds for each active symbol
        for symbol in self.active_symbols:
            # data_feeder = DataFeeder(symbol)
            # await data_feeder.start()
            # self.data_feeders[symbol] = data_feeder
            pass
        
        logger.info(f"Started data feeds for {len(self.active_symbols)} symbols")
    
    async def _stop_data_feeds(self) -> None:
        """
        Stop all market data feeds.
        """
        logger.info("Stopping data feeds...")
        
        for symbol, feeder in self.data_feeders.items():
            try:
                # await feeder.stop()
                pass
            except Exception as e:
                logger.error(f"Error stopping data feed for {symbol}: {e}")
        
        self.data_feeders.clear()
        logger.info("Data feeds stopped")
    
    async def _start_strategies(self) -> None:
        """
        Start all registered trading strategies.
        """
        logger.info("Starting trading strategies...")
        
        for name, strategy in self.strategies.items():
            try:
                # await strategy.start()
                logger.info(f"Started strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to start strategy {name}: {e}")
        
        logger.info(f"Started {len(self.strategies)} strategies")
    
    async def _stop_strategies(self) -> None:
        """
        Stop all running trading strategies.
        """
        logger.info("Stopping trading strategies...")
        
        for name, strategy in self.strategies.items():
            try:
                # await strategy.stop()
                logger.info(f"Stopped strategy: {name}")
            except Exception as e:
                logger.error(f"Error stopping strategy {name}: {e}")
        
        logger.info("Trading strategies stopped")
    
    async def _cancel_pending_orders(self) -> None:
        """
        Cancel all pending orders before shutdown.
        """
        if not self.pending_orders:
            return
        
        logger.info(f"Cancelling {len(self.pending_orders)} pending orders...")
        
        for order_id, order in self.pending_orders.items():
            try:
                # Cancel order through broker
                # await self.brokers[order.broker].cancel_order(order_id)
                logger.info(f"Cancelled order: {order_id}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
        
        self.pending_orders.clear()
    
    # EventHandler implementation
    async def handle(self, event: Event) -> None:
        """
        Handle events from the event bus.
        
        Args:
            event: Event to handle
        """
        self.stats.events_processed += 1
        
        try:
            if isinstance(event, MarketDataEvent):
                await self._handle_market_data(event)
            elif isinstance(event, SignalEvent):
                await self._handle_signal(event)
            elif isinstance(event, FillEvent):
                await self._handle_fill(event)
            elif isinstance(event, RiskEvent):
                await self._handle_risk_event(event)
            elif isinstance(event, SystemEvent):
                await self._handle_system_event(event)
            
        except Exception as e:
            self.stats.errors_encountered += 1
            logger.error(f"Error handling event {event.event_type}: {e}")
    
    def can_handle(self, event_type: EventType) -> bool:
        """
        Check if this handler can handle the event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            bool: True if can handle
        """
        return event_type in [
            EventType.MARKET_DATA_RECEIVED,
            EventType.SIGNAL_GENERATED,
            EventType.ORDER_FILLED,
            EventType.RISK_THRESHOLD_EXCEEDED,
            EventType.SYSTEM_ERROR
        ]
    
    async def _handle_market_data(self, event: MarketDataEvent) -> None:
        """
        Handle market data events by forwarding to strategies.
        """
        # Forward to relevant strategies
        for strategy in self.strategies.values():
            if hasattr(strategy, 'on_market_data'):
                try:
                    # await strategy.on_market_data(event)
                    pass
                except Exception as e:
                    logger.error(f"Strategy market data error: {e}")
    
    async def _handle_signal(self, event: SignalEvent) -> None:
        """
        Handle trading signals by performing risk checks and placing orders.
        """
        self.stats.signals_generated += 1
        
        logger.info(f"Signal received: {event.signal_type} {event.symbol} strength={event.strength}")
        
        # Perform risk checks
        risk_approved = await self._check_risk(event)
        
        if not risk_approved:
            logger.warning(f"Signal rejected by risk management: {event.symbol}")
            return
        
        # Convert signal to order
        order = await self._signal_to_order(event)
        
        if order:
            # Place order
            await self._place_order(order)
    
    async def _handle_fill(self, event: FillEvent) -> None:
        """
        Handle order fill events by updating positions and portfolio.
        """
        self.stats.fills_received += 1
        
        logger.info(f"Fill received: {event.side} {event.quantity} {event.symbol} @ {event.fill_price}")
        
        # Update position
        if event.symbol not in self.active_positions:
            self.active_positions[event.symbol] = 0.0
        
        if event.side == 'BUY':
            self.active_positions[event.symbol] += event.quantity
        else:
            self.active_positions[event.symbol] -= event.quantity
        
        # Remove from pending orders
        if event.order_id in self.pending_orders:
            del self.pending_orders[event.order_id]
        
        # Send trade alert
        await slack_notifier.trade_alert(
            action=event.side,
            symbol=event.symbol,
            quantity=event.quantity,
            price=event.fill_price,
            commission=event.commission
        )
    
    async def _handle_risk_event(self, event: RiskEvent) -> None:
        """
        Handle risk management events.
        """
        logger.warning(f"Risk event: {event.risk_type} - {event.message}")
        
        # Send risk alert
        await slack_notifier.risk_alert(
            message=event.message,
            risk_level=event.severity.lower(),
            risk_type=event.risk_type,
            current_value=event.current_value,
            threshold=event.threshold_value
        )
        
        # Take action based on severity
        if event.severity == 'CRITICAL':
            await self._handle_critical_risk(event)
    
    async def _handle_system_event(self, event: SystemEvent) -> None:
        """
        Handle system events.
        """
        if event.system_type == EventType.SYSTEM_ERROR:
            self.stats.errors_encountered += 1
            logger.error(f"System error: {event.message}")
        else:
            logger.info(f"System event: {event.message}")
    
    async def _check_risk(self, signal: SignalEvent) -> bool:
        """
        Perform risk checks on a trading signal.
        
        Args:
            signal: Trading signal to check
            
        Returns:
            bool: True if signal passes risk checks
        """
        # Implement risk checking logic
        # For now, return True as placeholder
        return True
    
    async def _signal_to_order(self, signal: SignalEvent) -> Optional[OrderEvent]:
        """
        Convert a trading signal to an order.
        
        Args:
            signal: Trading signal to convert
            
        Returns:
            Optional[OrderEvent]: Generated order or None
        """
        if signal.signal_type == 'HOLD':
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        
        return OrderEvent(
            symbol=signal.symbol,
            order_type='MARKET',
            side=signal.signal_type,
            quantity=position_size,
            price=signal.target_price,
            source="trading_engine"
        )
    
    def _calculate_position_size(self, signal: SignalEvent) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            float: Position size
        """
        # Implement position sizing logic
        # For now, use default position size
        return 100.0  # Placeholder
    
    async def _place_order(self, order: OrderEvent) -> None:
        """
        Place an order through the appropriate broker.
        
        Args:
            order: Order to place
        """
        self.stats.orders_placed += 1
        
        # Generate order ID
        order.order_id = f"ord_{datetime.now().timestamp()}"
        
        # Add to pending orders
        self.pending_orders[order.order_id] = order
        
        logger.info(f"Placing order: {order.side} {order.quantity} {order.symbol}")
        
        # Publish order event
        await event_bus.publish(order)
        
        # For simulation/backtest, simulate immediate fill
        if self.mode in [EngineMode.SIMULATION, EngineMode.BACKTEST]:
            await self._simulate_fill(order)
    
    async def _simulate_fill(self, order: OrderEvent) -> None:
        """
        Simulate order fill for backtesting/simulation.
        
        Args:
            order: Order to simulate fill for
        """
        # Simulate a fill after short delay
        await asyncio.sleep(0.1)
        
        fill_event = FillEvent(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=order.price or 100.0,  # Placeholder price
            commission=1.0,  # Placeholder commission
            order_id=order.order_id,
            execution_id=f"exec_{datetime.now().timestamp()}",
            source="simulation"
        )
        
        await event_bus.publish(fill_event)
    
    async def _handle_critical_risk(self, event: RiskEvent) -> None:
        """
        Handle critical risk events that may require immediate action.
        
        Args:
            event: Critical risk event
        """
        logger.critical(f"Critical risk event: {event.message}")
        
        # Implement emergency procedures
        # e.g., close all positions, stop trading, etc.
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add a symbol to track.
        
        Args:
            symbol: Symbol to add
        """
        self.active_symbols.add(symbol)
        logger.info(f"Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from tracking.
        
        Args:
            symbol: Symbol to remove
        """
        self.active_symbols.discard(symbol)
        logger.info(f"Removed symbol: {symbol}")
    
    def get_stats(self) -> EngineStats:
        """
        Get current engine statistics.
        
        Returns:
            EngineStats: Current statistics
        """
        if self.start_time:
            self.stats.uptime = datetime.now() - self.start_time
        
        self.stats.active_positions = len([p for p in self.active_positions.values() if p != 0])
        
        return self.stats
    
    def get_status(self) -> Dict[str, any]:
        """
        Get comprehensive engine status.
        
        Returns:
            Dict[str, any]: Engine status information
        """
        stats = self.get_stats()
        
        return {
            "state": self.state.value,
            "mode": self.mode.value,
            "uptime": str(stats.uptime),
            "active_symbols": list(self.active_symbols),
            "active_positions": dict(self.active_positions),
            "pending_orders": len(self.pending_orders),
            "stats": {
                "events_processed": stats.events_processed,
                "signals_generated": stats.signals_generated,
                "orders_placed": stats.orders_placed,
                "fills_received": stats.fills_received,
                "errors_encountered": stats.errors_encountered
            }
        }


# Global engine instance
trading_engine = TradingEngine()
