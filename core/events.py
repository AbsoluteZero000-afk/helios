"""
Helios Event System

Event-driven architecture for trading system communication.
Provides a decoupled messaging system between components.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from utils.logger import core_logger as logger


class EventType(str, Enum):
    """Core event types for the trading system."""
    # Market Data Events
    MARKET_DATA_RECEIVED = "market_data_received"
    PRICE_UPDATE = "price_update"
    VOLUME_UPDATE = "volume_update"
    
    # Signal Events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_CANCELLED = "signal_cancelled"
    
    # Order Events
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    
    # Portfolio Events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    PORTFOLIO_UPDATED = "portfolio_updated"
    
    # Risk Events
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    DRAWDOWN_LIMIT_HIT = "drawdown_limit_hit"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    
    # System Events
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK = "health_check"
    
    # Strategy Events
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"


@dataclass
class Event(ABC):
    """
    Base event class for all system events.
    
    All events must inherit from this class and implement the event_type property.
    FIXED: All required fields (no defaults) come first
    """
    # Required fields first (no defaults)
    # (no required fields in base class)
    
    # Optional/default fields after
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now().timestamp()}")
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the event type."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "event_type": self.event_type.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            **{k: v for k, v in self.__dict__.items() 
               if k not in ['timestamp', 'event_id', 'source', 'metadata']}
        }


@dataclass
class MarketDataEvent(Event):
    """Market data event containing price and volume information.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    symbol: str
    price: float
    volume: int
    
    # Optional fields after
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.MARKET_DATA_RECEIVED


@dataclass
class SignalEvent(Event):
    """Trading signal event from strategy analysis.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # Signal strength 0.0 to 1.0
    strategy_name: str
    
    # Optional fields after
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL_GENERATED


@dataclass
class OrderEvent(Event):
    """Order execution event.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    side: str  # 'BUY', 'SELL'
    quantity: float
    
    # Optional fields after
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "PENDING"
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_CREATED


@dataclass
class FillEvent(Event):
    """Order fill event.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    symbol: str
    side: str  # 'BUY', 'SELL'
    quantity: float
    fill_price: float
    commission: float
    order_id: str
    execution_id: str
    
    # No optional fields in this class
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_FILLED


@dataclass
class PositionEvent(Event):
    """Position update event.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    
    # Optional/default fields after
    realized_pnl: float = 0.0
    
    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_UPDATED


@dataclass
class RiskEvent(Event):
    """Risk management event.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    risk_type: str  # 'POSITION_SIZE', 'DRAWDOWN', 'CONCENTRATION'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    current_value: float
    threshold_value: float
    
    # Optional fields after
    symbol: Optional[str] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.RISK_THRESHOLD_EXCEEDED


@dataclass
class SystemEvent(Event):
    """System status event.
    FIXED: Required fields first, optional/default fields after"""
    # Required fields first (no defaults)
    system_type: EventType
    message: str
    
    # Optional fields after
    component: Optional[str] = None
    error_details: Optional[str] = None
    
    @property
    def event_type(self) -> EventType:
        return self.system_type


class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event."""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if this handler can handle the event type."""
        pass

class EventBus:
    """
    Central event bus for managing event flow in the trading system.
    
    Provides publish/subscribe pattern for decoupled component communication.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_task: Optional[asyncio.Task] = None
        
        logger.info("Event bus initialized")
    
    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            logger.warning("Event bus already running")
            return
        
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        
        # Publish system start event
        await self.publish(SystemEvent(
            system_type=EventType.SYSTEM_STARTED,
            message="Event bus started",
            component="event_bus"
        ))
    
    async def stop(self) -> None:
        """Stop the event bus processing."""
        if not self._running:
            return
        
        self._running = False
        
        # Publish system stop event
        await self.publish(SystemEvent(
            system_type=EventType.SYSTEM_STOPPED,
            message="Event bus stopping",
            component="event_bus"
        ))
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Handler to process events
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unsubscribe a handler from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
        """
        try:
            await self._event_queue.put(event)
            logger.debug(f"Published event: {event.event_type.value}")
        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event")
    
    async def _process_events(self) -> None:
        """
        Process events from the queue.
        """
        while self._running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                # No event received, continue loop
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """
        Handle a single event by dispatching to registered handlers.
        
        Args:
            event: Event to handle
        """
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"No handlers for event: {event.event_type.value}")
            return
        
        # Process handlers concurrently
        tasks = []
        for handler in handlers:
            if handler.can_handle(event.event_type):
                task = asyncio.create_task(self._safe_handle(handler, event))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        """
        Safely handle an event with error catching.
        
        Args:
            handler: Event handler
            event: Event to handle
        """
        try:
            await handler.handle(event)
        except Exception as e:
            logger.error(f"Handler error for {event.event_type.value}: {e}")
            
            # Publish error event
            error_event = SystemEvent(
                system_type=EventType.SYSTEM_ERROR,
                message=f"Handler error: {e}",
                component="event_bus",
                error_details=str(e)
            )
            # Avoid infinite loop by not awaiting publish
            asyncio.create_task(self._event_queue.put(error_event))
    
    def get_event_history(self, event_type: Optional[EventType] = None) -> List[Event]:
        """
        Get event history, optionally filtered by type.
        
        Args:
            event_type: Optional event type filter
            
        Returns:
            List[Event]: Event history
        """
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Dict[str, Any]: Event bus statistics
        """
        event_counts = {}
        for event in self._event_history:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "running": self._running,
            "total_events": len(self._event_history),
            "queue_size": self._event_queue.qsize(),
            "handler_count": sum(len(handlers) for handlers in self._handlers.values()),
            "event_type_counts": event_counts
        }


# Global event bus instance
event_bus = EventBus()
