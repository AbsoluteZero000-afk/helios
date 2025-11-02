"""
Helios Event System

Event-driven architecture for trading system communication.
Provides a decoupled messaging system between components.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Minimal logger import to avoid circular dependencies during startup
try:
    from utils.logger import core_logger as logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Core event types for the trading system."""
    MARKET_DATA_RECEIVED = "market_data_received"
    PRICE_UPDATE = "price_update"
    VOLUME_UPDATE = "volume_update"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_CANCELLED = "signal_cancelled"
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    PORTFOLIO_UPDATED = "portfolio_updated"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    DRAWDOWN_LIMIT_HIT = "drawdown_limit_hit"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_ERROR = "system_error"
    HEALTH_CHECK = "health_check"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"


@dataclass
class Event(ABC):
    """Base event class for all system events."""
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now().timestamp()}")
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the event type."""
        pass


# CRITICAL: All required fields (no defaults) MUST come before optional fields (with defaults)
# This is a Python dataclass requirement to avoid "non-default argument follows default argument"

@dataclass 
class MarketDataEvent(Event):
    """Market data event - REQUIRED fields first, then optional fields"""
    symbol: str  # REQUIRED - no default
    price: float  # REQUIRED - no default
    volume: int   # REQUIRED - no default
    # Optional fields with defaults come after
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.MARKET_DATA_RECEIVED


@dataclass
class SignalEvent(Event):
    """Trading signal event - REQUIRED fields first, then optional fields"""
    symbol: str           # REQUIRED
    signal_type: str      # REQUIRED
    strength: float       # REQUIRED  
    strategy_name: str    # REQUIRED
    # Optional fields with defaults
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL_GENERATED


@dataclass
class OrderEvent(Event):
    """Order execution event - REQUIRED fields first, then optional fields"""
    symbol: str        # REQUIRED
    order_type: str    # REQUIRED  
    side: str          # REQUIRED
    quantity: float    # REQUIRED
    # Optional fields with defaults
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "PENDING"
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_CREATED


@dataclass
class FillEvent(Event):
    """Order fill event - ALL fields are required (no defaults)"""
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    order_id: str
    execution_id: str
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_FILLED


@dataclass
class PositionEvent(Event):
    """Position update event - REQUIRED fields first, then optional fields"""
    symbol: str            # REQUIRED
    quantity: float        # REQUIRED
    avg_price: float       # REQUIRED
    market_value: float    # REQUIRED
    unrealized_pnl: float  # REQUIRED
    # Optional field with default
    realized_pnl: float = 0.0
    
    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_UPDATED


@dataclass
class RiskEvent(Event):
    """Risk management event - REQUIRED fields first, then optional fields"""
    risk_type: str        # REQUIRED
    severity: str         # REQUIRED
    message: str          # REQUIRED
    current_value: float  # REQUIRED
    threshold_value: float # REQUIRED
    # Optional field with default
    symbol: Optional[str] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.RISK_THRESHOLD_EXCEEDED


@dataclass
class SystemEvent(Event):
    """System status event - REQUIRED fields first, then optional fields"""
    system_type: EventType  # REQUIRED
    message: str            # REQUIRED
    # Optional fields with defaults
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
    """Central event bus for managing event flow."""
    
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
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        try:
            await self._event_queue.put(event)
            logger.debug(f"Published event: {event.event_type.value}")
        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event")
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event by dispatching to registered handlers."""
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
        """Safely handle an event with error catching."""
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
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
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
