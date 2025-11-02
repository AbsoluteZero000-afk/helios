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
    NOTE: No required fields defined here. Optional/default fields only.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now().timestamp()}")
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the event type."""
        ...


# IMPORTANT: For dataclasses, required (no default) fields MUST come before any fields with defaults.

@dataclass
class MarketDataEvent(Event):
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.MARKET_DATA_RECEIVED


@dataclass
class SignalEvent(Event):
    symbol: str
    signal_type: str
    strength: float
    strategy_name: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.SIGNAL_GENERATED


@dataclass
class OrderEvent(Event):
    symbol: str
    order_type: str
    side: str
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "PENDING"
    
    @property
    def event_type(self) -> EventType:
        return EventType.ORDER_CREATED


@dataclass
class FillEvent(Event):
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
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    
    @property
    def event_type(self) -> EventType:
        return EventType.POSITION_UPDATED


@dataclass
class RiskEvent(Event):
    risk_type: str
    severity: str
    message: str
    current_value: float
    threshold_value: float
    symbol: Optional[str] = None
    
    @property
    def event_type(self) -> EventType:
        return EventType.RISK_THRESHOLD_EXCEEDED


@dataclass
class SystemEvent(Event):
    system_type: EventType
    message: str
    component: Optional[str] = None
    error_details: Optional[str] = None
    
    @property
    def event_type(self) -> EventType:
        return self.system_type


class EventHandler(ABC):
    @abstractmethod
    async def handle(self, event: Event) -> None:
        ...
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        ...


class EventBus:
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._running = False
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processing_task: Optional[asyncio.Task] = None
        logger.info("Event bus initialized")
    
    async def start(self) -> None:
        if self._running:
            logger.warning("Event bus already running")
            return
        self._running = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
        await self.publish(SystemEvent(system_type=EventType.SYSTEM_STARTED, message="Event bus started", component="event_bus"))
    
    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        await self.publish(SystemEvent(system_type=EventType.SYSTEM_STOPPED, message="Event bus stopping", component="event_bus"))
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Event bus stopped")
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, [])
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"Subscribed handler to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from {event_type.value}")
    
    async def publish(self, event: Event) -> None:
        try:
            await self._event_queue.put(event)
            logger.debug(f"Published event: {event.event_type.value}")
        except asyncio.QueueFull:
            logger.error("Event queue full, dropping event")
    
    async def _process_events(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            logger.debug(f"No handlers for event: {event.event_type.value}")
            return
        tasks = [asyncio.create_task(self._safe_handle(h, event)) for h in handlers if h.can_handle(event.event_type)]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_handle(self, handler: EventHandler, event: Event) -> None:
        try:
            await handler.handle(event)
        except Exception as e:
            logger.error(f"Handler error for {event.event_type.value}: {e}")
            error_event = SystemEvent(system_type=EventType.SYSTEM_ERROR, message=f"Handler error: {e}", component="event_bus", error_details=str(e))
            asyncio.create_task(self._event_queue.put(error_event))
    
    def get_event_history(self, event_type: Optional[EventType] = None) -> List[Event]:
        if event_type:
            return [e for e in self._event_history if e.event_type == event_type]
        return self._event_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for e in self._event_history:
            key = e.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return {
            "running": self._running,
            "total_events": len(self._event_history),
            "queue_size": self._event_queue.qsize(),
            "handler_count": sum(len(v) for v in self._handlers.values()),
            "event_type_counts": counts,
        }


# Global event bus instance
event_bus = EventBus()
