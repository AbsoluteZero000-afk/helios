"""
Helios Broker Base Class v3

Abstract base class defining the broker interface for order execution,
position management, and account information.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderType(str, Enum):
    """Order types supported by brokers."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


@dataclass
class OrderRequest:
    """Order request data structure."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrderResponse:
    """Order response data structure."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    submitted_at: datetime
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position data structure."""
    symbol: str
    quantity: float
    market_value: float
    cost_basis: float
    average_entry_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    last_price: float
    updated_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccountInfo:
    """Account information data structure."""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    equity: float
    initial_margin: float
    maintenance_margin: float
    day_trade_count: int
    pattern_day_trader: bool
    updated_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BrokerBase(ABC):
    """
    Abstract base class for all broker implementations.
    
    Defines the standard interface for order execution, position management,
    and account information retrieval.
    """
    
    def __init__(self, name: str):
        """Initialize broker base."""
        self.name = name
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the broker.
        """
        pass
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place an order with the broker.
        
        Args:
            order_request: Order details
            
        Returns:
            OrderResponse: Order placement result
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get order information.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Optional[OrderResponse]: Order information or None
        """
        pass
    
    @abstractmethod
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """
        Get list of orders.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders to return
            
        Returns:
            List[OrderResponse]: List of orders
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List[Position]: List of positions
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Optional[Position]: Position information or None
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information.
        
        Returns:
            AccountInfo: Account details
        """
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Optional[Dict[str, Any]]: Market data or None
        """
        pass
    
    async def validate_order(self, order_request: OrderRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate an order request before submission.
        
        Args:
            order_request: Order to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Basic validation
        if order_request.quantity <= 0:
            return False, "Quantity must be positive"
        
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not order_request.price or order_request.price <= 0:
                return False, "Price must be specified and positive for limit orders"
        
        if order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP]:
            if not order_request.stop_price or order_request.stop_price <= 0:
                return False, "Stop price must be specified and positive for stop orders"
        
        return True, None
