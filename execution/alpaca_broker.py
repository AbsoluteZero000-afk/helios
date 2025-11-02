"""
Helios Alpaca Broker Integration v3

Production-ready Alpaca Markets broker integration with comprehensive
error handling, rate limiting, and order management.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import json

from alpaca_trade_api import REST, Stream
from alpaca_trade_api.common import URL
from alpaca_trade_api.entity import Order as AlpacaOrder, Position as AlpacaPosition

from execution.broker_base import (
    BrokerBase, OrderRequest, OrderResponse, Position, AccountInfo,
    OrderType, OrderSide, OrderStatus, TimeInForce
)
from config.settings import get_settings
from utils.logger import execution_logger as logger
from utils.slack import send_trade_alert, send_system_alert


class AlpacaBroker(BrokerBase):
    """
    Alpaca Markets broker implementation.
    
    Provides full integration with Alpaca's REST API for order execution,
    position management, and account information.
    """
    
    # Alpaca API status mapping
    STATUS_MAPPING = {
        "new": OrderStatus.SUBMITTED,
        "accepted": OrderStatus.SUBMITTED,
        "pending_new": OrderStatus.PENDING,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "done_for_day": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "replaced": OrderStatus.SUBMITTED,
        "pending_cancel": OrderStatus.PENDING,
        "pending_replace": OrderStatus.PENDING,
        "rejected": OrderStatus.REJECTED,
        "suspended": OrderStatus.REJECTED,
        "calculated": OrderStatus.SUBMITTED,
    }
    
    def __init__(self):
        """Initialize Alpaca broker."""
        super().__init__("alpaca")
        self.settings = get_settings()
        self._api_client: Optional[REST] = None
        self._stream_client: Optional[Stream] = None
        self._rate_limit_delay = 0.1  # 100ms between requests
        self._last_request_time = 0.0
        
        # Validate configuration
        if not self.settings.alpaca_api_key or not self.settings.alpaca_secret_key:
            raise ValueError("Alpaca API credentials not configured")
    
    async def connect(self) -> bool:
        """
        Connect to Alpaca API.
        
        Returns:
            bool: True if connection successful
        """
        try:
            # Initialize REST API client
            self._api_client = REST(
                key_id=self.settings.alpaca_api_key,
                secret_key=self.settings.alpaca_secret_key,
                base_url=URL(self.settings.alpaca_base_url),
                api_version='v2'
            )
            
            # Test connection by getting account info
            account = self._api_client.get_account()
            
            if account:
                self._connected = True
                logger.info(f"Connected to Alpaca: Account {account.id} ({account.status})")
                
                await send_system_alert(
                    f"Alpaca broker connected: {account.id}",
                    component="alpaca",
                    severity="success",
                    account_status=account.status,
                    buying_power=float(account.buying_power)
                )
                
                return True
            else:
                logger.error("Alpaca connection failed: No account information received")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            await send_system_alert(
                f"Alpaca connection failed: {e}",
                component="alpaca",
                severity="error"
            )
            return False
    
    async def disconnect(self) -> None:
        """
        Disconnect from Alpaca API.
        """
        self._connected = False
        self._api_client = None
        
        if self._stream_client:
            # Close stream connections if any
            self._stream_client = None
        
        logger.info("Disconnected from Alpaca")
    
    async def _rate_limit(self) -> None:
        """
        Apply rate limiting between API requests.
        """
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """
        Convert internal order type to Alpaca format.
        
        Args:
            order_type: Internal order type
            
        Returns:
            str: Alpaca order type
        """
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop"
        }
        return mapping.get(order_type, "market")
    
    def _convert_time_in_force(self, tif: TimeInForce) -> str:
        """
        Convert internal time in force to Alpaca format.
        
        Args:
            tif: Internal time in force
            
        Returns:
            str: Alpaca time in force
        """
        mapping = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok"
        }
        return mapping.get(tif, "day")
    
    def _convert_alpaca_order(self, alpaca_order: AlpacaOrder) -> OrderResponse:
        """
        Convert Alpaca order to internal format.
        
        Args:
            alpaca_order: Alpaca order object
            
        Returns:
            OrderResponse: Internal order representation
        """
        status = self.STATUS_MAPPING.get(alpaca_order.status, OrderStatus.PENDING)
        
        return OrderResponse(
            order_id=alpaca_order.id,
            client_order_id=alpaca_order.client_order_id,
            symbol=alpaca_order.symbol,
            side=OrderSide(alpaca_order.side.lower()),
            order_type=OrderType(alpaca_order.order_type),
            quantity=float(alpaca_order.qty),
            price=float(alpaca_order.limit_price) if alpaca_order.limit_price else None,
            status=status,
            submitted_at=alpaca_order.submitted_at,
            filled_quantity=float(alpaca_order.filled_qty or 0),
            average_fill_price=float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else None,
            commission=0.0,  # Alpaca doesn't charge commissions for stocks
            metadata={
                "alpaca_status": alpaca_order.status,
                "time_in_force": alpaca_order.time_in_force,
                "extended_hours": alpaca_order.extended_hours,
            }
        )
    
    def _convert_alpaca_position(self, alpaca_position: AlpacaPosition) -> Position:
        """
        Convert Alpaca position to internal format.
        
        Args:
            alpaca_position: Alpaca position object
            
        Returns:
            Position: Internal position representation
        """
        return Position(
            symbol=alpaca_position.symbol,
            quantity=float(alpaca_position.qty),
            market_value=float(alpaca_position.market_value),
            cost_basis=float(alpaca_position.cost_basis),
            average_entry_price=float(alpaca_position.avg_entry_price),
            unrealized_pnl=float(alpaca_position.unrealized_pl),
            unrealized_pnl_percent=float(alpaca_position.unrealized_plpc),
            last_price=float(alpaca_position.current_price),
            updated_at=datetime.now(timezone.utc),
            metadata={
                "side": alpaca_position.side,
                "exchange": getattr(alpaca_position, 'exchange', None)
            }
        )
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Place an order with Alpaca.
        
        Args:
            order_request: Order details
            
        Returns:
            OrderResponse: Order placement result
        """
        if not self._connected:
            raise RuntimeError("Alpaca broker not connected")
        
        # Validate order
        is_valid, error_message = await self.validate_order(order_request)
        if not is_valid:
            raise ValueError(f"Invalid order: {error_message}")
        
        # Apply rate limiting
        await self._rate_limit()
        
        try:
            # Convert to Alpaca format
            alpaca_order_type = self._convert_order_type(order_request.order_type)
            alpaca_tif = self._convert_time_in_force(order_request.time_in_force)
            
            # Prepare order parameters
            order_params = {
                "symbol": order_request.symbol,
                "qty": int(order_request.quantity) if order_request.quantity.is_integer() else order_request.quantity,
                "side": order_request.side.value,
                "type": alpaca_order_type,
                "time_in_force": alpaca_tif,
            }
            
            # Add price parameters if needed
            if order_request.price:
                order_params["limit_price"] = order_request.price
            
            if order_request.stop_price:
                order_params["stop_price"] = order_request.stop_price
            
            if order_request.client_order_id:
                order_params["client_order_id"] = order_request.client_order_id
            
            # Submit order to Alpaca
            alpaca_order = self._api_client.submit_order(**order_params)
            
            # Convert response
            order_response = self._convert_alpaca_order(alpaca_order)
            
            logger.info(
                f"Order placed with Alpaca: {order_response.order_id} "
                f"({order_response.symbol} {order_response.side.value} {order_response.quantity})"
            )
            
            # Send trade alert
            await send_trade_alert(
                action=order_response.side.value.upper(),
                symbol=order_response.symbol,
                quantity=order_response.quantity,
                price=order_response.price or 0.0,
                strategy="alpaca_order",
                order_id=order_response.order_id
            )
            
            return order_response
            
        except Exception as e:
            logger.error(f"Failed to place Alpaca order: {e}")
            
            await send_system_alert(
                f"Alpaca order placement failed: {e}",
                component="alpaca",
                severity="error",
                symbol=order_request.symbol,
                side=order_request.side.value
            )
            
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order with Alpaca.
        
        Args:
            order_id: Alpaca order ID
            
        Returns:
            bool: True if cancellation successful
        """
        if not self._connected:
            raise RuntimeError("Alpaca broker not connected")
        
        await self._rate_limit()
        
        try:
            self._api_client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            
            await send_system_alert(
                f"Order cancelled: {order_id}",
                component="alpaca",
                severity="info"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel Alpaca order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[OrderResponse]:
        """
        Get order information from Alpaca.
        
        Args:
            order_id: Order ID to retrieve
            
        Returns:
            Optional[OrderResponse]: Order information or None
        """
        if not self._connected:
            return None
        
        await self._rate_limit()
        
        try:
            alpaca_order = self._api_client.get_order(order_id)
            return self._convert_alpaca_order(alpaca_order)
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca order {order_id}: {e}")
            return None
    
    async def get_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[OrderResponse]:
        """
        Get list of orders from Alpaca.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders
            
        Returns:
            List[OrderResponse]: List of orders
        """
        if not self._connected:
            return []
        
        await self._rate_limit()
        
        try:
            # Convert status filter to Alpaca format
            alpaca_status = None
            if status:
                # Reverse lookup in status mapping
                for alpaca_st, internal_st in self.STATUS_MAPPING.items():
                    if internal_st == status:
                        alpaca_status = alpaca_st
                        break
            
            # Get orders from Alpaca
            alpaca_orders = self._api_client.list_orders(
                status=alpaca_status,
                limit=limit,
                symbols=symbol
            )
            
            # Convert to internal format
            orders = [self._convert_alpaca_order(order) for order in alpaca_orders]
            
            logger.debug(f"Retrieved {len(orders)} orders from Alpaca")
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca orders: {e}")
            return []
    
    async def get_positions(self) -> List[Position]:
        """
        Get all positions from Alpaca.
        
        Returns:
            List[Position]: List of positions
        """
        if not self._connected:
            return []
        
        await self._rate_limit()
        
        try:
            alpaca_positions = self._api_client.list_positions()
            positions = [self._convert_alpaca_position(pos) for pos in alpaca_positions]
            
            logger.debug(f"Retrieved {len(positions)} positions from Alpaca")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return []
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol from Alpaca.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Optional[Position]: Position information or None
        """
        if not self._connected:
            return None
        
        await self._rate_limit()
        
        try:
            alpaca_position = self._api_client.get_position(symbol)
            return self._convert_alpaca_position(alpaca_position)
            
        except Exception as e:
            # Position not found is expected for symbols we don't hold
            if "position does not exist" in str(e).lower():
                return None
            
            logger.error(f"Failed to get Alpaca position for {symbol}: {e}")
            return None
    
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information from Alpaca.
        
        Returns:
            AccountInfo: Account details
        """
        if not self._connected:
            raise RuntimeError("Alpaca broker not connected")
        
        await self._rate_limit()
        
        try:
            account = self._api_client.get_account()
            
            return AccountInfo(
                account_id=account.id,
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                initial_margin=float(account.initial_margin or 0),
                maintenance_margin=float(account.maintenance_margin or 0),
                day_trade_count=int(account.daytrade_count),
                pattern_day_trader=account.pattern_day_trader,
                updated_at=datetime.now(timezone.utc),
                metadata={
                    "status": account.status,
                    "trading_blocked": account.trading_blocked,
                    "transfers_blocked": account.transfers_blocked,
                    "account_blocked": account.account_blocked,
                    "created_at": str(account.created_at),
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get Alpaca account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data from Alpaca.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Optional[Dict[str, Any]]: Market data or None
        """
        if not self._connected:
            return None
        
        await self._rate_limit()
        
        try:
            # Get latest trade
            latest_trade = self._api_client.get_latest_trade(symbol)
            
            if latest_trade:
                return {
                    "symbol": symbol,
                    "price": float(latest_trade.price),
                    "size": int(latest_trade.size),
                    "timestamp": latest_trade.timestamp.isoformat(),
                    "exchange": latest_trade.exchange,
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def get_market_hours(self) -> Dict[str, Any]:
        """
        Get market hours information from Alpaca.
        
        Returns:
            Dict[str, Any]: Market hours information
        """
        if not self._connected:
            return {}
        
        await self._rate_limit()
        
        try:
            calendar = self._api_client.get_calendar()
            today = datetime.now().date()
            
            today_schedule = None
            for day in calendar:
                if day.date.date() == today:
                    today_schedule = day
                    break
            
            if today_schedule:
                return {
                    "date": today_schedule.date.isoformat(),
                    "open": today_schedule.open.isoformat(),
                    "close": today_schedule.close.isoformat(),
                    "is_open": self._api_client.get_clock().is_open,
                    "next_open": self._api_client.get_clock().next_open.isoformat(),
                    "next_close": self._api_client.get_clock().next_close.isoformat(),
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {}
    
    async def close_all_positions(self) -> List[OrderResponse]:
        """
        Close all open positions.
        
        Returns:
            List[OrderResponse]: List of close orders placed
        """
        if not self._connected:
            return []
        
        try:
            positions = await self.get_positions()
            close_orders = []
            
            for position in positions:
                if position.quantity != 0:
                    # Determine order side (opposite of position)
                    order_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                    
                    close_order = OrderRequest(
                        symbol=position.symbol,
                        side=order_side,
                        order_type=OrderType.MARKET,
                        quantity=abs(position.quantity),
                        client_order_id=f"close_{position.symbol}_{datetime.now().timestamp()}"
                    )
                    
                    order_response = await self.place_order(close_order)
                    close_orders.append(order_response)
                    
                    logger.info(f"Placed close order for {position.symbol}: {order_response.order_id}")
            
            if close_orders:
                await send_system_alert(
                    f"Closed {len(close_orders)} positions",
                    component="alpaca",
                    severity="warning",
                    positions_closed=len(close_orders)
                )
            
            return close_orders
            
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get broker connection information.
        
        Returns:
            Dict[str, Any]: Connection information
        """
        return {
            "name": self.name,
            "connected": self._connected,
            "base_url": self.settings.alpaca_base_url,
            "data_feed": self.settings.alpaca_data_feed,
            "rate_limit_delay": self._rate_limit_delay,
            "last_request_time": self._last_request_time,
        }
