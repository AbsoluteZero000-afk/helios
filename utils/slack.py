"""
Helios Slack Integration

Enhanced Slack notification system with emoji, color coding,
and contextual trading alerts.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import aiohttp
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("slack")


class MessageType(str, Enum):
    """Slack message types with corresponding emoji and colors."""
    TRADE = "trade"
    RISK = "risk"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    ERROR = "error"
    SUCCESS = "success"
    WARNING = "warning"
    INFO = "info"


class SlackNotifier:
    """
    Advanced Slack notification system for trading alerts.
    
    Provides structured, emoji-enhanced messages with color coding
    and contextual information for trading operations.
    """
    
    # Message type configurations
    MESSAGE_CONFIG = {
        MessageType.TRADE: {
            "emoji": "✅",
            "prefix": "TRADE",
            "color": "good"
        },
        MessageType.RISK: {
            "emoji": "⚠️",
            "prefix": "RISK",
            "color": "warning"
        },
        MessageType.SYSTEM: {
            "emoji": "💥",
            "prefix": "SYSTEM",
            "color": "#36a64f"
        },
        MessageType.PERFORMANCE: {
            "emoji": "📊",
            "prefix": "PERF",
            "color": "#439FE0"
        },
        MessageType.ERROR: {
            "emoji": "🔴",
            "prefix": "ERROR",
            "color": "danger"
        },
        MessageType.SUCCESS: {
            "emoji": "☀️",
            "prefix": "SUCCESS",
            "color": "good"
        },
        MessageType.WARNING: {
            "emoji": "🟡",
            "prefix": "WARNING",
            "color": "warning"
        },
        MessageType.INFO: {
            "emoji": "🔵",
            "prefix": "INFO",
            "color": "#36a64f"
        }
    }
    
    def __init__(self):
        """Initialize Slack notifier with settings."""
        self.settings = get_settings()
        self.webhook_url = self.settings.slack_webhook_url
        self.channel = self.settings.slack_channel
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Slack notifications disabled - no webhook URL configured")
    
    async def send_message(
        self,
        message: str,
        message_type: MessageType = MessageType.INFO,
        fields: Optional[Dict[str, Any]] = None,
        channel: Optional[str] = None
    ) -> bool:
        """
        Send enhanced message to Slack.
        
        Args:
            message: Main message content
            message_type: Type of message for emoji and color
            fields: Additional structured data
            channel: Override default channel
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.enabled:
            logger.debug(f"Slack disabled - would send: {message}")
            return False
        
        try:
            config = self.MESSAGE_CONFIG[message_type]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build the message payload
            payload = {
                "channel": channel or self.channel,
                "username": "Helios Trading Bot",
                "icon_emoji": ":robot_face:",
                "attachments": [
                    {
                        "color": config["color"],
                        "title": f"{config['emoji']} {config['prefix']}",
                        "text": message,
                        "footer": "Helios v2",
                        "ts": int(datetime.now().timestamp()),
                        "fields": []
                    }
                ]
            }
            
            # Add structured fields if provided
            if fields:
                for key, value in fields.items():
                    payload["attachments"][0]["fields"].append({
                        "title": key.replace("_", " ").title(),
                        "value": str(value),
                        "short": len(str(value)) < 30
                    })
            
            # Send the message
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Slack message sent: {message_type.value}")
                        return True
                    else:
                        logger.error(f"Slack API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def trade_alert(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str = None,
        **kwargs
    ) -> bool:
        """
        Send trading action alert.
        
        Args:
            action: Trade action (BUY, SELL, etc.)
            symbol: Trading symbol
            quantity: Position size
            price: Execution price
            strategy: Strategy name
            **kwargs: Additional context
            
        Returns:
            bool: True if sent successfully
        """
        message = f"{action} {quantity:.4f} {symbol} @ ${price:.2f}"
        
        fields = {
            "Symbol": symbol,
            "Action": action,
            "Quantity": f"{quantity:.4f}",
            "Price": f"${price:.2f}",
            "Value": f"${quantity * price:.2f}"
        }
        
        if strategy:
            fields["Strategy"] = strategy
        
        fields.update(kwargs)
        
        return await self.send_message(
            message,
            MessageType.TRADE,
            fields
        )
    
    async def risk_alert(
        self,
        message: str,
        risk_level: str = "medium",
        **kwargs
    ) -> bool:
        """
        Send risk management alert.
        
        Args:
            message: Risk alert message
            risk_level: Risk severity level
            **kwargs: Additional context
            
        Returns:
            bool: True if sent successfully
        """
        fields = {"Risk_Level": risk_level.upper()}
        fields.update(kwargs)
        
        return await self.send_message(
            message,
            MessageType.RISK,
            fields
        )
    
    async def system_alert(
        self,
        message: str,
        component: str = None,
        severity: str = "info",
        **kwargs
    ) -> bool:
        """
        Send system status alert.
        
        Args:
            message: System message
            component: System component name
            severity: Alert severity
            **kwargs: Additional context
            
        Returns:
            bool: True if sent successfully
        """
        message_type = {
            "error": MessageType.ERROR,
            "warning": MessageType.WARNING,
            "success": MessageType.SUCCESS,
            "info": MessageType.SYSTEM
        }.get(severity.lower(), MessageType.SYSTEM)
        
        fields = {}
        if component:
            fields["Component"] = component
        fields["Severity"] = severity.upper()
        fields.update(kwargs)
        
        return await self.send_message(
            message,
            message_type,
            fields
        )
    
    async def performance_report(
        self,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Send performance metrics report.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            bool: True if sent successfully
        """
        total_return = metrics.get("total_return", 0)
        message = f"Performance Update: {total_return:.2%} total return"
        
        return await self.send_message(
            message,
            MessageType.PERFORMANCE,
            metrics
        )
    
    def send_sync(
        self,
        message: str,
        message_type: MessageType = MessageType.INFO,
        **kwargs
    ) -> bool:
        """
        Synchronous message sending wrapper.
        
        Args:
            message: Message content
            message_type: Message type
            **kwargs: Additional arguments
            
        Returns:
            bool: True if sent successfully
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.send_message(message, message_type, **kwargs)
            )
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.send_message(message, message_type, **kwargs)
                )
            finally:
                loop.close()


# Global notifier instance
slack_notifier = SlackNotifier()


# Convenience functions
async def send_trade_alert(action: str, symbol: str, quantity: float, price: float, **kwargs):
    """Convenience function for trade alerts."""
    return await slack_notifier.trade_alert(action, symbol, quantity, price, **kwargs)


async def send_risk_alert(message: str, **kwargs):
    """Convenience function for risk alerts."""
    return await slack_notifier.risk_alert(message, **kwargs)


async def send_system_alert(message: str, **kwargs):
    """Convenience function for system alerts."""
    return await slack_notifier.system_alert(message, **kwargs)
