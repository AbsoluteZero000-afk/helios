"""
Helios Enhanced Logger v3

Centralized Loguru-based logging with structured output, context binding,
file rotation, compression, and integration with monitoring systems.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import json
from datetime import datetime, timezone

from config.settings import get_settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure Loguru logging with enhanced formatting and file rotation.
    
    Args:
        log_level: Override default log level from settings
    """
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Determine log level
    level = log_level or settings.log_level.value
    
    # Console handler with color formatting
    if settings.log_json_format:
        # JSON format for structured logging
        def json_formatter(record):
            json_record = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "function": record["function"],
                "line": record["line"],
                "message": record["message"],
            }
            
            # Add extra context if present
            if record["extra"]:
                json_record.update(record["extra"])
            
            return json.dumps(json_record)
        
        logger.add(
            sys.stderr,
            format=json_formatter,
            level=level,
            backtrace=True,
            diagnose=not settings.is_production()
        )
    else:
        # Human-readable format
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=not settings.is_production()
        )
    
    # File handler with rotation if enabled
    if settings.log_to_file:
        # Ensure log directory exists
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if settings.log_json_format:
            # JSON file format
            def json_file_formatter(record):
                json_record = {
                    "timestamp": record["time"].isoformat(),
                    "level": record["level"].name,
                    "logger": record["name"],
                    "module": record["module"],
                    "function": record["function"],
                    "line": record["line"],
                    "message": record["message"],
                    "thread": record["thread"].name,
                    "process": record["process"].name,
                }
                
                # Add extra context if present
                if record["extra"]:
                    json_record["context"] = record["extra"]
                
                # Add exception info if present
                if record["exception"]:
                    json_record["exception"] = {
                        "type": record["exception"].type.__name__,
                        "value": str(record["exception"].value),
                        "traceback": record["exception"].traceback
                    }
                
                return json.dumps(json_record)
            
            logger.add(
                settings.log_file_path,
                format=json_file_formatter,
                level=level,
                rotation=settings.log_rotation_size,
                retention=f"{settings.log_retention_days} days",
                compression=settings.log_compression,
                backtrace=True,
                diagnose=not settings.is_production(),
                enqueue=True  # Thread-safe logging
            )
        else:
            # Human-readable file format
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            
            logger.add(
                settings.log_file_path,
                format=file_format,
                level=level,
                rotation=settings.log_rotation_size,
                retention=f"{settings.log_retention_days} days",
                compression=settings.log_compression,
                backtrace=True,
                diagnose=not settings.is_production(),
                enqueue=True
            )
    
    # Add custom handler for critical errors (could integrate with external monitoring)
    def critical_handler(record):
        if record["level"].name == "CRITICAL":
            # This could send to external monitoring service
            # For now, we'll just ensure it's prominently logged
            pass
    
    # Set initial log message
    logger.info(f"Helios v3 logging initialized - Level: {level}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"JSON format: {settings.log_json_format}")
    
    if settings.debug:
        logger.debug("Debug mode enabled - extensive diagnostics active")


def get_logger(name: str = None):
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Logger name for identification and filtering
        
    Returns:
        Configured logger instance with context binding
    """
    if name:
        return logger.bind(component=name)
    return logger


class ContextLogger:
    """
    Context-aware logger for trading operations.
    
    Provides structured logging with trading-specific context and
    automatic field enrichment for better observability.
    """
    
    def __init__(self, component: str):
        """
        Initialize context logger.
        
        Args:
            component: Component name (e.g., 'strategy', 'risk', 'execution')
        """
        self.component = component
        self.logger = logger.bind(component=component)
    
    def with_context(self, **context) -> 'ContextLogger':
        """
        Create new logger instance with additional context.
        
        Args:
            **context: Additional context fields
            
        Returns:
            ContextLogger: New logger with enhanced context
        """
        new_logger = ContextLogger(self.component)
        new_logger.logger = self.logger.bind(**context)
        return new_logger
    
    def trade(self, message: str, symbol: str = None, **kwargs):
        """Log trading-related messages with trade context."""
        context = {"event_type": "trade", "symbol": symbol, **kwargs}
        self.logger.bind(**context).info(f"🔄 TRADE: {message}")
    
    def risk(self, message: str, level: str = "WARNING", **kwargs):
        """Log risk management messages with risk context."""
        context = {"event_type": "risk", "risk_level": level, **kwargs}
        
        if level.upper() == "CRITICAL":
            self.logger.bind(**context).critical(f"⚠️ RISK: {message}")
        elif level.upper() == "ERROR":
            self.logger.bind(**context).error(f"⚠️ RISK: {message}")
        else:
            self.logger.bind(**context).warning(f"⚠️ RISK: {message}")
    
    def system(self, message: str, level: str = "INFO", **kwargs):
        """Log system-related messages with system context."""
        context = {"event_type": "system", "system_level": level, **kwargs}
        
        if level.upper() == "CRITICAL":
            self.logger.bind(**context).critical(f"💥 SYSTEM: {message}")
        elif level.upper() == "ERROR":
            self.logger.bind(**context).error(f"💥 SYSTEM: {message}")
        elif level.upper() == "WARNING":
            self.logger.bind(**context).warning(f"💥 SYSTEM: {message}")
        else:
            self.logger.bind(**context).info(f"💥 SYSTEM: {message}")
    
    def performance(self, message: str, **kwargs):
        """Log performance metrics with performance context."""
        context = {"event_type": "performance", **kwargs}
        self.logger.bind(**context).info(f"📊 PERF: {message}")
    
    def execution(self, message: str, order_id: str = None, **kwargs):
        """Log order execution messages."""
        context = {"event_type": "execution", "order_id": order_id, **kwargs}
        self.logger.bind(**context).info(f"🎯 EXEC: {message}")
    
    def data(self, message: str, data_type: str = None, **kwargs):
        """Log data processing messages."""
        context = {"event_type": "data", "data_type": data_type, **kwargs}
        self.logger.bind(**context).debug(f"📊 DATA: {message}")
    
    def sentinel(self, message: str, repair_type: str = None, **kwargs):
        """Log System Sentinel messages."""
        context = {"event_type": "sentinel", "repair_type": repair_type, **kwargs}
        self.logger.bind(**context).info(f"🤖 SENTINEL: {message}")
    
    def debug(self, message: str, **kwargs):
        """Log debug messages with context."""
        if kwargs:
            self.logger.bind(**kwargs).debug(message)
        else:
            self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """Log info messages with context."""
        if kwargs:
            self.logger.bind(**kwargs).info(message)
        else:
            self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning messages with context."""
        if kwargs:
            self.logger.bind(**kwargs).warning(message)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error messages with context."""
        if kwargs:
            self.logger.bind(**kwargs).error(message)
        else:
            self.logger.error(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical messages with context."""
        if kwargs:
            self.logger.bind(**kwargs).critical(message)
        else:
            self.logger.critical(message)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        if kwargs:
            self.logger.bind(**kwargs).exception(message)
        else:
            self.logger.exception(message)


# Enhanced logger utilities
def log_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    strategy: str = None,
    order_id: str = None,
    **kwargs
) -> None:
    """
    Log trade execution with structured data.
    
    Args:
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Shares/units traded
        price: Execution price
        strategy: Strategy name
        order_id: Order identifier
        **kwargs: Additional context
    """
    context = {
        "event_type": "trade_execution",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "value": quantity * price,
        "strategy": strategy,
        "order_id": order_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    logger.bind(**context).info(
        f"🔄 TRADE EXECUTED: {side} {quantity:,.0f} {symbol} @ ${price:.2f} "
        f"(Value: ${quantity * price:,.2f})"
    )


def log_risk(
    event_type: str,
    severity: str,
    message: str,
    symbol: str = None,
    portfolio_value: float = None,
    **kwargs
) -> None:
    """
    Log risk management events with structured data.
    
    Args:
        event_type: Type of risk event
        severity: Risk severity level
        message: Risk message
        symbol: Affected symbol
        portfolio_value: Current portfolio value
        **kwargs: Additional context
    """
    context = {
        "event_type": "risk_management",
        "risk_event_type": event_type,
        "severity": severity,
        "symbol": symbol,
        "portfolio_value": portfolio_value,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    log_func = logger.critical if severity == "CRITICAL" else logger.warning
    log_func = log_func.bind(**context)
    
    log_func(f"⚠️ RISK {severity}: {message}")


def log_system(
    component: str,
    event: str,
    status: str,
    details: str = None,
    **kwargs
) -> None:
    """
    Log system events with structured data.
    
    Args:
        component: System component
        event: Event description
        status: Event status (success, warning, error)
        details: Additional details
        **kwargs: Additional context
    """
    context = {
        "event_type": "system_event",
        "component": component,
        "event": event,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    if details:
        context["details"] = details
    
    if status.lower() == "error":
        log_func = logger.error
        emoji = "💥"
    elif status.lower() == "warning":
        log_func = logger.warning
        emoji = "⚠️"
    else:
        log_func = logger.info
        emoji = "✅"
    
    log_func.bind(**context)(f"{emoji} SYSTEM: {component} - {event} ({status})")


def log_performance(
    metric_type: str,
    value: float,
    benchmark: float = None,
    period: str = None,
    **kwargs
) -> None:
    """
    Log performance metrics with structured data.
    
    Args:
        metric_type: Type of performance metric
        value: Metric value
        benchmark: Benchmark comparison value
        period: Time period for metric
        **kwargs: Additional context
    """
    context = {
        "event_type": "performance_metric",
        "metric_type": metric_type,
        "value": value,
        "benchmark": benchmark,
        "period": period,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    benchmark_text = f" (benchmark: {benchmark})" if benchmark else ""
    period_text = f" [{period}]" if period else ""
    
    logger.bind(**context).info(
        f"📊 PERFORMANCE: {metric_type} = {value:.4f}{benchmark_text}{period_text}"
    )


# Pre-configured component loggers with enhanced context
core_logger = ContextLogger("core")
strategy_logger = ContextLogger("strategy")
risk_logger = ContextLogger("risk")
execution_logger = ContextLogger("execution")
data_logger = ContextLogger("data")
sentinel_logger = ContextLogger("sentinel")
celery_logger = ContextLogger("celery")
database_logger = ContextLogger("database")
redis_logger = ContextLogger("redis")
health_logger = ContextLogger("health")


# Structured logging helpers
def create_audit_log(
    action: str,
    component: str,
    result: str,
    details: Dict[str, Any] = None,
    user: str = None
) -> None:
    """
    Create audit log entry for compliance and tracking.
    
    Args:
        action: Action performed
        component: Component affected
        result: Action result
        details: Additional details
        user: User performing action
    """
    audit_context = {
        "event_type": "audit",
        "action": action,
        "component": component,
        "result": result,
        "user": user or "system",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    if details:
        audit_context["details"] = details
    
    logger.bind(**audit_context).info(f"📜 AUDIT: {user or 'SYSTEM'} {action} {component} - {result}")


def create_security_log(
    event_type: str,
    severity: str,
    description: str,
    source_ip: str = None,
    user_agent: str = None,
    **kwargs
) -> None:
    """
    Create security-related log entry.
    
    Args:
        event_type: Type of security event
        severity: Security severity level
        description: Event description
        source_ip: Source IP address
        user_agent: User agent string
        **kwargs: Additional context
    """
    security_context = {
        "event_type": "security",
        "security_event_type": event_type,
        "severity": severity,
        "source_ip": source_ip,
        "user_agent": user_agent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    log_func = logger.critical if severity == "CRITICAL" else logger.warning
    log_func.bind(**security_context)(f"🔒 SECURITY: {event_type} - {description}")
