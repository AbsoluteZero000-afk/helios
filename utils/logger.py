"""
Helios Logging System

Centralized logging configuration using Loguru with enhanced formatting,
file rotation, and Slack integration for critical errors.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
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
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
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
        diagnose=True
    )
    
    # File handler with rotation if enabled
    if settings.log_to_file:
        # Ensure log directory exists
        log_path = Path(settings.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            settings.log_file_path,
            format=file_format,
            level=level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Set initial log message
    logger.info(f"Helios logging initialized - Level: {level}")
    logger.info(f"Environment: {settings.environment.value}")
    
    if settings.debug:
        logger.debug("Debug mode enabled")


def get_logger(name: str = None):
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Logger name for identification
        
    Returns:
        Configured logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class ContextLogger:
    """
    Context-aware logger for trading operations.
    
    Provides structured logging with trading-specific context.
    """
    
    def __init__(self, component: str):
        """
        Initialize context logger.
        
        Args:
            component: Component name (e.g., 'strategy', 'risk', 'execution')
        """
        self.component = component
        self.logger = logger.bind(component=component)
    
    def trade(self, message: str, symbol: str = None, **kwargs):
        """Log trading-related messages."""
        context = {"symbol": symbol, **kwargs}
        self.logger.info(f"🔄 TRADE: {message}", **context)
    
    def risk(self, message: str, level: str = "WARNING", **kwargs):
        """Log risk management messages."""
        if level.upper() == "ERROR":
            self.logger.error(f"⚠️ RISK: {message}", **kwargs)
        else:
            self.logger.warning(f"⚠️ RISK: {message}", **kwargs)
    
    def system(self, message: str, level: str = "INFO", **kwargs):
        """Log system-related messages."""
        if level.upper() == "ERROR":
            self.logger.error(f"💥 SYSTEM: {message}", **kwargs)
        elif level.upper() == "WARNING":
            self.logger.warning(f"💥 SYSTEM: {message}", **kwargs)
        else:
            self.logger.info(f"💥 SYSTEM: {message}", **kwargs)
    
    def performance(self, message: str, **kwargs):
        """Log performance metrics."""
        self.logger.info(f"📊 PERF: {message}", **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug messages."""
        self.logger.debug(f"🔍 DEBUG: {message}", **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info messages."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning messages."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error messages."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical messages."""
        self.logger.critical(message, **kwargs)


# Pre-configured component loggers
core_logger = ContextLogger("core")
strategy_logger = ContextLogger("strategy")
risk_logger = ContextLogger("risk")
execution_logger = ContextLogger("execution")
data_logger = ContextLogger("data")
sentinel_logger = ContextLogger("sentinel")
