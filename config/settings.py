"""
Helios Configuration Management

Centralized configuration system using Pydantic for type validation
and environment variable management.
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum


class Environment(str, Enum):
    """Environment types for deployment configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels for application output."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Helios application settings with validation.
    
    All settings can be overridden via environment variables.
    """
    
    # Application
    app_name: str = "Helios"
    version: str = "2.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/helios",
        env="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # Trading APIs
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    fmp_api_key: Optional[str] = Field(default=None, env="FMP_API_KEY")
    
    # Security
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET")
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "logs/helios.log"
    
    # Trading Parameters
    max_portfolio_risk: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Maximum portfolio risk as decimal (e.g., 0.02 = 2%)"
    )
    max_daily_drawdown: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Maximum daily drawdown threshold"
    )
    default_position_size: float = Field(
        default=0.01,
        ge=0.001,
        le=0.1,
        description="Default position size as portfolio percentage"
    )
    max_open_positions: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of concurrent open positions"
    )
    
    # Notifications
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    slack_channel: str = Field(default="#trading-alerts", env="SLACK_CHANNEL")
    
    # System Monitoring
    sentinel_enabled: bool = True
    sentinel_check_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Sentinel system check interval in seconds"
    )
    health_check_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval in seconds"
    )
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(('postgresql://', 'sqlite://')):
            raise ValueError('Database URL must be PostgreSQL or SQLite')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings.
    
    Returns:
        Settings: Configured application settings
    """
    return settings


def is_production() -> bool:
    """
    Check if running in production environment.
    
    Returns:
        bool: True if production environment
    """
    return settings.environment == Environment.PRODUCTION


def is_development() -> bool:
    """
    Check if running in development environment.
    
    Returns:
        bool: True if development environment
    """
    return settings.environment == Environment.DEVELOPMENT
