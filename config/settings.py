"""
Helios Configuration Management v3

Enhanced configuration system with Pydantic for type validation,
environment variable management, and comprehensive trading parameters.
"""

import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseSettings, Field, validator, root_validator
from enum import Enum


class Environment(str, Enum):
    """Environment types for deployment configuration."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DOCKER = "docker"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels for application output."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradingMode(str, Enum):
    """Trading execution modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class Settings(BaseSettings):
    """
    Helios v3 application settings with comprehensive validation.
    
    All settings can be overridden via environment variables.
    """
    
    # Application Metadata
    app_name: str = "Helios"
    version: str = "3.0.0"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+psycopg2://helios:helios@localhost:5432/helios",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=10, ge=1, le=50)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_echo: bool = False
    database_pool_timeout: int = Field(default=30, ge=5, le=300)
    database_pool_recycle: int = Field(default=3600, ge=300, le=86400)
    
    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    redis_max_connections: int = Field(default=20, ge=1, le=100)
    redis_socket_keepalive: bool = True
    redis_socket_keepalive_options: Dict[str, int] = Field(
        default_factory=lambda: {"TCP_KEEPIDLE": 1, "TCP_KEEPINTVL": 3, "TCP_KEEPCNT": 5}
    )
    
    # Celery Configuration
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        env="CELERY_RESULT_BACKEND"
    )
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: List[str] = Field(default_factory=lambda: ["json"])
    celery_timezone: str = "UTC"
    celery_worker_prefetch_multiplier: int = Field(default=1, ge=1, le=10)
    celery_task_acks_late: bool = True
    celery_worker_max_tasks_per_child: int = Field(default=1000, ge=100, le=10000)
    
    # Trading APIs
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL"
    )
    alpaca_ws_url: str = Field(
        default="wss://stream.data.alpaca.markets/v2",
        env="ALPACA_WS_URL"
    )
    alpaca_data_feed: str = Field(default="iex", env="ALPACA_DATA_FEED")
    
    # Financial Market API
    fmp_api_key: Optional[str] = Field(default=None, env="FMP_API_KEY")
    fmp_base_url: str = Field(
        default="https://financialmodelingprep.com/api/v3",
        env="FMP_BASE_URL"
    )
    
    # Security
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET")
    api_secret_key: Optional[str] = Field(default=None, env="API_SECRET_KEY")
    
    # Logging Configuration
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_file_path: str = "/logs/helios.log"
    log_json_format: bool = False
    log_rotation_size: str = "100MB"
    log_retention_days: int = Field(default=30, ge=1, le=365)
    log_compression: str = "zip"
    
    # Trading Configuration
    initial_capital: float = Field(
        default=100000.0,
        ge=1000.0,
        le=10000000.0,
        description="Initial trading capital in USD"
    )
    trading_mode: TradingMode = TradingMode.PAPER
    max_portfolio_risk: float = Field(
        default=0.10,
        ge=0.001,
        le=0.5,
        description="Maximum portfolio risk as decimal (e.g., 0.10 = 10%)"
    )
    max_daily_drawdown: float = Field(
        default=0.20,
        ge=0.01,
        le=0.5,
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
        le=100,
        description="Maximum number of concurrent open positions"
    )
    min_order_value: float = Field(
        default=100.0,
        ge=1.0,
        le=10000.0,
        description="Minimum order value in USD"
    )
    max_order_value: float = Field(
        default=10000.0,
        ge=100.0,
        le=1000000.0,
        description="Maximum order value in USD"
    )
    
    # Risk Management
    risk_check_enabled: bool = True
    risk_position_limit: float = Field(
        default=0.05,
        ge=0.001,
        le=0.2,
        description="Maximum position size as portfolio percentage"
    )
    risk_correlation_limit: float = Field(
        default=0.30,
        ge=0.1,
        le=0.9,
        description="Maximum correlation between positions"
    )
    risk_var_confidence: float = Field(
        default=0.95,
        ge=0.9,
        le=0.99,
        description="VaR confidence level"
    )
    risk_lookback_days: int = Field(
        default=252,
        ge=20,
        le=1000,
        description="Risk calculation lookback period in days"
    )
    
    # Notifications (Slack)
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    slack_channel: str = Field(default="helios-alerts", env="SLACK_CHANNEL")
    slack_username: str = Field(default="Helios-Bot", env="SLACK_USERNAME")
    slack_icon_emoji: str = Field(default=":robot_face:", env="SLACK_ICON_EMOJI")
    slack_mention_channel: bool = False
    slack_mention_users: List[str] = Field(default_factory=list)
    
    # System Sentinel Configuration
    sentinel_enabled: bool = True
    sentinel_auto_repair: bool = True
    sentinel_max_repair_attempts: int = Field(default=3, ge=1, le=10)
    sentinel_check_interval: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Sentinel system check interval in seconds"
    )
    sentinel_health_endpoint: bool = True
    sentinel_repair_log_file: str = "/logs/sentinel_repairs.json"
    sentinel_db_validation: bool = True
    sentinel_redis_validation: bool = True
    sentinel_celery_validation: bool = True
    
    # Health Check Configuration
    health_check_enabled: bool = True
    health_check_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval in seconds"
    )
    health_check_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Health check timeout in seconds"
    )
    health_check_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Health check retry attempts"
    )
    
    # Performance Monitoring
    metrics_enabled: bool = True
    metrics_collection_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Metrics collection interval in seconds"
    )
    metrics_retention_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Metrics retention period in days"
    )
    metrics_detailed_logging: bool = False
    
    # Data Configuration
    data_storage_path: str = "/data"
    data_cache_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Data cache TTL in seconds"
    )
    data_batch_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Data processing batch size"
    )
    data_async_processing: bool = True
    
    # Strategy Configuration
    strategy_config_file: str = "config/strategy.yaml"
    strategy_reload_interval: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Strategy config reload interval in seconds"
    )
    strategy_validation_strict: bool = True
    
    # Container Configuration
    container_timezone: str = "UTC"
    container_user_id: int = Field(default=1000, ge=1, le=65535)
    container_group_id: int = Field(default=1000, ge=1, le=65535)
    container_memory_limit: str = "2G"
    container_cpu_limit: str = "1000m"
    
    @validator('database_url')
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v.startswith(('postgresql://', 'postgresql+psycopg2://', 'sqlite://')):
            raise ValueError('Database URL must be PostgreSQL or SQLite')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        """Validate Redis URL format."""
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    @validator('log_file_path')
    def validate_log_path(cls, v):
        """Ensure log directory exists or can be created."""
        log_dir = Path(v).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('data_storage_path')
    def validate_data_path(cls, v):
        """Ensure data directory exists or can be created."""
        data_dir = Path(v)
        data_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    @root_validator
    def validate_order_values(cls, values):
        """Validate order value constraints."""
        min_val = values.get('min_order_value', 100.0)
        max_val = values.get('max_order_value', 10000.0)
        
        if min_val >= max_val:
            raise ValueError('min_order_value must be less than max_order_value')
        
        return values
    
    @root_validator
    def validate_risk_parameters(cls, values):
        """Validate risk management parameters."""
        portfolio_risk = values.get('max_portfolio_risk', 0.10)
        position_risk = values.get('risk_position_limit', 0.05)
        
        if position_risk > portfolio_risk:
            raise ValueError('Individual position risk cannot exceed portfolio risk')
        
        return values
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        use_enum_values = True


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
    return settings.environment in [Environment.LOCAL, Environment.DEVELOPMENT]


def is_docker() -> bool:
    """
    Check if running in Docker environment.
    
    Returns:
        bool: True if Docker environment
    """
    return settings.environment == Environment.DOCKER


def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration dictionary.
    
    Returns:
        Dict[str, Any]: Database configuration
    """
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "echo": settings.database_echo,
        "pool_timeout": settings.database_pool_timeout,
        "pool_recycle": settings.database_pool_recycle,
    }


def get_redis_config() -> Dict[str, Any]:
    """
    Get Redis configuration dictionary.
    
    Returns:
        Dict[str, Any]: Redis configuration
    """
    return {
        "url": settings.redis_url,
        "max_connections": settings.redis_max_connections,
        "socket_keepalive": settings.redis_socket_keepalive,
        "socket_keepalive_options": settings.redis_socket_keepalive_options,
    }


def get_celery_config() -> Dict[str, Any]:
    """
    Get Celery configuration dictionary.
    
    Returns:
        Dict[str, Any]: Celery configuration
    """
    return {
        "broker_url": settings.celery_broker_url,
        "result_backend": settings.celery_result_backend,
        "task_serializer": settings.celery_task_serializer,
        "result_serializer": settings.celery_result_serializer,
        "accept_content": settings.celery_accept_content,
        "timezone": settings.celery_timezone,
        "worker_prefetch_multiplier": settings.celery_worker_prefetch_multiplier,
        "task_acks_late": settings.celery_task_acks_late,
        "worker_max_tasks_per_child": settings.celery_worker_max_tasks_per_child,
    }
