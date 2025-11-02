"""
Helios Configuration Management v3 (Pydantic v2 compliant)

Enhanced configuration system with Pydantic v2 + pydantic-settings for type validation,
environment variable management, and comprehensive trading parameters.
"""

from __future__ import annotations

from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


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
    Pydantic v2 compliant: imports BaseSettings from pydantic-settings,
    uses model_config and new validator decorators.
    """

    # Application Metadata
    app_name: str = "Helios"
    version: str = "3.0.0"
    environment: Environment = Field(default=Environment.LOCAL, alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Database Configuration
    database_url: str = Field(
        default="postgresql+psycopg2://helios:helios@localhost:5432/helios",
        alias="DATABASE_URL",
    )
    database_pool_size: int = Field(default=10, ge=1, le=50)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    database_echo: bool = False
    database_pool_timeout: int = Field(default=30, ge=5, le=300)
    database_pool_recycle: int = Field(default=3600, ge=300, le=86400)

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_max_connections: int = Field(default=20, ge=1, le=100)
    redis_socket_keepalive: bool = True
    redis_socket_keepalive_options: Dict[str, int] = Field(
        default_factory=lambda: {"TCP_KEEPIDLE": 1, "TCP_KEEPINTVL": 3, "TCP_KEEPCNT": 5}
    )

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/1", alias="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND")
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    celery_accept_content: List[str] = Field(default_factory=lambda: ["json"])
    celery_timezone: str = "UTC"
    celery_worker_prefetch_multiplier: int = Field(default=1, ge=1, le=10)
    celery_task_acks_late: bool = True
    celery_worker_max_tasks_per_child: int = Field(default=1000, ge=100, le=10000)

    # Trading APIs
    alpaca_api_key: Optional[str] = Field(default=None, alias="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, alias="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL")
    alpaca_ws_url: str = Field(default="wss://stream.data.alpaca.markets/v2", alias="ALPACA_WS_URL")
    alpaca_data_feed: str = Field(default="iex", alias="ALPACA_DATA_FEED")

    # Financial Market API
    fmp_api_key: Optional[str] = Field(default=None, alias="FMP_API_KEY")
    fmp_base_url: str = Field(default="https://financialmodelingprep.com/api/v3", alias="FMP_BASE_URL")

    # Security
    encryption_key: Optional[str] = Field(default=None, alias="ENCRYPTION_KEY")
    jwt_secret: Optional[str] = Field(default=None, alias="JWT_SECRET")
    api_secret_key: Optional[str] = Field(default=None, alias="API_SECRET_KEY")

    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, alias="LOG_LEVEL")
    log_to_file: bool = Field(default=True, alias="LOG_TO_FILE")
    log_file_path: str = Field(default="/logs/helios.log", alias="LOG_FILE_PATH")
    log_json_format: bool = Field(default=False, alias="LOG_JSON_FORMAT")
    log_rotation_size: str = Field(default="100MB", alias="LOG_ROTATION_SIZE")
    log_retention_days: int = Field(default=30, ge=1, le=365, alias="LOG_RETENTION_DAYS")
    log_compression: str = Field(default="zip", alias="LOG_COMPRESSION")

    # Trading Configuration
    initial_capital: float = Field(default=100000.0, ge=1000.0, le=10000000.0, alias="INITIAL_CAPITAL")
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, alias="TRADING_MODE")
    max_portfolio_risk: float = Field(default=0.10, ge=0.001, le=0.5, alias="MAX_PORTFOLIO_RISK")
    max_daily_drawdown: float = Field(default=0.20, ge=0.01, le=0.5, alias="MAX_DAILY_DRAWDOWN")
    default_position_size: float = Field(default=0.01, ge=0.001, le=0.1, alias="DEFAULT_POSITION_SIZE")
    max_open_positions: int = Field(default=10, ge=1, le=100, alias="MAX_OPEN_POSITIONS")
    min_order_value: float = Field(default=100.0, ge=1.0, le=10000.0, alias="MIN_ORDER_VALUE")
    max_order_value: float = Field(default=10000.0, ge=100.0, le=1000000.0, alias="MAX_ORDER_VALUE")

    # Risk Management
    risk_check_enabled: bool = Field(default=True, alias="RISK_CHECK_ENABLED")
    risk_position_limit: float = Field(default=0.05, ge=0.001, le=0.2, alias="RISK_POSITION_LIMIT")
    risk_correlation_limit: float = Field(default=0.30, ge=0.1, le=0.9, alias="RISK_CORRELATION_LIMIT")
    risk_var_confidence: float = Field(default=0.95, ge=0.9, le=0.99, alias="RISK_VAR_CONFIDENCE")
    risk_lookback_days: int = Field(default=252, ge=20, le=1000, alias="RISK_LOOKBACK_DAYS")

    # Notifications (Slack)
    slack_webhook_url: Optional[str] = Field(default=None, alias="SLACK_WEBHOOK_URL")
    slack_channel: str = Field(default="helios-alerts", alias="SLACK_CHANNEL")
    slack_username: str = Field(default="Helios-Bot", alias="SLACK_USERNAME")
    slack_icon_emoji: str = Field(default=":robot_face:", alias="SLACK_ICON_EMOJI")
    slack_mention_channel: bool = Field(default=False, alias="SLACK_MENTION_CHANNEL")
    slack_mention_users: List[str] = Field(default_factory=list, alias="SLACK_MENTION_USERS")

    # System Sentinel Configuration
    sentinel_enabled: bool = Field(default=True, alias="SENTINEL_ENABLED")
    sentinel_auto_repair: bool = Field(default=True, alias="SENTINEL_AUTO_REPAIR")
    sentinel_max_repair_attempts: int = Field(default=3, ge=1, le=10, alias="SENTINEL_MAX_REPAIR_ATTEMPTS")
    sentinel_check_interval: int = Field(default=60, ge=10, le=3600, alias="SENTINEL_CHECK_INTERVAL")
    sentinel_health_endpoint: bool = Field(default=True, alias="SENTINEL_HEALTH_ENDPOINT")
    sentinel_repair_log_file: str = Field(default="/logs/sentinel_repairs.json", alias="SENTINEL_REPAIR_LOG_FILE")
    sentinel_db_validation: bool = Field(default=True, alias="SENTINEL_DB_VALIDATION")
    sentinel_redis_validation: bool = Field(default=True, alias="SENTINEL_REDIS_VALIDATION")
    sentinel_celery_validation: bool = Field(default=True, alias="SENTINEL_CELERY_VALIDATION")

    # Health Check Configuration
    health_check_enabled: bool = Field(default=True, alias="HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(default=30, ge=5, le=300, alias="HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=10, ge=1, le=60, alias="HEALTH_CHECK_TIMEOUT")
    health_check_retries: int = Field(default=3, ge=1, le=10, alias="HEALTH_CHECK_RETRIES")

    # Performance Monitoring
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    metrics_collection_interval: int = Field(default=300, ge=60, le=3600, alias="METRICS_COLLECTION_INTERVAL")
    metrics_retention_days: int = Field(default=90, ge=7, le=365, alias="METRICS_RETENTION_DAYS")
    metrics_detailed_logging: bool = Field(default=False, alias="METRICS_DETAILED_LOGGING")

    # Data Configuration
    data_storage_path: str = Field(default="/data", alias="DATA_STORAGE_PATH")
    data_cache_ttl: int = Field(default=3600, ge=60, le=86400, alias="DATA_CACHE_TTL")
    data_batch_size: int = Field(default=1000, ge=100, le=10000, alias="DATA_BATCH_SIZE")
    data_async_processing: bool = Field(default=True, alias="DATA_ASYNC_PROCESSING")

    # Strategy Configuration
    strategy_config_file: str = Field(default="config/strategy.yaml", alias="STRATEGY_CONFIG_FILE")
    strategy_reload_interval: int = Field(default=300, ge=60, le=3600, alias="STRATEGY_RELOAD_INTERVAL")
    strategy_validation_strict: bool = Field(default=True, alias="STRATEGY_VALIDATION_STRICT")

    # Container Configuration
    container_timezone: str = Field(default="UTC", alias="CONTAINER_TIMEZONE")
    container_user_id: int = Field(default=1000, ge=1, le=65535, alias="CONTAINER_USER_ID")
    container_group_id: int = Field(default=1000, ge=1, le=65535, alias="CONTAINER_GROUP_ID")
    container_memory_limit: str = Field(default="2G", alias="CONTAINER_MEMORY_LIMIT")
    container_cpu_limit: str = Field(default="1000m", alias="CONTAINER_CPU_LIMIT")

    # Pydantic v2 settings
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
        "validate_assignment": True,
        "case_sensitive": False,
        "use_enum_values": True,
    }

    # Validators (v2)
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("postgresql://", "postgresql+psycopg2://", "sqlite://")):
            raise ValueError("Database URL must be PostgreSQL or SQLite")
        return v

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        if not v.startswith("redis://"):
            raise ValueError("Redis URL must start with redis://")
        return v

    @field_validator("log_file_path")
    @classmethod
    def validate_log_path(cls, v: str) -> str:
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("data_storage_path")
    @classmethod
    def validate_data_path(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_order_values(self) -> "Settings":
        if self.min_order_value >= self.max_order_value:
            raise ValueError("min_order_value must be less than max_order_value")
        return self

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> "Settings":
        if self.risk_position_limit > self.max_portfolio_risk:
            raise ValueError("Individual position risk cannot exceed portfolio risk")
        return self


# Lazy singleton
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def is_production() -> bool:
    return get_settings().environment == Environment.PRODUCTION


def is_development() -> bool:
    return get_settings().environment in {Environment.LOCAL, Environment.DEVELOPMENT}


def is_docker() -> bool:
    return get_settings().environment == Environment.DOCKER


def get_database_config() -> Dict[str, Any]:
    s = get_settings()
    return {
        "url": s.database_url,
        "pool_size": s.database_pool_size,
        "max_overflow": s.database_max_overflow,
        "echo": s.database_echo,
        "pool_timeout": s.database_pool_timeout,
        "pool_recycle": s.database_pool_recycle,
    }


def get_redis_config() -> Dict[str, Any]:
    s = get_settings()
    return {
        "url": s.redis_url,
        "max_connections": s.redis_max_connections,
        "socket_keepalive": s.redis_socket_keepalive,
        "socket_keepalive_options": s.redis_socket_keepalive_options,
    }


def get_celery_config() -> Dict[str, Any]:
    s = get_settings()
    return {
        "broker_url": s.celery_broker_url,
        "result_backend": s.celery_result_backend,
        "task_serializer": s.celery_task_serializer,
        "result_serializer": s.celery_result_serializer,
        "accept_content": s.celery_accept_content,
        "timezone": s.celery_timezone,
        "worker_prefetch_multiplier": s.celery_worker_prefetch_multiplier,
        "task_acks_late": s.celery_task_acks_late,
        "worker_max_tasks_per_child": s.celery_worker_max_tasks_per_child,
    }
