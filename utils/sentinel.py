"""
Helios System Sentinel v3

Advanced system monitoring and self-healing capabilities with comprehensive
auto-repair, database validation, Redis monitoring, and Celery health checks.

The Sentinel performs integrity audits, detects issues, and attempts automated
remediation with detailed logging and Slack notifications.
"""

import sys
import os
import json
import asyncio
import subprocess
import importlib
import compileall
import traceback
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager

# Bootstrap sys.path when executed directly to allow 'config' package imports
# Ensures running `python utils/sentinel.py --status` works from repo root
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import psutil
from sqlalchemy import text

from config.settings import get_settings
from utils.logger import sentinel_logger as logger
from utils.slack import slack_notifier, MessageType
from utils.health import health_checker, HealthStatus


class RepairType(str, Enum):
    """Types of automated repairs."""
    MODULE_IMPORT = "module_import"
    PACKAGE_INSTALL = "package_install"
    SYNTAX_ERROR = "syntax_error"
    DATABASE_MIGRATION = "database_migration"
    REDIS_CONNECTION = "redis_connection"
    CELERY_WORKER = "celery_worker"
    FILE_CORRUPTION = "file_corruption"
    MEMORY_CLEANUP = "memory_cleanup"
    CACHE_CLEAR = "cache_clear"
    SERVICE_RESTART = "service_restart"
    DEPENDENCY_CONFLICT = "dependency_conflict"


class SeverityLevel(str, Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RepairAttempt:
    """Record of a repair attempt."""
    repair_type: RepairType
    component: str
    issue_description: str
    repair_action: str
    success: bool
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class IntegrityCheckResult:
    """Result of an integrity check."""
    component: str
    passed: bool
    message: str
    severity: SeverityLevel
    details: Dict[str, Any] = field(default_factory=dict)
    repair_recommended: bool = False
    repair_type: Optional[RepairType] = None


class SystemSentinel:
    """
    Advanced system monitoring and self-healing sentinel.
    
    Performs comprehensive integrity audits, monitors system health,
    detects issues, and attempts automated repairs with detailed logging.
    """
    
    def __init__(self):
        """Initialize System Sentinel."""
        self.settings = get_settings()
        self.start_time = datetime.now(timezone.utc)
        self.is_running = False
        self.repair_history: List[RepairAttempt] = []
        self.last_integrity_check: Optional[datetime] = None
        
        # Critical modules to monitor (Python 3.11 compatible)
        self.critical_modules = [
            "pandas", "numpy", "sqlalchemy", "redis", "celery",
            "aiohttp", "loguru", "pydantic", "psutil", "psycopg2"
        ]
        
        # Optional modules (warning if missing, not critical)
        # Note: Removed pandas-ta due to Python version compatibility
        self.optional_modules = [
            "talib",  # Preferred for technical analysis
            "ta",     # Alternative technical analysis
            "alpaca_trade_api", "yfinance", "ccxt",
            "slack_sdk", "fastapi", "uvicorn"
        ]
        
        # Problematic packages to watch for
        self.problematic_packages = {
            "pandas-ta": "Incompatible with Python 3.11+, use talib instead",
            "pandas_ta": "Incompatible with Python 3.11+, use talib instead"
        }
        
        # Critical files and classes to validate
        self.critical_components = {
            "core.engine.TradingEngine": "Main trading engine class",
            "execution.paper_broker.PaperBroker": "Paper trading broker",
            "utils.logger.setup_logging": "Logging setup function",
            "strategies.base.BaseStrategy": "Base strategy class",
            "utils.db.db_manager": "Database manager instance",
            "utils.redis_queue.redis_manager": "Redis manager instance",
        }
        
        # Repair function registry
        self.repair_registry: Dict[RepairType, Callable] = {
            RepairType.PACKAGE_INSTALL: self._repair_package_install,
            RepairType.MODULE_IMPORT: self._repair_module_import,
            RepairType.SYNTAX_ERROR: self._repair_syntax_error,
            RepairType.DATABASE_MIGRATION: self._repair_database_migration,
            RepairType.REDIS_CONNECTION: self._repair_redis_connection,
            RepairType.CELERY_WORKER: self._repair_celery_worker,
            RepairType.FILE_CORRUPTION: self._repair_file_corruption,
            RepairType.MEMORY_CLEANUP: self._repair_memory_cleanup,
            RepairType.CACHE_CLEAR: self._repair_cache_clear,
            RepairType.SERVICE_RESTART: self._repair_service_restart,
            RepairType.DEPENDENCY_CONFLICT: self._repair_dependency_conflict,
        }
        
        logger.info("System Sentinel v3 initialized with TA-Lib preference")

    # ... rest of file unchanged ...
