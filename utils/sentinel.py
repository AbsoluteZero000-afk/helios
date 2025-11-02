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
        
        # Critical modules to monitor
        self.critical_modules = [
            "pandas", "numpy", "sqlalchemy", "redis", "celery",
            "aiohttp", "loguru", "pydantic", "psutil", "psycopg2"
        ]
        
        # Optional modules (warning if missing, not critical)
        self.optional_modules = [
            "talib", "ta", "alpaca_trade_api", "yfinance", "ccxt",
            "slack_sdk", "fastapi", "uvicorn"
        ]
        
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
        }
        
        logger.info("System Sentinel v3 initialized")
    
    async def start_monitoring(self) -> None:
        """
        Start continuous system monitoring and integrity checks.
        """
        if self.is_running:
            logger.warning("Sentinel already running")
            return
        
        self.is_running = True
        logger.info("Starting System Sentinel monitoring")
        
        # Perform initial integrity audit
        await self.full_integrity_audit()
        
        # Send startup notification
        await slack_notifier.system_alert(
            "System Sentinel v3 activated - comprehensive monitoring commenced",
            component="sentinel",
            severity="success",
            version="3.0.0",
            auto_repair=self.settings.sentinel_auto_repair
        )
        
        # Start monitoring loop
        if self.settings.sentinel_enabled:
            asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """
        Stop system monitoring.
        """
        self.is_running = False
        logger.info("System Sentinel monitoring stopped")
        
        await slack_notifier.system_alert(
            "System Sentinel deactivated",
            component="sentinel",
            severity="info",
            uptime=str(datetime.now(timezone.utc) - self.start_time)
        )
    
    async def _monitoring_loop(self) -> None:
        """
        Main monitoring loop with periodic integrity checks.
        """
        while self.is_running:
            try:
                # Perform health check
                health_report = await health_checker.check_all()
                
                # Check for critical issues
                critical_issues = [
                    check for check in health_report.checks
                    if check.status == HealthStatus.CRITICAL
                ]
                
                if critical_issues:
                    await self._handle_critical_issues(critical_issues)
                
                # Periodic full integrity audit
                if (not self.last_integrity_check or 
                    datetime.now(timezone.utc) - self.last_integrity_check > timedelta(hours=1)):
                    await self.full_integrity_audit()
                
                # Wait for next check interval
                await asyncio.sleep(self.settings.sentinel_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await slack_notifier.system_alert(
                    f"Sentinel monitoring error: {e}",
                    component="sentinel",
                    severity="error"
                )
                await asyncio.sleep(30)  # Shorter interval on error
    
    async def full_integrity_audit(self) -> List[IntegrityCheckResult]:
        """
        Perform comprehensive system integrity audit.
        
        Returns:
            List[IntegrityCheckResult]: List of all integrity check results
        """
        logger.info("Starting full system integrity audit")
        start_time = time.time()
        
        results = []
        
        # 1. Syntax and compilation check
        results.extend(await self._check_syntax_integrity())
        
        # 2. Module import checks
        results.extend(await self._check_module_imports())
        
        # 3. Critical component validation
        results.extend(await self._check_critical_components())
        
        # 4. Database connectivity and schema validation
        results.extend(await self._check_database_integrity())
        
        # 5. Redis connectivity and performance
        results.extend(await self._check_redis_integrity())
        
        # 6. Celery worker validation
        results.extend(await self._check_celery_integrity())
        
        # 7. File system integrity
        results.extend(await self._check_filesystem_integrity())
        
        # Process results and attempt repairs if enabled
        if self.settings.sentinel_auto_repair:
            repair_tasks = []
            for result in results:
                if not result.passed and result.repair_recommended and result.repair_type:
                    repair_tasks.append(self._attempt_repair(
                        result.repair_type,
                        result.component,
                        result.message
                    ))
            
            if repair_tasks:
                repair_results = await asyncio.gather(*repair_tasks, return_exceptions=True)
                logger.info(f"Completed {len(repair_results)} repair attempts")
        
        self.last_integrity_check = datetime.now(timezone.utc)
        audit_duration = time.time() - start_time
        
        # Generate summary
        failed_checks = [r for r in results if not r.passed]
        critical_failures = [r for r in failed_checks if r.severity == SeverityLevel.CRITICAL]
        
        logger.info(
            f"Integrity audit completed in {audit_duration:.2f}s: "
            f"{len(results)} checks, {len(failed_checks)} failures, "
            f"{len(critical_failures)} critical"
        )
        
        # Send summary to Slack if there are significant issues
        if critical_failures:
            await slack_notifier.system_alert(
                f"Integrity audit found {len(critical_failures)} critical issues",
                component="sentinel",
                severity="error",
                audit_duration=f"{audit_duration:.2f}s",
                total_checks=len(results),
                failed_checks=len(failed_checks)
            )
        
        return results
    
    async def _check_syntax_integrity(self) -> List[IntegrityCheckResult]:
        """
        Check Python syntax integrity using compileall.
        
        Returns:
            List[IntegrityCheckResult]: Syntax check results
        """
        results = []
        
        try:
            # Run compileall on current directory
            result = subprocess.run(
                [sys.executable, "-m", "compileall", ".", "-q"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                results.append(IntegrityCheckResult(
                    component="syntax_check",
                    passed=True,
                    message="All Python files compile successfully",
                    severity=SeverityLevel.LOW
                ))
            else:
                results.append(IntegrityCheckResult(
                    component="syntax_check",
                    passed=False,
                    message=f"Syntax errors detected: {result.stderr}",
                    severity=SeverityLevel.HIGH,
                    repair_recommended=True,
                    repair_type=RepairType.SYNTAX_ERROR,
                    details={"stderr": result.stderr, "stdout": result.stdout}
                ))
                
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="syntax_check",
                passed=False,
                message=f"Syntax check failed: {e}",
                severity=SeverityLevel.CRITICAL,
                details={"error": str(e)}
            ))
        
        return results
    
    async def _check_module_imports(self) -> List[IntegrityCheckResult]:
        """
        Check critical and optional module imports.
        
        Returns:
            List[IntegrityCheckResult]: Module import check results
        """
        results = []
        
        # Check critical modules
        for module_name in self.critical_modules:
            try:
                importlib.import_module(module_name)
                results.append(IntegrityCheckResult(
                    component=f"module_{module_name}",
                    passed=True,
                    message=f"Critical module '{module_name}' imports successfully",
                    severity=SeverityLevel.LOW
                ))
            except ImportError as e:
                results.append(IntegrityCheckResult(
                    component=f"module_{module_name}",
                    passed=False,
                    message=f"Critical module '{module_name}' import failed: {e}",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.PACKAGE_INSTALL,
                    details={"module": module_name, "error": str(e)}
                ))
        
        # Check optional modules
        for module_name in self.optional_modules:
            try:
                importlib.import_module(module_name)
                results.append(IntegrityCheckResult(
                    component=f"optional_module_{module_name}",
                    passed=True,
                    message=f"Optional module '{module_name}' imports successfully",
                    severity=SeverityLevel.LOW
                ))
            except ImportError as e:
                results.append(IntegrityCheckResult(
                    component=f"optional_module_{module_name}",
                    passed=False,
                    message=f"Optional module '{module_name}' import failed: {e}",
                    severity=SeverityLevel.MEDIUM,
                    repair_recommended=True,
                    repair_type=RepairType.PACKAGE_INSTALL,
                    details={"module": module_name, "error": str(e)}
                ))
        
        return results
    
    async def _check_critical_components(self) -> List[IntegrityCheckResult]:
        """
        Check that critical classes and functions can be imported.
        
        Returns:
            List[IntegrityCheckResult]: Component validation results
        """
        results = []
        
        for component_path, description in self.critical_components.items():
            try:
                module_path, attr_name = component_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                component = getattr(module, attr_name)
                
                results.append(IntegrityCheckResult(
                    component=component_path,
                    passed=True,
                    message=f"Critical component '{component_path}' accessible",
                    severity=SeverityLevel.LOW,
                    details={"description": description, "type": str(type(component))}
                ))
                
            except (ImportError, AttributeError) as e:
                results.append(IntegrityCheckResult(
                    component=component_path,
                    passed=False,
                    message=f"Critical component '{component_path}' not accessible: {e}",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.FILE_CORRUPTION,
                    details={"description": description, "error": str(e)}
                ))
        
        return results
    
    async def _check_database_integrity(self) -> List[IntegrityCheckResult]:
        """
        Check database connectivity and schema integrity.
        
        Returns:
            List[IntegrityCheckResult]: Database integrity results
        """
        results = []
        
        try:
            # Import database manager
            from utils.db import db_manager
            
            # Test basic connectivity
            if db_manager.test_connection():
                results.append(IntegrityCheckResult(
                    component="database_connection",
                    passed=True,
                    message="Database connection successful",
                    severity=SeverityLevel.LOW
                ))
                
                # Check table existence
                try:
                    table_info = db_manager.get_table_info()
                    expected_tables = {
                        "trades", "orders", "positions", "performance_snapshots",
                        "sentinel_repairs", "risk_metrics"
                    }
                    
                    existing_tables = set(table_info.keys())
                    missing_tables = expected_tables - existing_tables
                    
                    if missing_tables:
                        results.append(IntegrityCheckResult(
                            component="database_schema",
                            passed=False,
                            message=f"Missing database tables: {missing_tables}",
                            severity=SeverityLevel.HIGH,
                            repair_recommended=True,
                            repair_type=RepairType.DATABASE_MIGRATION,
                            details={"missing_tables": list(missing_tables), "existing_tables": list(existing_tables)}
                        ))
                    else:
                        results.append(IntegrityCheckResult(
                            component="database_schema",
                            passed=True,
                            message="All required database tables exist",
                            severity=SeverityLevel.LOW,
                            details={"tables": table_info}
                        ))
                        
                except Exception as e:
                    results.append(IntegrityCheckResult(
                        component="database_schema",
                        passed=False,
                        message=f"Database schema check failed: {e}",
                        severity=SeverityLevel.HIGH,
                        details={"error": str(e)}
                    ))
            else:
                results.append(IntegrityCheckResult(
                    component="database_connection",
                    passed=False,
                    message="Database connection failed",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.DATABASE_MIGRATION
                ))
                
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="database_check",
                passed=False,
                message=f"Database integrity check failed: {e}",
                severity=SeverityLevel.CRITICAL,
                details={"error": str(e)}
            ))
        
        return results
    
    async def _check_redis_integrity(self) -> List[IntegrityCheckResult]:
        """
        Check Redis connectivity and performance.
        
        Returns:
            List[IntegrityCheckResult]: Redis integrity results
        """
        results = []
        
        try:
            from utils.redis_queue import redis_manager
            
            if redis_manager.test_connection():
                results.append(IntegrityCheckResult(
                    component="redis_connection",
                    passed=True,
                    message="Redis connection successful",
                    severity=SeverityLevel.LOW
                ))
                
                # Test basic operations
                test_key = "sentinel_health_check"
                test_value = {"timestamp": datetime.now(timezone.utc).isoformat()}
                
                if (redis_manager.set(test_key, test_value, ttl=60) and
                    redis_manager.get(test_key) and
                    redis_manager.delete(test_key)):
                    
                    results.append(IntegrityCheckResult(
                        component="redis_operations",
                        passed=True,
                        message="Redis operations (set/get/delete) working",
                        severity=SeverityLevel.LOW
                    ))
                else:
                    results.append(IntegrityCheckResult(
                        component="redis_operations",
                        passed=False,
                        message="Redis operations failing",
                        severity=SeverityLevel.HIGH,
                        repair_recommended=True,
                        repair_type=RepairType.REDIS_CONNECTION
                    ))
            else:
                results.append(IntegrityCheckResult(
                    component="redis_connection",
                    passed=False,
                    message="Redis connection failed",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.REDIS_CONNECTION
                ))
                
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="redis_check",
                passed=False,
                message=f"Redis integrity check failed: {e}",
                severity=SeverityLevel.CRITICAL,
                details={"error": str(e)}
            ))
        
        return results
    
    async def _check_celery_integrity(self) -> List[IntegrityCheckResult]:
        """
        Check Celery worker connectivity and task queue health.
        
        Returns:
            List[IntegrityCheckResult]: Celery integrity results
        """
        results = []
        
        try:
            # For now, we'll check if the Celery module can be imported
            # and if the broker (Redis) is accessible
            import celery
            
            results.append(IntegrityCheckResult(
                component="celery_import",
                passed=True,
                message="Celery module imports successfully",
                severity=SeverityLevel.LOW
            ))
            
            # Check broker connectivity (Redis in our case)
            from utils.redis_queue import redis_manager
            
            if redis_manager.test_connection():
                results.append(IntegrityCheckResult(
                    component="celery_broker",
                    passed=True,
                    message="Celery broker (Redis) accessible",
                    severity=SeverityLevel.LOW
                ))
            else:
                results.append(IntegrityCheckResult(
                    component="celery_broker",
                    passed=False,
                    message="Celery broker (Redis) not accessible",
                    severity=SeverityLevel.HIGH,
                    repair_recommended=True,
                    repair_type=RepairType.CELERY_WORKER
                ))
                
        except ImportError as e:
            results.append(IntegrityCheckResult(
                component="celery_import",
                passed=False,
                message=f"Celery import failed: {e}",
                severity=SeverityLevel.CRITICAL,
                repair_recommended=True,
                repair_type=RepairType.PACKAGE_INSTALL,
                details={"error": str(e)}
            ))
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="celery_check",
                passed=False,
                message=f"Celery integrity check failed: {e}",
                severity=SeverityLevel.CRITICAL,
                details={"error": str(e)}
            ))
        
        return results
    
    async def _check_filesystem_integrity(self) -> List[IntegrityCheckResult]:
        """
        Check critical files and directories exist and are accessible.
        
        Returns:
            List[IntegrityCheckResult]: Filesystem integrity results
        """
        results = []
        
        # Critical directories
        critical_dirs = [
            "config", "core", "strategies", "utils", "data", "execution", "risk"
        ]
        
        for dir_name in critical_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                results.append(IntegrityCheckResult(
                    component=f"directory_{dir_name}",
                    passed=True,
                    message=f"Critical directory '{dir_name}' exists",
                    severity=SeverityLevel.LOW
                ))
            else:
                results.append(IntegrityCheckResult(
                    component=f"directory_{dir_name}",
                    passed=False,
                    message=f"Critical directory '{dir_name}' missing",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.FILE_CORRUPTION
                ))
        
        # Critical files
        critical_files = [
            "main.py", "requirements.txt", "config/settings.py",
            "core/engine.py", "utils/logger.py"
        ]
        
        for file_name in critical_files:
            file_path = Path(file_name)
            if file_path.exists() and file_path.is_file():
                results.append(IntegrityCheckResult(
                    component=f"file_{file_name}",
                    passed=True,
                    message=f"Critical file '{file_name}' exists",
                    severity=SeverityLevel.LOW
                ))
            else:
                results.append(IntegrityCheckResult(
                    component=f"file_{file_name}",
                    passed=False,
                    message=f"Critical file '{file_name}' missing",
                    severity=SeverityLevel.CRITICAL,
                    repair_recommended=True,
                    repair_type=RepairType.FILE_CORRUPTION
                ))
        
        # Check log and data directories
        log_dir = Path(self.settings.log_file_path).parent
        data_dir = Path(self.settings.data_storage_path)
        
        for dir_path, name in [(log_dir, "logs"), (data_dir, "data")]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                results.append(IntegrityCheckResult(
                    component=f"directory_{name}",
                    passed=True,
                    message=f"Directory '{name}' accessible",
                    severity=SeverityLevel.LOW
                ))
            except Exception as e:
                results.append(IntegrityCheckResult(
                    component=f"directory_{name}",
                    passed=False,
                    message=f"Directory '{name}' not accessible: {e}",
                    severity=SeverityLevel.HIGH,
                    details={"error": str(e)}
                ))
        
        return results
    
    async def _handle_critical_issues(self, critical_issues: List[Any]) -> None:
        """
        Handle critical issues found during health checks.
        
        Args:
            critical_issues: List of critical health check results
        """
        logger.error(f"Critical issues detected: {len(critical_issues)}")
        
        for issue in critical_issues:
            await slack_notifier.system_alert(
                f"CRITICAL: {issue.component} - {issue.message}",
                component=issue.component,
                severity="error"
            )
        
        # If auto-repair is enabled, attempt to fix issues
        if self.settings.sentinel_auto_repair:
            logger.info("Attempting to resolve critical issues automatically")
            # This would be expanded to map health check issues to repair types
    
    async def _attempt_repair(
        self,
        repair_type: RepairType,
        component: str,
        issue_description: str
    ) -> RepairAttempt:
        """
        Attempt automated repair for a detected issue.
        
        Args:
            repair_type: Type of repair to attempt
            component: Component being repaired
            issue_description: Description of the issue
            
        Returns:
            RepairAttempt: Repair attempt result
        """
        if len(self.repair_history) >= self.settings.sentinel_max_repair_attempts:
            logger.warning("Maximum repair attempts reached, skipping repair")
            return RepairAttempt(
                repair_type=repair_type,
                component=component,
                issue_description=issue_description,
                repair_action="Skipped - max attempts reached",
                success=False,
                error_message="Maximum repair attempts exceeded"
            )
        
        start_time = time.time()
        
        try:
            repair_func = self.repair_registry.get(repair_type)
            if not repair_func:
                raise ValueError(f"Unknown repair type: {repair_type}")
            
            logger.info(f"Attempting repair: {repair_type} for {component}")
            
            repair_action, success, error_message = await repair_func(component, issue_description)
            
            execution_time = time.time() - start_time
            
            repair_attempt = RepairAttempt(
                repair_type=repair_type,
                component=component,
                issue_description=issue_description,
                repair_action=repair_action,
                success=success,
                error_message=error_message,
                execution_time_seconds=execution_time
            )
            
            self.repair_history.append(repair_attempt)
            
            # Log repair attempt to file
            await self._log_repair_attempt(repair_attempt)
            
            # Store in database if available
            await self._store_repair_attempt(repair_attempt)
            
            # Send Slack notification
            severity = "success" if success else "warning"
            await slack_notifier.system_alert(
                f"Auto-repair {'successful' if success else 'failed'}: {repair_type}",
                component="sentinel",
                severity=severity,
                target=component,
                repair_action=repair_action,
                execution_time=f"{execution_time:.2f}s"
            )
            
            return repair_attempt
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            repair_attempt = RepairAttempt(
                repair_type=repair_type,
                component=component,
                issue_description=issue_description,
                repair_action="Repair attempt failed",
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
            
            self.repair_history.append(repair_attempt)
            
            logger.error(f"Repair attempt failed: {e}")
            
            await slack_notifier.system_alert(
                f"Auto-repair error: {repair_type} - {e}",
                component="sentinel",
                severity="error",
                target=component
            )
            
            return repair_attempt
    
    # Repair implementation methods
    async def _repair_package_install(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to install missing package."""
        try:
            # Extract package name from component
            if "module_" in component:
                package_name = component.replace("module_", "").replace("optional_module_", "")
            else:
                package_name = component
            
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package_name],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return f"Successfully installed {package_name}", True, None
            else:
                return f"Failed to install {package_name}", False, result.stderr
                
        except Exception as e:
            return "Package install attempt failed", False, str(e)
    
    async def _repair_module_import(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to fix module import issues."""
        try:
            # Try to reload the module
            module_name = component.replace("module_", "")
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
                return f"Reloaded module {module_name}", True, None
            else:
                importlib.import_module(module_name)
                return f"Successfully imported {module_name}", True, None
                
        except Exception as e:
            return "Module import repair failed", False, str(e)
    
    async def _repair_syntax_error(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to fix syntax errors (placeholder implementation)."""
        # This is a complex repair that would require more sophisticated logic
        return "Syntax error repair not implemented", False, "Complex repair required"
    
    async def _repair_database_migration(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to create missing database tables."""
        try:
            from utils.db import db_manager
            db_manager.create_tables()
            return "Database tables created/updated", True, None
        except Exception as e:
            return "Database migration failed", False, str(e)
    
    async def _repair_redis_connection(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to fix Redis connection issues."""
        try:
            from utils.redis_queue import redis_manager
            # Try to reinitialize Redis connection
            redis_manager.cleanup()
            redis_manager.initialize()
            
            if redis_manager.test_connection():
                return "Redis connection restored", True, None
            else:
                return "Redis connection still failing", False, "Connection test failed"
                
        except Exception as e:
            return "Redis repair failed", False, str(e)
    
    async def _repair_celery_worker(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to fix Celery worker issues."""
        # This would typically involve restarting workers or checking broker connectivity
        return "Celery worker repair not implemented", False, "Requires manual intervention"
    
    async def _repair_file_corruption(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to fix file corruption issues."""
        # This would involve restoring files from backup or recreating them
        return "File corruption repair not implemented", False, "Requires manual intervention"
    
    async def _repair_memory_cleanup(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt memory cleanup."""
        try:
            import gc
            gc.collect()
            return "Memory cleanup performed", True, None
        except Exception as e:
            return "Memory cleanup failed", False, str(e)
    
    async def _repair_cache_clear(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Clear Python cache files."""
        try:
            import shutil
            cache_cleared = 0
            for root, dirs, files in os.walk('.'):
                for dirname in dirs[:]:
                    if dirname == '__pycache__':
                        cache_path = os.path.join(root, dirname)
                        shutil.rmtree(cache_path)
                        cache_cleared += 1
                        dirs.remove(dirname)
            
            return f"Cleared {cache_cleared} cache directories", True, None
        except Exception as e:
            return "Cache clear failed", False, str(e)
    
    async def _repair_service_restart(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt service restart (placeholder)."""
        return "Service restart not implemented", False, "Requires container orchestration"
    
    async def _log_repair_attempt(self, repair_attempt: RepairAttempt) -> None:
        """
        Log repair attempt to JSON file.
        
        Args:
            repair_attempt: Repair attempt to log
        """
        try:
            log_file = Path(self.settings.sentinel_repair_log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing log
            repair_log = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        repair_log = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    repair_log = []
            
            # Append new repair attempt
            repair_log.append(repair_attempt.to_dict())
            
            # Keep only last 1000 entries
            if len(repair_log) > 1000:
                repair_log = repair_log[-1000:]
            
            # Write back to file
            with open(log_file, 'w') as f:
                json.dump(repair_log, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to log repair attempt: {e}")
    
    async def _store_repair_attempt(self, repair_attempt: RepairAttempt) -> None:
        """
        Store repair attempt in database.
        
        Args:
            repair_attempt: Repair attempt to store
        """
        try:
            from utils.db import db_manager, SentinelRepair
            
            with db_manager.get_session() as session:
                repair_record = SentinelRepair(
                    repair_type=repair_attempt.repair_type.value,
                    component=repair_attempt.component,
                    issue_description=repair_attempt.issue_description,
                    repair_action=repair_attempt.repair_action,
                    success=repair_attempt.success,
                    error_message=repair_attempt.error_message,
                    execution_time_seconds=repair_attempt.execution_time_seconds,
                    metadata=repair_attempt.metadata
                )
                session.add(repair_record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to store repair attempt in database: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Sentinel status.
        
        Returns:
            Dict[str, Any]: Sentinel status information
        """
        uptime = datetime.now(timezone.utc) - self.start_time
        
        return {
            "running": self.is_running,
            "version": "3.0.0",
            "uptime": str(uptime),
            "uptime_seconds": uptime.total_seconds(),
            "auto_repair_enabled": self.settings.sentinel_auto_repair,
            "max_repair_attempts": self.settings.sentinel_max_repair_attempts,
            "check_interval": self.settings.sentinel_check_interval,
            "last_integrity_check": self.last_integrity_check.isoformat() if self.last_integrity_check else None,
            "repair_history_count": len(self.repair_history),
            "successful_repairs": sum(1 for r in self.repair_history if r.success),
            "failed_repairs": sum(1 for r in self.repair_history if not r.success),
            "critical_modules": self.critical_modules,
            "optional_modules": self.optional_modules,
        }
    
    def get_repair_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent repair history.
        
        Args:
            limit: Maximum number of repair attempts to return
            
        Returns:
            List[Dict[str, Any]]: Recent repair attempts
        """
        recent_repairs = self.repair_history[-limit:] if limit else self.repair_history
        return [repair.to_dict() for repair in recent_repairs]


# Global sentinel instance
system_sentinel = SystemSentinel()


# CLI interface
async def main():
    """Main CLI interface for Sentinel."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Helios System Sentinel v3")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run integrity audit")
    parser.add_argument("--audit", action="store_true", help="Perform full integrity audit")
    parser.add_argument("--history", type=int, default=10, help="Show repair history (default: 10)")
    
    args = parser.parse_args()
    
    if args.status:
        status = system_sentinel.get_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.dry_run:
        print("Performing dry-run integrity audit...")
        # Temporarily disable auto-repair
        original_auto_repair = system_sentinel.settings.sentinel_auto_repair
        system_sentinel.settings.sentinel_auto_repair = False
        
        results = await system_sentinel.full_integrity_audit()
        
        # Restore original setting
        system_sentinel.settings.sentinel_auto_repair = original_auto_repair
        
        print(f"\nAudit completed: {len(results)} checks performed")
        failed_checks = [r for r in results if not r.passed]
        
        if failed_checks:
            print(f"\nIssues found ({len(failed_checks)}):")
            for check in failed_checks:
                print(f"  - {check.component}: {check.message} ({check.severity.value})")
        else:
            print("\nNo issues found - system integrity OK")
    
    elif args.audit:
        print("Performing full integrity audit with repairs...")
        results = await system_sentinel.full_integrity_audit()
        
        print(f"\nAudit completed: {len(results)} checks performed")
        failed_checks = [r for r in results if not r.passed]
        
        if failed_checks:
            print(f"\nIssues found ({len(failed_checks)}):")
            for check in failed_checks:
                print(f"  - {check.component}: {check.message} ({check.severity.value})")
        
        repairs = system_sentinel.get_repair_history(limit=10)
        if repairs:
            print(f"\nRecent repairs ({len(repairs)}):")
            for repair in repairs[-5:]:  # Show last 5
                status = "✓" if repair['success'] else "✗"
                print(f"  {status} {repair['repair_type']}: {repair['component']}")
    
    else:
        history = system_sentinel.get_repair_history(limit=args.history)
        if history:
            print(f"Repair History (last {len(history)}):")
            for repair in history:
                status = "✓" if repair['success'] else "✗"
                timestamp = repair['timestamp'][:19]  # Remove microseconds
                print(f"  {status} {timestamp} {repair['repair_type']}: {repair['component']}")
                if not repair['success'] and repair.get('error_message'):
                    print(f"    Error: {repair['error_message']}")
        else:
            print("No repair history available")


def cli():
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
