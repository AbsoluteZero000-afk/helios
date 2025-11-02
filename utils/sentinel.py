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
            "System Sentinel v3 activated - TA-Lib optimized monitoring commenced",
            component="sentinel",
            severity="success",
            version="3.0.0",
            auto_repair=self.settings.sentinel_auto_repair,
            talib_available=self._is_talib_available()
        )
        
        # Start monitoring loop
        if self.settings.sentinel_enabled:
            asyncio.create_task(self._monitoring_loop())
    
    def _is_talib_available(self) -> bool:
        """Check if TA-Lib is properly installed."""
        try:
            import talib
            # Try a simple operation
            test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
            talib.SMA(test_data, timeperiod=3)
            return True
        except Exception:
            return False
    
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
        logger.info("Starting full system integrity audit (TA-Lib optimized)")
        start_time = time.time()
        
        results = []
        
        # 1. Syntax and compilation check
        results.extend(await self._check_syntax_integrity())
        
        # 2. Module import checks (with TA-Lib priority)
        results.extend(await self._check_module_imports())
        
        # 3. TA-Lib specific validation
        results.extend(await self._check_talib_integrity())
        
        # 4. Critical component validation
        results.extend(await self._check_critical_components())
        
        # 5. Database connectivity and schema validation
        results.extend(await self._check_database_integrity())
        
        # 6. Redis connectivity and performance
        results.extend(await self._check_redis_integrity())
        
        # 7. Celery worker validation
        results.extend(await self._check_celery_integrity())
        
        # 8. File system integrity
        results.extend(await self._check_filesystem_integrity())
        
        # 9. Dependency conflict detection
        results.extend(await self._check_dependency_conflicts())
        
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
        elif failed_checks:
            await slack_notifier.system_alert(
                f"Integrity audit completed with {len(failed_checks)} minor issues",
                component="sentinel",
                severity="warning",
                audit_duration=f"{audit_duration:.2f}s",
                total_checks=len(results)
            )
        
        return results
    
    async def _check_talib_integrity(self) -> List[IntegrityCheckResult]:
        """
        Specific validation for TA-Lib installation and functionality.
        
        Returns:
            List[IntegrityCheckResult]: TA-Lib specific check results
        """
        results = []
        
        try:
            import talib
            
            # Test basic TA-Lib functionality
            test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            
            # Test common indicators
            sma_result = talib.SMA(test_data, timeperiod=5)
            rsi_result = talib.RSI(test_data, timeperiod=5)
            
            if not np.isnan(sma_result[-1]) and not np.isnan(rsi_result[-1]):
                results.append(IntegrityCheckResult(
                    component="talib_functionality",
                    passed=True,
                    message="TA-Lib functioning correctly with test calculations",
                    severity=SeverityLevel.LOW,
                    details={
                        "sma_test": float(sma_result[-1]),
                        "rsi_test": float(rsi_result[-1])
                    }
                ))
            else:
                results.append(IntegrityCheckResult(
                    component="talib_functionality",
                    passed=False,
                    message="TA-Lib calculations returning NaN values",
                    severity=SeverityLevel.HIGH,
                    repair_recommended=True,
                    repair_type=RepairType.PACKAGE_INSTALL
                ))
            
        except ImportError:
            results.append(IntegrityCheckResult(
                component="talib_import",
                passed=False,
                message="TA-Lib not installed or not accessible",
                severity=SeverityLevel.HIGH,
                repair_recommended=True,
                repair_type=RepairType.PACKAGE_INSTALL,
                details={
                    "recommendation": "Install TA-Lib: pip install TA-Lib",
                    "macos_hint": "brew install ta-lib (required on macOS)"
                }
            ))
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="talib_validation",
                passed=False,
                message=f"TA-Lib validation failed: {e}",
                severity=SeverityLevel.HIGH,
                details={"error": str(e)}
            ))
        
        return results
    
    async def _check_dependency_conflicts(self) -> List[IntegrityCheckResult]:
        """
        Check for problematic package installations.
        
        Returns:
            List[IntegrityCheckResult]: Dependency conflict check results
        """
        results = []
        
        try:
            # Check for problematic packages
            for package_name, issue_description in self.problematic_packages.items():
                try:
                    __import__(package_name.replace('-', '_'))
                    # If import succeeds, the problematic package is installed
                    results.append(IntegrityCheckResult(
                        component=f"problematic_package_{package_name}",
                        passed=False,
                        message=f"Problematic package '{package_name}' is installed: {issue_description}",
                        severity=SeverityLevel.MEDIUM,
                        repair_recommended=True,
                        repair_type=RepairType.DEPENDENCY_CONFLICT,
                        details={"package": package_name, "issue": issue_description}
                    ))
                except ImportError:
                    # Good - problematic package is not installed
                    results.append(IntegrityCheckResult(
                        component=f"problematic_package_{package_name}",
                        passed=True,
                        message=f"Problematic package '{package_name}' correctly not installed",
                        severity=SeverityLevel.LOW
                    ))
            
            # Check Python version compatibility
            python_version = sys.version_info
            if python_version.major == 3 and python_version.minor >= 11:
                results.append(IntegrityCheckResult(
                    component="python_version",
                    passed=True,
                    message=f"Python version {python_version.major}.{python_version.minor} compatible",
                    severity=SeverityLevel.LOW,
                    details={"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
                ))
            else:
                results.append(IntegrityCheckResult(
                    component="python_version",
                    passed=False,
                    message=f"Python {python_version.major}.{python_version.minor} may have compatibility issues",
                    severity=SeverityLevel.MEDIUM,
                    details={"version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"}
                ))
                
        except Exception as e:
            results.append(IntegrityCheckResult(
                component="dependency_conflict_check",
                passed=False,
                message=f"Dependency conflict check failed: {e}",
                severity=SeverityLevel.MEDIUM,
                details={"error": str(e)}
            ))
        
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
        Check critical and optional module imports with TA-Lib priority.
        
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
        
        # Check optional modules with special handling for TA-Lib
        for module_name in self.optional_modules:
            try:
                importlib.import_module(module_name)
                
                # Special validation for talib
                if module_name == "talib":
                    results.append(IntegrityCheckResult(
                        component=f"preferred_module_{module_name}",
                        passed=True,
                        message=f"Preferred TA-Lib module '{module_name}' available",
                        severity=SeverityLevel.LOW
                    ))
                else:
                    results.append(IntegrityCheckResult(
                        component=f"optional_module_{module_name}",
                        passed=True,
                        message=f"Optional module '{module_name}' imports successfully",
                        severity=SeverityLevel.LOW
                    ))
                    
            except ImportError as e:
                severity = SeverityLevel.HIGH if module_name == "talib" else SeverityLevel.MEDIUM
                
                results.append(IntegrityCheckResult(
                    component=f"optional_module_{module_name}",
                    passed=False,
                    message=f"Optional module '{module_name}' import failed: {e}",
                    severity=severity,
                    repair_recommended=True,
                    repair_type=RepairType.PACKAGE_INSTALL,
                    details={
                        "module": module_name, 
                        "error": str(e),
                        "install_hint": "brew install ta-lib" if module_name == "talib" and sys.platform == "darwin" else None
                    }
                ))
        
        return results
    
    # ... (rest of the methods remain the same as they were already comprehensive)
    async def _check_critical_components(self) -> List[IntegrityCheckResult]:
        """Check that critical classes and functions can be imported."""
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
    
    # Enhanced repair methods
    async def _repair_package_install(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to install missing package with TA-Lib special handling."""
        try:
            # Extract package name from component
            if "module_" in component:
                package_name = component.replace("module_", "").replace("optional_module_", "")
            else:
                package_name = component
            
            # Special handling for TA-Lib
            if package_name == "talib":
                # On macOS, suggest system-level install first
                if sys.platform == "darwin":
                    return (
                        "TA-Lib installation requires system-level install: brew install ta-lib",
                        False,
                        "Manual system installation required on macOS"
                    )
                else:
                    # Try pip install on other platforms
                    package_name = "TA-Lib"  # Correct PyPI package name
            
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
    
    async def _repair_dependency_conflict(self, component: str, issue: str) -> Tuple[str, bool, Optional[str]]:
        """Attempt to resolve dependency conflicts."""
        try:
            # Extract package name from component
            package_name = component.replace("problematic_package_", "")
            
            if package_name in self.problematic_packages:
                # Attempt to uninstall problematic package
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", package_name],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    return f"Successfully removed problematic package {package_name}", True, None
                else:
                    return f"Failed to remove {package_name}", False, result.stderr
            else:
                return "Unknown dependency conflict", False, "Unrecognized conflict type"
                
        except Exception as e:
            return "Dependency conflict resolution failed", False, str(e)
    
    # (Include all other existing methods - they remain unchanged)
    # ... rest of the methods from the previous implementation
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive Sentinel status including TA-Lib information.
        
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
            "talib_available": self._is_talib_available(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "problematic_packages": list(self.problematic_packages.keys())
        }


# (Include all remaining methods from the original implementation)
# Global sentinel instance
system_sentinel = SystemSentinel()


# CLI interface (enhanced with TA-Lib status)
async def main():
    """Enhanced main CLI interface for Sentinel."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Helios System Sentinel v3")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry-run integrity audit")
    parser.add_argument("--audit", action="store_true", help="Perform full integrity audit")
    parser.add_argument("--history", type=int, default=10, help="Show repair history (default: 10)")
    parser.add_argument("--talib-check", action="store_true", help="Check TA-Lib installation")
    
    args = parser.parse_args()
    
    if args.talib_check:
        print("Checking TA-Lib installation...")
        if system_sentinel._is_talib_available():
            print("✓ TA-Lib is properly installed and functional")
        else:
            print("✗ TA-Lib is not available")
            if sys.platform == "darwin":
                print("  macOS: Run 'brew install ta-lib' then 'pip install TA-Lib'")
            else:
                print("  Linux: Install system dependencies then 'pip install TA-Lib'")
    
    elif args.status:
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
    
    else:
        # Default: show repair history
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
