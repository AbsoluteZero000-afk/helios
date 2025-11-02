"""
Helios Health Check System v3

Comprehensive health monitoring for all system components
including database, Redis, Celery, and application services.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

import psutil
from sqlalchemy import text

from config.settings import get_settings
from utils.logger import get_logger
from utils.db import db_manager
from utils.redis_queue import redis_manager

logger = get_logger("health")


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Individual health check result."""
    component: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    total_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Dict[str, int] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive system health checker.
    
    Performs health checks on all critical system components.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.settings = get_settings()
        self.timeout = self.settings.health_check_timeout
        self.retries = self.settings.health_check_retries
    
    async def check_all(self) -> SystemHealthReport:
        """
        Perform all health checks.
        
        Returns:
            SystemHealthReport: Complete health report
        """
        start_time = time.time()
        checks = []
        
        # Define all health checks
        health_checks = [
            ("system_resources", self._check_system_resources),
            ("database", self._check_database),
            ("redis", self._check_redis),
            ("celery", self._check_celery),
            ("disk_space", self._check_disk_space),
            ("memory", self._check_memory),
            ("application", self._check_application),
        ]
        
        # Run all checks concurrently
        tasks = [
            self._run_check_with_timeout(name, check_func)
            for name, check_func in health_checks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                checks.append(HealthCheckResult(
                    component="unknown",
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {result}",
                    duration_ms=0.0,
                    error=str(result)
                ))
            else:
                checks.append(result)
        
        total_duration = (time.time() - start_time) * 1000
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Generate summary
        summary = {
            status.value: sum(1 for check in checks if check.status == status)
            for status in HealthStatus
        }
        
        return SystemHealthReport(
            overall_status=overall_status,
            checks=checks,
            total_duration_ms=total_duration,
            summary=summary
        )
    
    async def _run_check_with_timeout(self, name: str, check_func) -> HealthCheckResult:
        """
        Run health check with timeout and retry logic.
        
        Args:
            name: Check name
            check_func: Check function to run
            
        Returns:
            HealthCheckResult: Check result
        """
        for attempt in range(self.retries + 1):
            try:
                result = await asyncio.wait_for(
                    check_func(),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                if attempt == self.retries:
                    return HealthCheckResult(
                        component=name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check timed out after {self.timeout}s",
                        duration_ms=self.timeout * 1000,
                        error="timeout"
                    )
                await asyncio.sleep(1)  # Brief delay before retry
            except Exception as e:
                if attempt == self.retries:
                    return HealthCheckResult(
                        component=name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check failed: {e}",
                        duration_ms=0.0,
                        error=str(e)
                    )
                await asyncio.sleep(1)  # Brief delay before retry
        
        # Should never reach here, but just in case
        return HealthCheckResult(
            component=name,
            status=HealthStatus.UNKNOWN,
            message="Unknown error occurred",
            duration_ms=0.0
        )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """
        Check system resource utilization.
        
        Returns:
            HealthCheckResult: System resources check result
        """
        start_time = time.time()
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get load average (Unix only)
            try:
                load_avg = psutil.getloadavg()
                load_1min = load_avg[0]
            except (AttributeError, OSError):
                load_1min = 0.0
            
            # Determine status
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = max(status, HealthStatus.CRITICAL)
                messages.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = max(status, HealthStatus.WARNING)
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            message = "; ".join(messages) if messages else f"System resources normal (CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%)"
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_bytes": memory.available,
                    "load_average_1min": load_1min,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"System resources check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """
        Check database connectivity and performance.
        
        Returns:
            HealthCheckResult: Database check result
        """
        start_time = time.time()
        
        try:
            # Test basic connectivity
            if not db_manager.test_connection():
                return HealthCheckResult(
                    component="database",
                    status=HealthStatus.CRITICAL,
                    message="Database connection failed",
                    duration_ms=(time.time() - start_time) * 1000,
                    error="connection_failed"
                )
            
            # Test query performance
            query_start = time.time()
            with db_manager.get_session() as session:
                result = session.execute(text("SELECT 1 as test, NOW() as timestamp"))
                row = result.fetchone()
            query_duration = (time.time() - query_start) * 1000
            
            # Get table information
            table_info = db_manager.get_table_info()
            
            # Determine status based on query performance
            if query_duration > 1000:  # 1 second
                status = HealthStatus.WARNING
                message = f"Database responsive but slow (query: {query_duration:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database healthy (query: {query_duration:.0f}ms)"
            
            return HealthCheckResult(
                component="database",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "query_duration_ms": query_duration,
                    "tables": table_info,
                    "connection_pool_size": self.settings.database_pool_size,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """
        Check Redis connectivity and performance.
        
        Returns:
            HealthCheckResult: Redis check result
        """
        start_time = time.time()
        
        try:
            # Test basic connectivity
            if not redis_manager.test_connection():
                return HealthCheckResult(
                    component="redis",
                    status=HealthStatus.CRITICAL,
                    message="Redis connection failed",
                    duration_ms=(time.time() - start_time) * 1000,
                    error="connection_failed"
                )
            
            # Test performance with a simple operation
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now(timezone.utc).isoformat()}
            
            op_start = time.time()
            redis_manager.set(test_key, test_value, ttl=60)
            retrieved_value = redis_manager.get(test_key)
            redis_manager.delete(test_key)
            op_duration = (time.time() - op_start) * 1000
            
            # Get Redis info
            redis_info = redis_manager.get_info()
            
            # Determine status
            if op_duration > 100:  # 100ms
                status = HealthStatus.WARNING
                message = f"Redis responsive but slow (ops: {op_duration:.0f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Redis healthy (ops: {op_duration:.0f}ms)"
            
            return HealthCheckResult(
                component="redis",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "operation_duration_ms": op_duration,
                    "redis_info": redis_info,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_celery(self) -> HealthCheckResult:
        """
        Check Celery worker connectivity.
        
        Returns:
            HealthCheckResult: Celery check result
        """
        start_time = time.time()
        
        try:
            # This is a simplified check - in a real implementation,
            # you would import your Celery app and check worker status
            
            # For now, we'll just check if Celery broker (Redis) is accessible
            # This should be replaced with actual Celery worker inspection
            
            status = HealthStatus.HEALTHY
            message = "Celery check not fully implemented - using Redis broker status"
            
            # Check broker connectivity (Redis in our case)
            broker_healthy = redis_manager.test_connection()
            
            if not broker_healthy:
                status = HealthStatus.CRITICAL
                message = "Celery broker (Redis) not accessible"
            
            return HealthCheckResult(
                component="celery",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "broker_url": self.settings.celery_broker_url,
                    "result_backend": self.settings.celery_result_backend,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="celery",
                status=HealthStatus.CRITICAL,
                message=f"Celery check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """
        Check available disk space.
        
        Returns:
            HealthCheckResult: Disk space check result
        """
        start_time = time.time()
        
        try:
            # Check disk space for root directory
            disk_usage = psutil.disk_usage('/')
            used_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)  # Convert to GB
            
            # Determine status
            if used_percent > 95 or free_gb < 1:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            elif used_percent > 90 or free_gb < 5:
                status = HealthStatus.WARNING
                message = f"Low disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            
            return HealthCheckResult(
                component="disk_space",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "used_percent": used_percent,
                    "free_bytes": disk_usage.free,
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Disk space check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_memory(self) -> HealthCheckResult:
        """
        Check memory usage details.
        
        Returns:
            HealthCheckResult: Memory check result
        """
        start_time = time.time()
        
        try:
            # Get process memory info
            process = psutil.Process()
            process_memory = process.memory_info()
            process_percent = process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Determine status
            if process_percent > 80:
                status = HealthStatus.CRITICAL
                message = f"Critical process memory usage: {process_percent:.1f}%"
            elif process_percent > 60:
                status = HealthStatus.WARNING
                message = f"High process memory usage: {process_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {process_percent:.1f}%"
            
            return HealthCheckResult(
                component="memory",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "process_memory_percent": process_percent,
                    "process_rss_bytes": process_memory.rss,
                    "process_vms_bytes": process_memory.vms,
                    "system_memory_percent": system_memory.percent,
                    "system_available_bytes": system_memory.available,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="memory",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def _check_application(self) -> HealthCheckResult:
        """
        Check application-specific health.
        
        Returns:
            HealthCheckResult: Application check result
        """
        start_time = time.time()
        
        try:
            # Check if critical modules can be imported
            critical_modules = [
                "config.settings",
                "core.engine",
                "utils.logger",
                "strategies.base",
            ]
            
            import_failures = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError as e:
                    import_failures.append(f"{module}: {e}")
            
            if import_failures:
                status = HealthStatus.CRITICAL
                message = f"Critical module import failures: {'; '.join(import_failures)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All critical modules importable"
            
            return HealthCheckResult(
                component="application",
                status=status,
                message=message,
                duration_ms=(time.time() - start_time) * 1000,
                details={
                    "critical_modules": critical_modules,
                    "import_failures": import_failures,
                    "app_version": self.settings.version,
                    "environment": self.settings.environment.value,
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="application",
                status=HealthStatus.CRITICAL,
                message=f"Application check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def _determine_overall_status(self, checks: List[HealthCheckResult]) -> HealthStatus:
        """
        Determine overall system status from individual checks.
        
        Args:
            checks: List of health check results
            
        Returns:
            HealthStatus: Overall system status
        """
        if not checks:
            return HealthStatus.UNKNOWN
        
        # If any check is critical, overall is critical
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        
        # If any check is warning, overall is warning
        if any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        
        # If all checks are healthy, overall is healthy
        if all(check.status == HealthStatus.HEALTHY for check in checks):
            return HealthStatus.HEALTHY
        
        # Otherwise, unknown
        return HealthStatus.UNKNOWN


# Global health checker instance
health_checker = HealthChecker()


# Convenience functions
async def check_system_health() -> SystemHealthReport:
    """Perform complete system health check."""
    return await health_checker.check_all()


def health_check() -> None:
    """Simple synchronous health check for Docker health check."""
    try:
        # Basic checks that can run synchronously
        
        # Check if we can import critical modules
        import config.settings
        import utils.logger
        
        # Check basic system resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        if cpu_percent > 95 or memory.percent > 95:
            raise Exception("System resources critically low")
        
        logger.debug("Health check passed")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise
