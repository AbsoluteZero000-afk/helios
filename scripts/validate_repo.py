#!/usr/bin/env python3
"""
Helios v3 Repository Validation Script

Comprehensive validation of repository structure, code quality,
and system readiness for local development and deployment.
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util


def print_section(title: str) -> None:
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, details: str = "") -> None:
    """Print formatted test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"{color}{status:<8}{reset} {test_name:<40} {details}")
    return passed


def run_command(command: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
    """Run system command and return result."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def validate_file_structure() -> List[bool]:
    """Validate repository file structure."""
    print_section("FILE STRUCTURE VALIDATION")
    
    results = []
    
    # Critical files
    critical_files = [
        "Dockerfile",
        "docker-compose.yml",
        "pyproject.toml",
        "requirements.txt",
        ".env.example",
        "main.py",
        "run_backtest.py",
        "pytest.ini",
        "README.md",
    ]
    
    for file_path in critical_files:
        exists = Path(file_path).exists()
        results.append(print_result(f"File: {file_path}", exists))
    
    # Critical directories
    critical_dirs = [
        "config",
        "core",
        "data",
        "execution",
        "risk",
        "strategies",
        "utils",
        "workers",
        "tests",
        "k8s",
        ".github/workflows"
    ]
    
    for dir_path in critical_dirs:
        exists = Path(dir_path).is_dir()
        results.append(print_result(f"Directory: {dir_path}", exists))
    
    # Specific module files
    module_files = [
        "config/settings.py",
        "config/strategy.yaml",
        "core/engine.py",
        "core/events.py",
        "utils/logger.py",
        "utils/sentinel.py",
        "utils/db.py",
        "utils/redis_queue.py",
        "utils/health.py",
        "execution/broker_base.py",
        "execution/alpaca_broker.py",
        "strategies/momentum_breakout.py",
        "workers/tasks.py",
        "data/storage.py",
        "data/processor.py"
    ]
    
    for file_path in module_files:
        exists = Path(file_path).exists()
        results.append(print_result(f"Module: {file_path}", exists))
    
    return results


def validate_python_syntax() -> bool:
    """Validate Python syntax using compileall."""
    print_section("PYTHON SYNTAX VALIDATION")
    
    success, stdout, stderr = run_command([sys.executable, "-m", "compileall", ".", "-q"])
    
    if success:
        print_result("Python syntax compilation", True, "All files compile successfully")
        return True
    else:
        print_result("Python syntax compilation", False, f"Errors: {stderr[:100]}...")
        if stderr:
            print(f"\nCompilation errors:\n{stderr}")
        return False


def validate_imports() -> List[bool]:
    """Validate critical module imports."""
    print_section("MODULE IMPORT VALIDATION")
    
    results = []
    
    critical_modules = [
        "config.settings",
        "core.engine",
        "core.events",
        "utils.logger",
        "utils.sentinel",
        "utils.db",
        "utils.redis_queue",
        "utils.health",
        "strategies.base",
        "strategies.trend_following",
        "strategies.mean_reversion",
        "strategies.momentum_breakout",
        "execution.broker_base",
        "execution.paper_broker",
        "execution.alpaca_broker",
        "risk.manager",
        "workers.tasks",
        "data.storage",
        "data.processor",
    ]
    
    for module_name in critical_modules:
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                Path(*module_name.split('.')) / "__init__.py" if module_name.count('.') > 0 else Path(f"{module_name}.py")
            )
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                results.append(print_result(f"Import: {module_name}", True))
            else:
                # Try regular import
                __import__(module_name)
                results.append(print_result(f"Import: {module_name}", True))
                
        except Exception as e:
            results.append(print_result(f"Import: {module_name}", False, f"Error: {str(e)[:50]}..."))
    
    return results


def validate_dependencies() -> List[bool]:
    """Validate external dependencies."""
    print_section("DEPENDENCY VALIDATION")
    
    results = []
    
    critical_packages = [
        "pandas",
        "numpy",
        "sqlalchemy",
        "redis",
        "celery",
        "aiohttp",
        "loguru",
        "pydantic",
        "psutil",
        "psycopg2",
    ]
    
    optional_packages = [
        "talib",
        "ta",
        "alpaca_trade_api",
        "slack_sdk",
        "yfinance"
    ]
    
    for package in critical_packages:
        try:
            __import__(package)
            results.append(print_result(f"Critical package: {package}", True))
        except ImportError as e:
            results.append(print_result(f"Critical package: {package}", False, f"Missing: {e}"))
    
    for package in optional_packages:
        try:
            __import__(package)
            results.append(print_result(f"Optional package: {package}", True))
        except ImportError:
            results.append(print_result(f"Optional package: {package}", False, "Not installed (optional)"))
    
    return results


def validate_configuration() -> List[bool]:
    """Validate configuration files and settings."""
    print_section("CONFIGURATION VALIDATION")
    
    results = []
    
    # Check .env.example
    env_example = Path(".env.example")
    if env_example.exists():
        with open(env_example) as f:
            env_content = f.read()
        
        required_vars = [
            "DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL",
            "LOG_LEVEL", "SENTINEL_ENABLED", "SLACK_WEBHOOK_URL"
        ]
        
        missing_vars = [var for var in required_vars if var not in env_content]
        
        if not missing_vars:
            results.append(print_result(".env.example completeness", True))
        else:
            results.append(print_result(".env.example completeness", False, f"Missing: {missing_vars[:3]}"))
    else:
        results.append(print_result(".env.example exists", False, "File missing"))
    
    # Check strategy.yaml
    strategy_config = Path("config/strategy.yaml")
    if strategy_config.exists():
        try:
            import yaml
            with open(strategy_config) as f:
                config = yaml.safe_load(f)
            
            has_strategies = bool(config.get('strategies', {}))
            results.append(print_result("strategy.yaml validity", has_strategies, f"Strategies: {len(config.get('strategies', {}))}"))
        except Exception as e:
            results.append(print_result("strategy.yaml validity", False, f"Parse error: {e}"))
    else:
        results.append(print_result("strategy.yaml exists", False, "File missing"))
    
    return results


def validate_docker_setup() -> List[bool]:
    """Validate Docker configuration."""
    print_section("DOCKER SETUP VALIDATION")
    
    results = []
    
    # Check Docker availability
    docker_available, _, _ = run_command(["docker", "--version"])
    results.append(print_result("Docker installation", docker_available))
    
    if docker_available:
        # Check Docker Compose
        compose_available, _, _ = run_command(["docker-compose", "--version"])
        results.append(print_result("Docker Compose installation", compose_available))
        
        # Validate Dockerfile syntax
        dockerfile_valid, _, _ = run_command(["docker", "build", "--dry-run", "."], timeout=10)
        results.append(print_result("Dockerfile syntax", dockerfile_valid))
        
        # Validate docker-compose.yml
        if compose_available:
            compose_valid, _, stderr = run_command(["docker-compose", "config", "-q"])
            results.append(print_result("docker-compose.yml syntax", compose_valid, stderr[:50] if stderr else ""))
    
    return results


def run_quick_tests() -> bool:
    """Run quick smoke tests."""
    print_section("QUICK SMOKE TESTS")
    
    # Check if pytest is available
    pytest_available, _, _ = run_command(["python", "-m", "pytest", "--version"])
    
    if not pytest_available:
        print_result("pytest availability", False, "pytest not installed")
        return False
    
    # Run quick tests
    success, stdout, stderr = run_command(["python", "-m", "pytest", "-q", "--tb=short"], timeout=120)
    
    if success:
        print_result("Quick test suite", True, "All tests passed")
        return True
    else:
        print_result("Quick test suite", False, f"Test failures detected")
        if stderr:
            print(f"\nTest errors:\n{stderr[:500]}...")
        return False


def validate_sentinel_dry_run() -> bool:
    """Run Sentinel dry-run validation."""
    print_section("SYSTEM SENTINEL VALIDATION")
    
    try:
        # Set minimal environment for dry run
        os.environ["SENTINEL_ENABLED"] = "true"
        os.environ["SENTINEL_AUTO_REPAIR"] = "false"
        os.environ["LOG_LEVEL"] = "ERROR"  # Reduce log noise
        
        # Run Sentinel dry-run
        success, stdout, stderr = run_command(
            ["python", "utils/sentinel.py", "--dry-run"],
            timeout=60
        )
        
        if success:
            print_result("Sentinel dry-run", True, "Integrity checks passed")
            return True
        else:
            print_result("Sentinel dry-run", False, "Issues detected")
            if stderr:
                print(f"\nSentinel output:\n{stderr[:300]}...")
            return False
            
    except Exception as e:
        print_result("Sentinel dry-run", False, f"Exception: {e}")
        return False


def generate_validation_report(results: Dict[str, List[bool]]) -> Dict[str, Any]:
    """Generate comprehensive validation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "overall_status": "unknown",
        "summary": {},
        "details": results
    }
    
    # Calculate summary statistics
    all_results = []
    for category, category_results in results.items():
        total = len(category_results)
        passed = sum(category_results)
        failed = total - passed
        
        report["summary"][category] = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0
        }
        
        all_results.extend(category_results)
    
    # Overall status
    total_tests = len(all_results)
    total_passed = sum(all_results)
    overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0.0
    
    if overall_pass_rate >= 0.95:
        report["overall_status"] = "excellent"
    elif overall_pass_rate >= 0.85:
        report["overall_status"] = "good"
    elif overall_pass_rate >= 0.70:
        report["overall_status"] = "acceptable"
    else:
        report["overall_status"] = "needs_attention"
    
    report["overall_pass_rate"] = overall_pass_rate
    report["total_tests"] = total_tests
    report["total_passed"] = total_passed
    report["total_failed"] = total_tests - total_passed
    
    return report


def main() -> int:
    """Main validation function."""
    print("\n🚀 Helios v3 Repository Validation")
    print(f"Validating repository at: {Path.cwd()}")
    
    # Run all validations
    validation_results = {
        "file_structure": validate_file_structure(),
        "dependencies": validate_dependencies(),
        "configuration": validate_configuration(),
        "docker_setup": validate_docker_setup(),
        "imports": validate_imports(),
    }
    
    # Add syntax validation
    syntax_valid = validate_python_syntax()
    validation_results["python_syntax"] = [syntax_valid]
    
    # Add test validation
    tests_passed = run_quick_tests()
    validation_results["tests"] = [tests_passed]
    
    # Add Sentinel validation
    sentinel_ok = validate_sentinel_dry_run()
    validation_results["sentinel"] = [sentinel_ok]
    
    # Generate and display report
    report = generate_validation_report(validation_results)
    
    print_section("VALIDATION SUMMARY")
    
    for category, summary in report["summary"].items():
        status = "✓" if summary["failed"] == 0 else f"✗ ({summary['failed']} failed)"
        print(f"{category.replace('_', ' ').title():<20} {status:<15} {summary['passed']}/{summary['total']} passed")
    
    print(f"\nOverall Status: {report['overall_status'].upper()}")
    print(f"Pass Rate: {report['overall_pass_rate']:.1%} ({report['total_passed']}/{report['total_tests']})")
    
    # Save report to file
    report_path = Path("validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    # Recommendations based on results
    if report["overall_pass_rate"] < 0.95:
        print("\n⚠️  RECOMMENDATIONS:")
        
        for category, results_list in validation_results.items():
            failed_count = len(results_list) - sum(results_list)
            if failed_count > 0:
                print(f"  - Fix {failed_count} issues in {category.replace('_', ' ')}")
        
        if not tests_passed:
            print("  - Run 'pytest -v' to see detailed test failures")
        
        if not sentinel_ok:
            print("  - Run 'python utils/sentinel.py --dry-run' for detailed Sentinel report")
    
    else:
        print("\n✅ Repository validation PASSED - Ready for development!")
        print("\nNext steps:")
        print("  1. Copy .env.example to .env and configure")
        print("  2. Run: docker-compose up --build")
        print("  3. Test: python run_backtest.py --symbol AAPL --strategy trend_following")
    
    # Return exit code based on critical failures
    critical_failures = (
        not syntax_valid or
        not any(validation_results["imports"]) or
        len(validation_results["file_structure"]) == 0
    )
    
    return 1 if critical_failures else 0


if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, str(Path.cwd()))
    
    # Import datetime after path setup
    from datetime import datetime
    
    exit_code = main()
    print(f"\n🎆 Validation completed with exit code: {exit_code}")
    sys.exit(exit_code)
