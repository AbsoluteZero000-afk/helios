#!/usr/bin/env python3
"""
Helios v3 Automated Setup Script (no-venv)

Automated installation and validation script that handles:
- TA-Lib installation hints (platform-specific)
- Dependencies installation (uses requirements-clean.txt)
- System validation (fixed temp script quoting)
- Initial configuration

Note: This version assumes you're already inside an active virtual environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_step(step_num: int, total_steps: int, description: str):
    """Print formatted step information."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 60)


def run_command(command: list, description: str = "") -> bool:
    """Run a command and return success status."""
    try:
        if description:
            print(description)
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Some tools write infos to stderr; show trimmed
            err = result.stderr.strip()
            if err:
                print(err)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def check_python_version() -> bool:
    v = sys.version_info
    ok = (v.major == 3 and v.minor >= 11)
    if ok:
        print(f"✅ Python {v.major}.{v.minor}.{v.micro} is compatible")
    else:
        print(f"❌ Python {v.major}.{v.minor} is not recommended (use 3.11+)")
    return ok


def ensure_talib_hint() -> None:
    system = platform.system().lower()
    if system == "darwin":
        print("🍎 macOS hint: If TA-Lib fails at runtime, run: brew install ta-lib && pip install TA-Lib")
    elif system == "linux":
        print("🐧 Linux hint: Install system libs if needed (Ubuntu: sudo apt-get install libta-lib-dev)")
    elif system == "windows":
        print("🪟 Windows hint: Use prebuilt TA-Lib wheels if pip build fails.")


def install_dependencies() -> bool:
    # Upgrade pip first
    if not run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip"):
        return False
    
    # Try TA-Lib wheel (non-fatal if fails)
    print("Installing TA-Lib Python package (non-fatal if it fails)...")
    run_command([sys.executable, "-m", "pip", "install", "TA-Lib"], "Installing TA-Lib")

    # Install from clean requirements (no websockets conflict)
    return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements-clean.txt"], "Installing dependencies from requirements-clean.txt")


def validate_installation() -> bool:
    # Write a robust temp validation script (no unterminated strings)
    test_lines = [
        "import sys\n",
        "print(f'Python: {sys.version}')\n",
        "\n",
        "# Core libs\n",
        "try:\n",
        "    import pandas, numpy\n",
        "    print('✅ Core libs: pandas, numpy')\n",
        "except Exception as e:\n",
        "    print(f'❌ Core libs failed: {e}')\n",
        "    raise\n",
        "\n",
        "# Infra libs\n",
        "try:\n",
        "    import sqlalchemy, redis, celery, loguru, pydantic\n",
        "    print('✅ Infra libs: SQLAlchemy, Redis, Celery, Loguru, Pydantic')\n",
        "except Exception as e:\n",
        "    print(f'❌ Infra libs failed: {e}')\n",
        "    raise\n",
        "\n",
        "# TA-Lib check (non-fatal)\n",
        "try:\n",
        "    import talib\n",
        "    talib.SMA([1.0,2.0,3.0,4.0,5.0], timeperiod=3)\n",
        "    print('✅ TA-Lib working')\n",
        "except ImportError:\n",
        "    print('⚠️  TA-Lib not available - pandas fallback will be used')\n",
        "except Exception as e:\n",
        "    print(f'⚠️  TA-Lib error: {e} - pandas fallback will be used')\n",
        "\n",
        "print('🎉 Validation successful')\n",
    ]
    tmp = Path(".setup_validation.py")
    tmp.write_text("".join(test_lines), encoding="utf-8")
    try:
        ok = run_command([sys.executable, str(tmp)], "Validating installation")
        return ok
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def create_env_file() -> bool:
    if Path(".env").exists():
        print("✅ .env file already exists")
        return True
    sample = Path(".env.example")
    if sample.exists():
        Path(".env").write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
        print("✅ Created .env from .env.example — edit it to add your secrets")
        return True
    print("⚠️  .env.example not found; create .env manually")
    return False


def print_next_steps():
    print("\n" + "=" * 60)
    print("🎉 HELIOS v3 SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1) Edit .env (Slack webhook, Alpaca keys, DB/Redis URLs)")
    print("  2) Quick checks: python scripts/quick_check.py")
    print("  3) Sentinel:   python utils/sentinel.py --status")
    print("  4) Backtest:    python run_backtest.py --symbol AAPL --strategy trend_following")
    print("  5) Docker:      docker-compose up --build")


def main():
    print("🚀 Helios v3 Automated Setup (no-venv)")
    print("=====================================")

    total = 5

    # 1. Python version
    print_step(1, total, "Checking Python Version")
    if not check_python_version():
        sys.exit(1)

    # 2. TA-Lib hints
    print_step(2, total, "TA-Lib System Hints")
    ensure_talib_hint()

    # 3. Install deps
    print_step(3, total, "Installing Python Dependencies")
    if not install_dependencies():
        print("❌ Setup failed: dependency installation failed")
        sys.exit(1)

    # 4. Validate
    print_step(4, total, "Validating Installation")
    if not validate_installation():
        print("❌ Setup failed: validation failed")
        sys.exit(1)

    # 5. Create .env
    print_step(5, total, "Creating Configuration Files")
    create_env_file()

    print_next_steps()


if __name__ == "__main__":
    main()
