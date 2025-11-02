#!/usr/bin/env python3
"""
Helios v3 Automated Setup Script

Automated installation and validation script that handles:
- Virtual environment creation
- TA-Lib installation (with platform-specific guidance)
- Dependencies installation
- System validation
- Initial configuration
"""

import os
import sys
import subprocess
import platform
import venv
from pathlib import Path


def print_step(step_num: int, total_steps: int, description: str):
    """Print formatted step information."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 60)


def run_command(command: list, description: str = "") -> bool:
    """Run a command and return success status."""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not recommended (use 3.11+)")
        return False


def setup_virtual_environment() -> bool:
    """Create and setup virtual environment."""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        print("Creating virtual environment...")
        venv.create(".venv", with_pip=True)
        print("✅ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def get_python_executable() -> str:
    """Get the Python executable path for the virtual environment."""
    if platform.system() == "Windows":
        return ".venv/Scripts/python"
    else:
        return ".venv/bin/python"


def get_pip_executable() -> str:
    """Get the pip executable path for the virtual environment."""
    if platform.system() == "Windows":
        return ".venv/Scripts/pip"
    else:
        return ".venv/bin/pip"


def install_talib() -> bool:
    """Install TA-Lib with platform-specific instructions."""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("🍎 macOS detected - Installing TA-Lib via Homebrew...")
        
        # Check if homebrew is installed
        if not run_command(["brew", "--version"], "Checking Homebrew"):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
        
        # Install TA-Lib via Homebrew
        if not run_command(["brew", "install", "ta-lib"], "Installing TA-Lib system library"):
            print("❌ Failed to install TA-Lib via Homebrew")
            return False
        
        print("✅ TA-Lib system library installed")
        
    elif system == "linux":
        print("🐧 Linux detected - You may need to install TA-Lib system dependencies")
        print("   Ubuntu/Debian: sudo apt-get install libta-lib-dev")
        print("   CentOS/RHEL: sudo yum install ta-lib-devel")
        print("   Or compile from source: http://ta-lib.org/hdr_dw.html")
        
        # Continue anyway - pip might work
        
    elif system == "windows":
        print("🪟 Windows detected - TA-Lib installation may require pre-compiled wheels")
        
    else:
        print(f"❓ Unknown system: {system} - Proceeding with pip installation")
    
    return True


def install_dependencies() -> bool:
    """Install Python dependencies."""
    pip_exe = get_pip_executable()
    
    # Upgrade pip first
    if not run_command([pip_exe, "install", "--upgrade", "pip"], "Upgrading pip"):
        print("❌ Failed to upgrade pip")
        return False
    
    # Install TA-Lib Python package
    print("Installing TA-Lib Python package...")
    if not run_command([pip_exe, "install", "TA-Lib"], "Installing TA-Lib"):
        print("❌ TA-Lib installation failed")
        print("   This is common on some systems. The system will fall back to pandas calculations.")
        print("   For better performance, please install TA-Lib manually:")
        if platform.system().lower() == "darwin":
            print("   macOS: brew install ta-lib && pip install TA-Lib")
        else:
            print("   See: https://ta-lib.org/install/")
    
    # Install core dependencies using the clean requirements
    print("Installing core dependencies...")
    if not run_command([pip_exe, "install", "-r", "requirements-clean.txt"], "Installing dependencies"):
        print("❌ Failed to install core dependencies")
        return False
    
    print("✅ Dependencies installed successfully")
    return True


def validate_installation() -> bool:
    """Validate the installation."""
    python_exe = get_python_executable()
    
    # Test basic imports
    test_script = '''
import sys
print(f"Python: {sys.version}")

# Test critical imports
try:
    import pandas as pd
    import numpy as np
    print("✅ Core libraries: pandas, numpy")
except ImportError as e:
    print(f"❌ Core libraries failed: {e}")
    sys.exit(1)

# Test TA-Lib
try:
    import talib
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    sma = talib.SMA(test_data, timeperiod=5)
    print("✅ TA-Lib working correctly")
except ImportError:
    print("⚠️  TA-Lib not available - will use pandas fallback")
except Exception as e:
    print(f"⚠️  TA-Lib error: {e} - will use pandas fallback")

# Test other critical imports
try:
    import sqlalchemy
    import redis
    import celery
    import loguru
    import pydantic
    print("✅ Infrastructure libraries: SQLAlchemy, Redis, Celery, Loguru, Pydantic")
except ImportError as e:
    print(f"❌ Infrastructure libraries failed: {e}")
    sys.exit(1)

print("\n🎉 Installation validation successful!")
'''
    
    # Write test script to temporary file
    with open("temp_test.py", "w") as f:
        f.write(test_script)
    
    try:
        success = run_command([python_exe, "temp_test.py"], "Validating installation")
        os.remove("temp_test.py")
        return success
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
        return False


def create_env_file() -> bool:
    """Create .env file from template if it doesn't exist."""
    if Path(".env").exists():
        print("✅ .env file already exists")
        return True
    
    if Path(".env.example").exists():
        try:
            # Copy .env.example to .env
            with open(".env.example", "r") as src:
                content = src.read()
            
            with open(".env", "w") as dst:
                dst.write(content)
            
            print("✅ Created .env file from template")
            print("   Please edit .env to add your API keys and configuration")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("⚠️  .env.example not found - you'll need to create .env manually")
        return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 HELIOS v3 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Activate Virtual Environment:")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    
    print("\n2. Configure Environment:")
    print("   - Edit .env file with your API keys")
    print("   - Add Slack webhook URL for notifications")
    print("   - Add Alpaca API keys for trading")
    
    print("\n3. Verify Installation:")
    print("   python scripts/quick_check.py")
    print("   python utils/sentinel.py --status")
    
    print("\n4. Run Your First Backtest:")
    print("   python run_backtest.py --symbol AAPL --strategy trend_following")
    
    print("\n5. Start Development Environment:")
    print("   docker-compose up --build")
    
    print("\n📚 Documentation:")
    print("   - README.md: Complete setup and usage guide")
    print("   - config/strategy.yaml: Strategy configuration")
    print("   - UPGRADE_SUMMARY.md: v3 feature overview")
    
    print("\n🆘 Need Help?")
    print("   - Check logs: tail -f logs/helios.log")
    print("   - Validate system: python scripts/validate_repo.py")
    print("   - Test TA-Lib: python utils/sentinel.py --talib-check")
    
    print("\n" + "=" * 60)


def main():
    """Main setup function."""
    print("🚀 Helios v3 Automated Setup")
    print("==============================")
    
    total_steps = 7
    current_step = 0
    
    # Step 1: Check Python version
    current_step += 1
    print_step(current_step, total_steps, "Checking Python Version")
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        sys.exit(1)
    
    # Step 2: Setup virtual environment
    current_step += 1
    print_step(current_step, total_steps, "Setting Up Virtual Environment")
    if not setup_virtual_environment():
        print("❌ Setup failed: Virtual environment creation failed")
        sys.exit(1)
    
    # Step 3: Install TA-Lib system dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing TA-Lib System Dependencies")
    if not install_talib():
        print("❌ Setup failed: TA-Lib system installation failed")
        sys.exit(1)
    
    # Step 4: Install Python dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing Python Dependencies")
    if not install_dependencies():
        print("❌ Setup failed: Python dependencies installation failed")
        sys.exit(1)
    
    # Step 5: Validate installation
    current_step += 1
    print_step(current_step, total_steps, "Validating Installation")
    if not validate_installation():
        print("❌ Setup failed: Installation validation failed")
        sys.exit(1)
    
    # Step 6: Create .env file
    current_step += 1
    print_step(current_step, total_steps, "Creating Configuration Files")
    create_env_file()
    
    # Step 7: Complete and show next steps
    current_step += 1
    print_step(current_step, total_steps, "Setup Complete")
    print_next_steps()


if __name__ == "__main__":
    main()
