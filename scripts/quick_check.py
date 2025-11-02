#!/usr/bin/env python3
"""
Helios v3 Quick Validation and Setup Script

Fast validation script to verify system readiness and perform
basic smoke tests before full deployment.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Quick validation main function."""
    print("☀️ Helios v3 - Quick validation check")
    
    try:
        # Test basic imports
        print("Testing critical imports...")
        import config.settings
        import utils.logger
        import core.engine
        print("✓ Critical imports OK")
        
        # Test configuration loading
        print("Testing configuration...")
        settings = config.settings.get_settings()
        env_display = getattr(settings.environment, "value", settings.environment)
        print(f"✓ Settings loaded - Environment: {env_display}")
        
        # Test logging setup
        print("Testing logging...")
        utils.logger.setup_logging("INFO")
        logger = utils.logger.get_logger("validation")
        logger.info("Logging system operational")
        print("✓ Logging setup OK")
        
        print("\n✅ Quick validation PASSED - System ready!")
        print("\nNext steps:")
        print("  1. Run full validation: python scripts/validate_repo.py")
        print("  2. Setup environment: cp .env.example .env")
        print("  3. Start development: docker-compose up --build")
        
        return 0
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Install dependencies: pip install -r requirements.txt")
        return 1
    
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
