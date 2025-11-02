"""
Helios Main Entry Point v3

Enhanced main application entry point with comprehensive initialization,
Sentinel monitoring, database migrations, and graceful shutdown handling.
"""

import sys
import signal
import asyncio
import traceback
from typing import Optional
from datetime import datetime, timezone

from config.settings import get_settings
from utils.logger import setup_logging, get_logger
from utils.sentinel import system_sentinel
from utils.db import initialize_database
from utils.redis_queue import initialize_redis
from utils.slack import slack_notifier
from core.engine import TradingEngine, EngineMode

# Initialize logging first
setup_logging()
logger = get_logger("main")

# Global engine instance for signal handling
engine_instance: Optional[TradingEngine] = None


def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    signal_name = signal.Signals(signum).name
    logger.info(f"Received shutdown signal: {signal_name}")
    
    if engine_instance:
        # Create new event loop for shutdown if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # Schedule shutdown task
            loop.create_task(shutdown_application())
        else:
            # Run shutdown directly
            loop.run_until_complete(shutdown_application())
    
    sys.exit(0)


async def initialize_system() -> bool:
    """
    Initialize all system components.
    
    Returns:
        bool: True if initialization successful
    """
    try:
        logger.info("Initializing Helios v3 system components...")
        
        # Initialize database
        logger.info("Initializing database connection...")
        initialize_database()
        
        # Initialize Redis
        logger.info("Initializing Redis connection...")
        initialize_redis()
        
        # Start System Sentinel
        if get_settings().sentinel_enabled:
            logger.info("Starting System Sentinel monitoring...")
            await system_sentinel.start_monitoring()
        else:
            logger.info("System Sentinel disabled")
        
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        logger.error(traceback.format_exc())
        
        # Send critical alert
        try:
            await slack_notifier.system_alert(
                f"Helios v3 initialization FAILED: {e}",
                component="main",
                severity="critical",
                stack_trace=traceback.format_exc()[:1000]
            )
        except Exception:
            pass  # Don't let notification failures block shutdown
        
        return False


async def shutdown_application() -> None:
    """
    Gracefully shutdown the application and all components.
    """
    try:
        logger.info("Starting graceful shutdown...")
        
        # Stop trading engine if running
        global engine_instance
        if engine_instance:
            logger.info("Stopping trading engine...")
            await engine_instance.stop()
            engine_instance = None
        
        # Stop System Sentinel
        if system_sentinel.is_running:
            logger.info("Stopping System Sentinel...")
            await system_sentinel.stop_monitoring()
        
        # Cleanup database connections
        try:
            from utils.db import db_manager
            db_manager.cleanup()
        except Exception as e:
            logger.warning(f"Database cleanup warning: {e}")
        
        # Cleanup Redis connections
        try:
            from utils.redis_queue import redis_manager
            redis_manager.cleanup()
        except Exception as e:
            logger.warning(f"Redis cleanup warning: {e}")
        
        # Send shutdown notification
        try:
            await slack_notifier.system_alert(
                "Helios v3 shutdown completed",
                component="main",
                severity="info",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        except Exception:
            pass  # Don't let notification failures block shutdown
        
        logger.info("Graceful shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.error(traceback.format_exc())


async def run_trading_engine() -> None:
    """
    Run the main trading engine.
    """
    global engine_instance
    
    try:
        # Create and configure trading engine
        settings = get_settings()
        
        # Determine engine mode based on configuration
        if settings.trading_mode.value == "backtest":
            mode = EngineMode.BACKTEST
        elif settings.trading_mode.value == "paper":
            mode = EngineMode.PAPER
        elif settings.trading_mode.value == "live":
            mode = EngineMode.LIVE
        else:
            mode = EngineMode.SIMULATION
        
        logger.info(f"Starting trading engine in {mode.value} mode")
        
        engine_instance = TradingEngine(mode=mode)
        
        # Load strategy configuration
        await load_strategies(engine_instance)
        
        # Start the engine
        await engine_instance.start()
        
        # Send startup notification
        await slack_notifier.system_alert(
            f"Helios v3 trading engine started in {mode.value} mode",
            component="engine",
            severity="success",
            mode=mode.value,
            initial_capital=settings.initial_capital
        )
        
        # Keep engine running
        logger.info("Trading engine running. Press Ctrl+C to stop.")
        
        # Monitor engine health
        while engine_instance.is_running():
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Perform basic health check
            engine_stats = engine_instance.get_stats()
            if engine_stats.get("error_count", 0) > 100:
                logger.warning("High error count detected, considering restart")
        
        logger.info("Trading engine stopped")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Trading engine error: {e}")
        logger.error(traceback.format_exc())
        
        # Send error notification
        await slack_notifier.system_alert(
            f"Trading engine error: {e}",
            component="engine",
            severity="critical"
        )
    finally:
        # Ensure engine is stopped
        if engine_instance:
            await engine_instance.stop()


async def load_strategies(engine: TradingEngine) -> None:
    """
    Load and configure trading strategies.
    
    Args:
        engine: Trading engine instance
    """
    try:
        import yaml
        from strategies.base import StrategyConfig
        from strategies.trend_following import TrendFollowingStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.momentum_breakout import MomentumBreakoutStrategy
        
        settings = get_settings()
        
        # Load strategy configuration
        config_path = Path(settings.strategy_config_file)
        if not config_path.exists():
            logger.warning(f"Strategy config file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            strategy_config = yaml.safe_load(f)
        
        strategies_config = strategy_config.get('strategies', {})
        
        # Strategy class mapping
        strategy_classes = {
            'trend_following': TrendFollowingStrategy,
            'mean_reversion': MeanReversionStrategy,
            'momentum_breakout': MomentumBreakoutStrategy,
        }
        
        # Load enabled strategies
        for strategy_name, config in strategies_config.items():
            if not config.get('enabled', False):
                logger.info(f"Strategy '{strategy_name}' disabled, skipping")
                continue
            
            strategy_class = strategy_classes.get(strategy_name)
            if not strategy_class:
                logger.warning(f"Unknown strategy type: {strategy_name}")
                continue
            
            # Create strategy configuration
            strategy_cfg = StrategyConfig(
                name=strategy_name,
                enabled=config['enabled'],
                symbols=config.get('symbols', []),
                parameters=config.get('parameters', {}),
                risk_limits=config.get('risk_limits', {}),
                position_sizing=config.get('position_sizing', {})
            )
            
            # Create and register strategy
            strategy = strategy_class(strategy_cfg)
            engine.strategies[strategy_name] = strategy
            
            # Add symbols to engine
            for symbol in strategy_cfg.symbols:
                engine.add_symbol(symbol)
            
            logger.info(
                f"Loaded strategy '{strategy_name}' with {len(strategy_cfg.symbols)} symbols"
            )
        
        logger.info(f"Loaded {len(engine.strategies)} strategies")
        
    except Exception as e:
        logger.error(f"Failed to load strategies: {e}")
        raise


async def main_async() -> None:
    """
    Main asynchronous application entry point.
    """
    try:
        logger.info("Starting Helios v3...")
        
        # Initialize system components
        if not await initialize_system():
            logger.critical("System initialization failed, exiting")
            sys.exit(1)
        
        # Run the trading engine
        await run_trading_engine()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Critical application error: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    finally:
        # Ensure clean shutdown
        await shutdown_application()


def main() -> None:
    """
    Main application entry point.
    
    Sets up signal handlers and runs the async main function.
    """
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main async function
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Application terminated by keyboard interrupt")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
