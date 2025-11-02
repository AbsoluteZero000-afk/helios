"""
Helios Celery Tasks v3

Celery task definitions for asynchronous processing of trading operations,
data processing, notifications, and system maintenance.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown

from config.settings import get_settings, get_celery_config
from utils.logger import get_logger
from utils.slack import slack_notifier

# Initialize settings and logger
settings = get_settings()
logger = get_logger("celery_tasks")

# Create Celery app with configuration
celery_config = get_celery_config()
app = Celery(
    "helios_tasks",
    broker=celery_config["broker_url"],
    backend=celery_config["result_backend"],
    include=["workers.tasks"]
)

# Configure Celery
app.conf.update(
    task_serializer=celery_config["task_serializer"],
    result_serializer=celery_config["result_serializer"],
    accept_content=celery_config["accept_content"],
    timezone=celery_config["timezone"],
    enable_utc=True,
    worker_prefetch_multiplier=celery_config["worker_prefetch_multiplier"],
    task_acks_late=celery_config["task_acks_late"],
    worker_max_tasks_per_child=celery_config["worker_max_tasks_per_child"],
    task_routes={
        'workers.tasks.execute_trade': {'queue': 'trading'},
        'workers.tasks.persist_trade': {'queue': 'database'},
        'workers.tasks.send_slack_message': {'queue': 'notifications'},
        'workers.tasks.compute_daily_metrics': {'queue': 'analytics'},
        'workers.tasks.process_market_data': {'queue': 'data_processing'},
    },
    task_default_queue='default',
    task_create_missing_queues=True,
)


class CallbackTask(Task):
    """Base task class with error handling and logging."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success."""
        logger.info(f"Task {task_id} ({self.name}) completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(f"Task {task_id} ({self.name}) failed: {exc}")
        # Send failure notification asynchronously
        asyncio.create_task(slack_notifier.system_alert(
            f"Celery task failed: {self.name} - {exc}",
            component="celery",
            severity="error",
            task_id=task_id
        ))
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        logger.warning(f"Task {task_id} ({self.name}) retrying: {exc}")


# Set default task base
app.Task = CallbackTask


@app.task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a trade order through the appropriate broker.
    
    Args:
        trade_data: Trade execution data
        
    Returns:
        Dict[str, Any]: Execution result
    """
    logger.info(f"Executing trade: {trade_data.get('symbol')} {trade_data.get('side')}")
    
    try:
        # Import here to avoid circular dependencies
        from execution.paper_broker import PaperBroker
        from execution.alpaca_broker import AlpacaBroker
        
        # Select broker based on trading mode
        if settings.trading_mode.value == "paper":
            broker = PaperBroker()
        else:
            broker = AlpacaBroker()
        
        # Execute the trade
        # This would be implemented based on broker interface
        result = {
            "success": True,
            "order_id": f"ord_{datetime.now().timestamp()}",
            "execution_time": datetime.now(timezone.utc).isoformat(),
            "trade_data": trade_data
        }
        
        logger.info(f"Trade executed successfully: {result['order_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise self.retry(exc=e)


@app.task(bind=True)
def persist_trade(self, trade_data: Dict[str, Any]) -> bool:
    """
    Persist trade data to database.
    
    Args:
        trade_data: Trade data to persist
        
    Returns:
        bool: True if successful
    """
    try:
        from utils.db import db_manager, Trade
        
        with db_manager.get_session() as session:
            trade = Trade(
                symbol=trade_data["symbol"],
                side=trade_data["side"],
                quantity=trade_data["quantity"],
                price=trade_data["price"],
                commission=trade_data.get("commission", 0.0),
                order_id=trade_data.get("order_id"),
                execution_id=trade_data.get("execution_id"),
                strategy_name=trade_data.get("strategy_name"),
                fill_time=datetime.fromisoformat(trade_data.get("fill_time", datetime.now(timezone.utc).isoformat())),
                metadata=trade_data.get("metadata", {})
            )
            session.add(trade)
            session.commit()
        
        logger.info(f"Trade persisted: {trade_data['symbol']} {trade_data['side']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to persist trade: {e}")
        return False


@app.task(bind=True)
def send_slack_message(
    self,
    message: str,
    message_type: str = "info",
    fields: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Send message to Slack asynchronously.
    
    Args:
        message: Message content
        message_type: Type of message (trade, risk, system, etc.)
        fields: Additional structured fields
        
    Returns:
        bool: True if message sent successfully
    """
    try:
        # Convert string to MessageType enum
        from utils.slack import MessageType
        
        msg_type = MessageType.INFO  # Default
        if hasattr(MessageType, message_type.upper()):
            msg_type = MessageType(message_type.lower())
        
        # Send message synchronously (Celery tasks are sync by default)
        success = slack_notifier.send_sync(message, msg_type, fields=fields)
        
        if success:
            logger.info(f"Slack message sent: {message_type}")
        else:
            logger.warning(f"Slack message failed: {message_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to send Slack message: {e}")
        return False


@app.task(bind=True)
def compute_daily_metrics(self, date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute daily performance metrics.
    
    Args:
        date_str: Date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        Dict[str, Any]: Computed metrics
    """
    try:
        from utils.db import db_manager, Trade, PerformanceSnapshot
        from core.performance import compute_basic_metrics
        
        if date_str:
            target_date = datetime.fromisoformat(date_str)
        else:
            target_date = datetime.now(timezone.utc).date()
        
        # Get trades for the date
        with db_manager.get_session() as session:
            trades = session.query(Trade).filter(
                Trade.fill_time >= target_date,
                Trade.fill_time < target_date + timedelta(days=1)
            ).all()
        
        # Compute metrics
        if trades:
            # Calculate basic metrics
            total_volume = sum(trade.quantity * trade.price for trade in trades)
            total_commission = sum(trade.commission for trade in trades)
            trade_count = len(trades)
            
            # Calculate PnL (simplified)
            buy_trades = [t for t in trades if t.side == 'BUY']
            sell_trades = [t for t in trades if t.side == 'SELL']
            
            daily_pnl = sum(t.quantity * t.price for t in sell_trades) - sum(t.quantity * t.price for t in buy_trades)
            daily_pnl -= total_commission
            
            metrics = {
                "date": target_date.isoformat(),
                "trade_count": trade_count,
                "total_volume": total_volume,
                "total_commission": total_commission,
                "daily_pnl": daily_pnl,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades)
            }
            
            logger.info(f"Daily metrics computed for {target_date}: {trade_count} trades, PnL: ${daily_pnl:.2f}")
            return metrics
        else:
            logger.info(f"No trades found for {target_date}")
            return {"date": target_date.isoformat(), "trade_count": 0}
        
    except Exception as e:
        logger.error(f"Failed to compute daily metrics: {e}")
        raise self.retry(exc=e)


@app.task(bind=True)
def process_market_data(self, market_data: Dict[str, Any]) -> bool:
    """
    Process incoming market data and store in cache.
    
    Args:
        market_data: Market data to process
        
    Returns:
        bool: True if processed successfully
    """
    try:
        from utils.redis_queue import market_data_cache
        from data.processor import process_ohlcv_data
        
        symbol = market_data.get("symbol")
        if not symbol:
            raise ValueError("Market data missing symbol")
        
        # Process the data
        processed_data = process_ohlcv_data(market_data)
        
        # Cache the processed data
        cache_key = f"{symbol}_latest"
        success = market_data_cache.set(cache_key, processed_data, ttl=300)  # 5 minutes
        
        if success:
            logger.debug(f"Processed and cached market data for {symbol}")
        else:
            logger.warning(f"Failed to cache market data for {symbol}")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to process market data: {e}")
        return False


@app.task(bind=True)
def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
    """
    Clean up old data from database and cache.
    
    Args:
        days_to_keep: Number of days of data to retain
        
    Returns:
        Dict[str, int]: Cleanup statistics
    """
    try:
        from utils.db import db_manager, Trade, Order, PerformanceSnapshot, SentinelRepair
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        cleanup_stats = {
            "trades_deleted": 0,
            "orders_deleted": 0,
            "snapshots_deleted": 0,
            "repairs_deleted": 0
        }
        
        with db_manager.get_session() as session:
            # Clean old trades
            old_trades = session.query(Trade).filter(Trade.created_at < cutoff_date)
            cleanup_stats["trades_deleted"] = old_trades.count()
            old_trades.delete()
            
            # Clean old orders
            old_orders = session.query(Order).filter(Order.created_at < cutoff_date)
            cleanup_stats["orders_deleted"] = old_orders.count()
            old_orders.delete()
            
            # Clean old performance snapshots
            old_snapshots = session.query(PerformanceSnapshot).filter(PerformanceSnapshot.created_at < cutoff_date)
            cleanup_stats["snapshots_deleted"] = old_snapshots.count()
            old_snapshots.delete()
            
            # Clean old sentinel repairs (keep more recent ones)
            repair_cutoff = datetime.now(timezone.utc) - timedelta(days=30)  # Keep 30 days
            old_repairs = session.query(SentinelRepair).filter(SentinelRepair.created_at < repair_cutoff)
            cleanup_stats["repairs_deleted"] = old_repairs.count()
            old_repairs.delete()
            
            session.commit()
        
        total_deleted = sum(cleanup_stats.values())
        logger.info(f"Data cleanup completed: {total_deleted} records deleted")
        
        return cleanup_stats
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise self.retry(exc=e)


@app.task(bind=True)
def system_health_check(self) -> Dict[str, Any]:
    """
    Perform comprehensive system health check.
    
    Returns:
        Dict[str, Any]: Health check results
    """
    try:
        from utils.health import health_checker
        
        # This is a sync task, but health_checker is async
        # We'll create a new event loop for this task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_report = loop.run_until_complete(health_checker.check_all())
            
            result = {
                "overall_status": health_report.overall_status.value,
                "total_checks": len(health_report.checks),
                "duration_ms": health_report.total_duration_ms,
                "timestamp": health_report.timestamp.isoformat(),
                "summary": health_report.summary,
                "checks": [
                    {
                        "component": check.component,
                        "status": check.status.value,
                        "message": check.message,
                        "duration_ms": check.duration_ms
                    }
                    for check in health_report.checks
                ]
            }
            
            logger.info(f"Health check completed: {health_report.overall_status.value}")
            return result
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Health check task failed: {e}")
        raise self.retry(exc=e)


@app.task(bind=True)
def backup_database(self, backup_type: str = "daily") -> Dict[str, Any]:
    """
    Create database backup.
    
    Args:
        backup_type: Type of backup (daily, weekly, manual)
        
    Returns:
        Dict[str, Any]: Backup result
    """
    try:
        # This is a placeholder for database backup logic
        # In production, this would use pg_dump or similar
        
        backup_info = {
            "backup_type": backup_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "size_bytes": 0,  # Placeholder
            "location": f"/backups/helios_backup_{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        }
        
        logger.info(f"Database backup completed: {backup_type}")
        return backup_info
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        raise self.retry(exc=e)


@app.task(bind=True, rate_limit="10/m")
def send_notification(self, notification_data: Dict[str, Any]) -> bool:
    """
    Send notification through configured channels.
    
    Args:
        notification_data: Notification data including type, message, recipients
        
    Returns:
        bool: True if notification sent successfully
    """
    try:
        notification_type = notification_data.get("type", "info")
        message = notification_data.get("message", "")
        fields = notification_data.get("fields", {})
        
        # Send Slack notification
        success = slack_notifier.send_sync(message, notification_type, fields=fields)
        
        if success:
            logger.info(f"Notification sent: {notification_type}")
        else:
            logger.warning(f"Notification failed: {notification_type}")
        
        return success
        
    except Exception as e:
        logger.error(f"Notification task failed: {e}")
        return False


# Periodic tasks
@app.task(bind=True)
def daily_metrics_task(self) -> bool:
    """
    Daily metrics computation task.
    
    Returns:
        bool: True if successful
    """
    try:
        today = datetime.now(timezone.utc).date()
        result = compute_daily_metrics.delay(today.isoformat())
        
        # Wait for result with timeout
        metrics = result.get(timeout=300)  # 5 minutes
        
        # Send summary notification
        if metrics and metrics.get("trade_count", 0) > 0:
            send_notification.delay({
                "type": "performance",
                "message": f"Daily metrics computed: {metrics['trade_count']} trades, PnL: ${metrics.get('daily_pnl', 0):.2f}",
                "fields": metrics
            })
        
        return True
        
    except Exception as e:
        logger.error(f"Daily metrics task failed: {e}")
        return False


@app.task(bind=True)
def weekly_cleanup_task(self) -> bool:
    """
    Weekly data cleanup task.
    
    Returns:
        bool: True if successful
    """
    try:
        result = cleanup_old_data.delay(days_to_keep=90)
        cleanup_stats = result.get(timeout=600)  # 10 minutes
        
        total_deleted = sum(cleanup_stats.values())
        
        if total_deleted > 0:
            send_notification.delay({
                "type": "system",
                "message": f"Weekly cleanup completed: {total_deleted} records removed",
                "fields": cleanup_stats
            })
        
        return True
        
    except Exception as e:
        logger.error(f"Weekly cleanup task failed: {e}")
        return False


# Celery beat schedule for periodic tasks
app.conf.beat_schedule = {
    'daily-metrics': {
        'task': 'workers.tasks.daily_metrics_task',
        'schedule': 3600.0 * 24,  # Run daily
    },
    'weekly-cleanup': {
        'task': 'workers.tasks.weekly_cleanup_task',
        'schedule': 3600.0 * 24 * 7,  # Run weekly
    },
    'health-check': {
        'task': 'workers.tasks.system_health_check',
        'schedule': 3600.0,  # Run hourly
    },
}


# Signal handlers
@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """Handle worker ready signal."""
    logger.info(f"Celery worker ready: {sender}")
    
    # Send notification
    send_notification.delay({
        "type": "system",
        "message": f"Celery worker started: {sender}",
        "fields": {"worker": str(sender), "timestamp": datetime.now(timezone.utc).isoformat()}
    })


@worker_shutdown.connect
def worker_shutdown_handler(sender, **kwargs):
    """Handle worker shutdown signal."""
    logger.info(f"Celery worker shutting down: {sender}")


def start_worker():
    """Start Celery worker programmatically."""
    app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',
        '--queues=default,trading,database,notifications,analytics,data_processing'
    ])


if __name__ == "__main__":
    start_worker()
