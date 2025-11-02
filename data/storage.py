"""
Helios Data Storage v3

Simple data persistence layer for market data, signals, and performance metrics.
Provides efficient storage and retrieval with automatic cleanup and archiving.
"""

import json
import gzip
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from config.settings import get_settings
from utils.logger import data_logger as logger
from utils.redis_queue import market_data_cache, strategy_cache
from utils.db import db_manager, Trade, Order, Position as DBPosition


class DataStorage:
    """
    Data storage manager for market data, trading records, and analytics.
    
    Provides efficient storage to both database and cache layers with
    automatic cleanup, compression, and archival capabilities.
    """
    
    def __init__(self):
        """Initialize data storage manager."""
        self.settings = get_settings()
        self.storage_path = Path(self.settings.data_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "market_data").mkdir(exist_ok=True)
        (self.storage_path / "signals").mkdir(exist_ok=True)
        (self.storage_path / "performance").mkdir(exist_ok=True)
        (self.storage_path / "backups").mkdir(exist_ok=True)
        (self.storage_path / "archive").mkdir(exist_ok=True)
        
        logger.info(f"Data storage initialized at {self.storage_path}")
    
    async def store_market_data(
        self,
        symbol: str,
        data: Union[Dict[str, Any], pd.DataFrame],
        cache_ttl: Optional[int] = None
    ) -> bool:
        """
        Store market data to cache and optionally to disk.
        
        Args:
            symbol: Trading symbol
            data: Market data to store
            cache_ttl: Cache time-to-live in seconds
            
        Returns:
            bool: True if storage successful
        """
        try:
            cache_key = f"{symbol}_market_data"
            ttl = cache_ttl or self.settings.data_cache_ttl
            
            # Store in Redis cache
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to dict for caching
                cache_data = data.to_dict('records')
            else:
                cache_data = data
            
            cache_success = market_data_cache.set(cache_key, cache_data, ttl=ttl)
            
            # Store to disk (daily files)
            file_success = await self._store_market_data_to_disk(symbol, data)
            
            if cache_success and file_success:
                logger.debug(f"Market data stored for {symbol}")
                return True
            else:
                logger.warning(f"Partial failure storing market data for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store market data for {symbol}: {e}")
            return False
    
    async def _store_market_data_to_disk(
        self,
        symbol: str,
        data: Union[Dict[str, Any], pd.DataFrame]
    ) -> bool:
        """
        Store market data to daily files on disk.
        
        Args:
            symbol: Trading symbol
            data: Market data to store
            
        Returns:
            bool: True if successful
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            file_path = self.storage_path / "market_data" / f"{symbol}_{today}.json.gz"
            
            # Convert data to JSON-serializable format
            if isinstance(data, pd.DataFrame):
                json_data = data.to_dict('records')
            else:
                json_data = data if isinstance(data, list) else [data]
            
            # Append to existing file or create new
            existing_data = []
            if file_path.exists():
                with gzip.open(file_path, 'rt') as f:
                    existing_data = json.load(f)
            
            # Combine data
            if isinstance(existing_data, list):
                existing_data.extend(json_data)
            else:
                existing_data = json_data
            
            # Write compressed data
            with gzip.open(file_path, 'wt') as f:
                json.dump(existing_data, f, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store market data to disk for {symbol}: {e}")
            return False
    
    async def get_market_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        from_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve market data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data range
            end_date: End date for data range
            from_cache: Whether to try cache first
            
        Returns:
            Optional[pd.DataFrame]: Market data or None
        """
        try:
            # Try cache first if requested
            if from_cache:
                cache_key = f"{symbol}_market_data"
                cached_data = market_data_cache.get(cache_key)
                
                if cached_data:
                    df = pd.DataFrame(cached_data)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filter by date range if specified
                    if start_date or end_date:
                        df = self._filter_by_date_range(df, start_date, end_date)
                    
                    logger.debug(f"Retrieved {len(df)} market data records from cache for {symbol}")
                    return df
            
            # Load from disk if not in cache
            return await self._load_market_data_from_disk(symbol, start_date, end_date)
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    async def _load_market_data_from_disk(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load market data from disk files.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data range
            end_date: End date for data range
            
        Returns:
            Optional[pd.DataFrame]: Market data or None
        """
        try:
            market_data_dir = self.storage_path / "market_data"
            
            # Find relevant files
            pattern = f"{symbol}_*.json.gz"
            files = list(market_data_dir.glob(pattern))
            
            if not files:
                logger.debug(f"No market data files found for {symbol}")
                return None
            
            # Load and combine data from files
            all_data = []
            for file_path in sorted(files):
                try:
                    with gzip.open(file_path, 'rt') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            all_data.extend(file_data)
                        else:
                            all_data.append(file_data)
                except Exception as e:
                    logger.warning(f"Failed to load data from {file_path}: {e}")
            
            if not all_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Filter by date range if specified
            if start_date or end_date:
                df = self._filter_by_date_range(df, start_date, end_date)
            
            logger.debug(f"Loaded {len(df)} market data records from disk for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load market data from disk for {symbol}: {e}")
            return None
    
    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame with timestamp column
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if 'timestamp' not in df.columns:
            return df
        
        filtered_df = df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['timestamp'] >= start_date]
        
        if end_date:
            filtered_df = filtered_df[filtered_df['timestamp'] <= end_date]
        
        return filtered_df
    
    async def store_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Store trade execution data to database.
        
        Args:
            trade_data: Trade data to store
            
        Returns:
            bool: True if successful
        """
        try:
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
                    fill_time=trade_data.get("fill_time", datetime.now(timezone.utc)),
                    metadata=trade_data.get("metadata", {})
                )
                session.add(trade)
                session.commit()
            
            logger.info(f"Trade stored: {trade_data['symbol']} {trade_data['side']} {trade_data['quantity']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
            return False
    
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade records from database.
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy name
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records
            
        Returns:
            List[Dict[str, Any]]: Trade records
        """
        try:
            with db_manager.get_session() as session:
                query = session.query(Trade)
                
                # Apply filters
                if symbol:
                    query = query.filter(Trade.symbol == symbol)
                
                if strategy:
                    query = query.filter(Trade.strategy_name == strategy)
                
                if start_date:
                    query = query.filter(Trade.fill_time >= start_date)
                
                if end_date:
                    query = query.filter(Trade.fill_time <= end_date)
                
                # Order and limit
                query = query.order_by(Trade.fill_time.desc()).limit(limit)
                
                trades = query.all()
                
                # Convert to dictionaries
                trade_data = []
                for trade in trades:
                    trade_dict = {
                        "id": str(trade.id),
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "price": trade.price,
                        "commission": trade.commission,
                        "order_id": trade.order_id,
                        "execution_id": trade.execution_id,
                        "strategy_name": trade.strategy_name,
                        "fill_time": trade.fill_time.isoformat(),
                        "created_at": trade.created_at.isoformat(),
                        "metadata": trade.metadata
                    }
                    trade_data.append(trade_dict)
                
                logger.debug(f"Retrieved {len(trade_data)} trade records")
                return trade_data
                
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old data files and database records.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dict[str, int]: Cleanup statistics
        """
        cleanup_stats = {
            "files_archived": 0,
            "files_deleted": 0,
            "db_records_deleted": 0,
            "cache_keys_deleted": 0
        }
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up market data files
            market_data_dir = self.storage_path / "market_data"
            
            for file_path in market_data_dir.glob("*.json.gz"):
                # Extract date from filename
                try:
                    date_str = file_path.stem.split('_')[-1].replace('.json', '')
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if file_date < cutoff_date:
                        # Archive old files instead of deleting
                        archive_path = self.storage_path / "archive" / file_path.name
                        file_path.rename(archive_path)
                        cleanup_stats["files_archived"] += 1
                        
                except (ValueError, IndexError):
                    # Skip files that don't match expected naming pattern
                    continue
            
            # Clean up very old archived files (keep archives for 1 year)
            archive_cutoff = datetime.now() - timedelta(days=365)
            archive_dir = self.storage_path / "archive"
            
            for file_path in archive_dir.glob("*.json.gz"):
                try:
                    if file_path.stat().st_mtime < archive_cutoff.timestamp():
                        file_path.unlink()
                        cleanup_stats["files_deleted"] += 1
                except OSError:
                    continue
            
            logger.info(
                f"Data cleanup completed: {cleanup_stats['files_archived']} archived, "
                f"{cleanup_stats['files_deleted']} deleted"
            )
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            return cleanup_stats
    
    async def backup_database(self, backup_name: Optional[str] = None) -> bool:
        """
        Create a backup of critical database tables.
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            bool: True if backup successful
        """
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = self.storage_path / "backups" / f"{backup_name}.json.gz"
            
            # Export data from key tables
            backup_data = {}
            
            with db_manager.get_session() as session:
                # Export trades
                trades = session.query(Trade).all()
                backup_data["trades"] = [
                    {
                        "symbol": t.symbol,
                        "side": t.side,
                        "quantity": t.quantity,
                        "price": t.price,
                        "commission": t.commission,
                        "fill_time": t.fill_time.isoformat(),
                        "strategy_name": t.strategy_name,
                        "metadata": t.metadata
                    }
                    for t in trades
                ]
                
                # Export orders
                orders = session.query(Order).all()
                backup_data["orders"] = [
                    {
                        "order_id": o.order_id,
                        "symbol": o.symbol,
                        "side": o.side,
                        "order_type": o.order_type,
                        "quantity": o.quantity,
                        "price": o.price,
                        "status": o.status,
                        "created_at": o.created_at.isoformat(),
                        "metadata": o.metadata
                    }
                    for o in orders
                ]
            
            # Add backup metadata
            backup_data["metadata"] = {
                "backup_name": backup_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": "3.0.0",
                "record_counts": {
                    "trades": len(backup_data["trades"]),
                    "orders": len(backup_data["orders"])
                }
            }
            
            # Write compressed backup
            with gzip.open(backup_path, 'wt') as f:
                json.dump(backup_data, f, default=str)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage usage and statistics.
        
        Returns:
            Dict[str, Any]: Storage information
        """
        try:
            storage_info = {
                "storage_path": str(self.storage_path),
                "total_size_mb": 0.0,
                "directories": {}
            }
            
            # Calculate directory sizes
            for subdir in ["market_data", "signals", "performance", "backups", "archive"]:
                dir_path = self.storage_path / subdir
                
                if dir_path.exists():
                    total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                    file_count = len(list(dir_path.rglob('*')))
                    
                    storage_info["directories"][subdir] = {
                        "size_mb": total_size / (1024 * 1024),
                        "file_count": file_count
                    }
                    
                    storage_info["total_size_mb"] += total_size / (1024 * 1024)
            
            return storage_info
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {"error": str(e)}


# Global storage instance
data_storage = DataStorage()
