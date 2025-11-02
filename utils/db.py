"""
Helios Database Management v3

SQLAlchemy-based database layer with ORM models, migrations,
and connection management for PostgreSQL.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean,
    Text, JSON, ForeignKey, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import UUID
import uuid

from config.settings import get_settings, get_database_config
from utils.logger import get_logger

logger = get_logger("database")

# Base class for all ORM models
Base = declarative_base()


class TimestampMixin:
    """Mixin for automatic timestamp tracking."""
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


class Trade(Base, TimestampMixin):
    """Trade execution record."""
    __tablename__ = "trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    order_id = Column(String(100), index=True)
    execution_id = Column(String(100), unique=True)
    strategy_name = Column(String(100), index=True)
    fill_time = Column(DateTime(timezone=True))
    metadata = Column(JSON, default=dict)
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_trades_symbol_created', 'symbol', 'created_at'),
        Index('ix_trades_strategy_created', 'strategy_name', 'created_at'),
    )


class Order(Base, TimestampMixin):
    """Order tracking record."""
    __tablename__ = "orders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    order_id = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # BUY, SELL
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP
    quantity = Column(Float, nullable=False)
    price = Column(Float)
    stop_price = Column(Float)
    time_in_force = Column(String(10), default="DAY")
    status = Column(String(20), nullable=False, index=True)  # PENDING, FILLED, CANCELLED, REJECTED
    filled_quantity = Column(Float, default=0.0)
    average_fill_price = Column(Float)
    strategy_name = Column(String(100), index=True)
    broker = Column(String(50), default="paper")
    metadata = Column(JSON, default=dict)
    
    # Relationship to trades
    trades = relationship("Trade", backref="order", foreign_keys="Trade.order_id", primaryjoin="Order.order_id == Trade.order_id")
    
    __table_args__ = (
        Index('ix_orders_symbol_status', 'symbol', 'status'),
        Index('ix_orders_strategy_status', 'strategy_name', 'status'),
    )


class Position(Base, TimestampMixin):
    """Current position tracking."""
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, unique=True)
    quantity = Column(Float, nullable=False)
    average_price = Column(Float, nullable=False)
    market_value = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    strategy_name = Column(String(100), index=True)
    last_price = Column(Float)
    metadata = Column(JSON, default=dict)
    
    __table_args__ = (
        Index('ix_positions_symbol', 'symbol'),
        Index('ix_positions_strategy', 'strategy_name'),
    )


class PerformanceSnapshot(Base, TimestampMixin):
    """Daily performance snapshots."""
    __tablename__ = "performance_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    daily_return = Column(Float, default=0.0)
    total_return = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    volatility = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    strategy_name = Column(String(100), index=True)
    metadata = Column(JSON, default=dict)
    
    __table_args__ = (
        Index('ix_performance_date_strategy', 'date', 'strategy_name'),
        UniqueConstraint('date', 'strategy_name', name='uq_performance_date_strategy'),
    )


class SentinelRepair(Base, TimestampMixin):
    """System Sentinel repair log."""
    __tablename__ = "sentinel_repairs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repair_type = Column(String(50), nullable=False, index=True)
    component = Column(String(100), nullable=False)
    issue_description = Column(Text, nullable=False)
    repair_action = Column(Text, nullable=False)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    execution_time_seconds = Column(Float)
    metadata = Column(JSON, default=dict)
    
    __table_args__ = (
        Index('ix_sentinel_repairs_type_created', 'repair_type', 'created_at'),
        Index('ix_sentinel_repairs_component_created', 'component', 'created_at'),
    )


class RiskMetric(Base, TimestampMixin):
    """Risk management metrics."""
    __tablename__ = "risk_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    portfolio_var = Column(Float)  # Value at Risk
    portfolio_cvar = Column(Float)  # Conditional VaR
    beta = Column(Float)
    correlation_risk = Column(Float)
    concentration_risk = Column(Float)
    liquidity_risk = Column(Float)
    leverage_ratio = Column(Float)
    risk_score = Column(Float)
    metadata = Column(JSON, default=dict)
    
    __table_args__ = (
        Index('ix_risk_metrics_date', 'date'),
    )


class DatabaseManager:
    """
    Database connection and session management.
    
    Provides connection pooling, session management, and migration utilities.
    """
    
    def __init__(self):
        """Initialize database manager."""
        self.settings = get_settings()
        self.db_config = get_database_config()
        self._engine = None
        self._session_factory = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize database connection and create tables.
        """
        if self._initialized:
            logger.warning("Database already initialized")
            return
        
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.db_config["url"],
                pool_size=self.db_config["pool_size"],
                max_overflow=self.db_config["max_overflow"],
                echo=self.db_config["echo"],
                pool_timeout=self.db_config["pool_timeout"],
                pool_recycle=self.db_config["pool_recycle"],
                poolclass=QueuePool,
                pool_pre_ping=True,  # Validate connections before use
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
            
            # Test connection
            self.test_connection()
            
            # Create tables if they don't exist
            self.create_tables()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            bool: True if connection successful
        """
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            logger.debug("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def create_tables(self) -> None:
        """
        Create all tables if they don't exist.
        """
        try:
            Base.metadata.create_all(self._engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """
        Drop all tables. Use with caution!
        """
        try:
            Base.metadata.drop_all(self._engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup.
        
        Yields:
            Session: SQLAlchemy session
        """
        if not self._initialized:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_info(self) -> Dict[str, Any]:
        """
        Get database table information.
        
        Returns:
            Dict[str, Any]: Table information
        """
        tables_info = {}
        
        try:
            with self.get_session() as session:
                for table_name in Base.metadata.tables.keys():
                    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                    result = session.execute(text(count_query))
                    count = result.fetchone()[0]
                    tables_info[table_name] = {"row_count": count}
                    
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            
        return tables_info
    
    def cleanup(self) -> None:
        """
        Clean up database connections.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections cleaned up")


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_session():
    """Get database session context manager."""
    return db_manager.get_session()


def initialize_database():
    """Initialize database connection and tables."""
    db_manager.initialize()


def test_database_connection() -> bool:
    """Test database connectivity."""
    return db_manager.test_connection()


def get_database_info() -> Dict[str, Any]:
    """Get database information."""
    return db_manager.get_table_info()
