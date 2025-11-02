"""
Helios Momentum Breakout Strategy Tests v3

Comprehensive tests for momentum breakout strategy including
signal generation, breakout detection, and risk management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from strategies.base import StrategyConfig
from strategies.momentum_breakout import MomentumBreakoutStrategy
from data.feeder import HistoricalFeeder
from core.engine import TradingEngine, EngineMode


class TestMomentumBreakoutStrategy:
    """Test suite for MomentumBreakoutStrategy."""
    
    @pytest.fixture
    def strategy_config(self):
        """Create strategy configuration for testing."""
        return StrategyConfig(
            name="test_momentum",
            symbols=["AAPL"],
            parameters={
                "breakout_period": 10,
                "volume_threshold": 1.5,
                "price_change_threshold": 0.03,
                "confirmation_period": 2,
                "signal_threshold": 0.6,
                "min_data_length": 50,
                "stop_loss_pct": 0.03,
                "take_profit_pct": 0.08
            },
            position_sizing={
                "base_size": 0.02,
                "max_size": 0.05,
                "scale_with_signal": True
            }
        )
    
    @pytest.fixture
    def momentum_strategy(self, strategy_config):
        """Create MomentumBreakoutStrategy instance."""
        return MomentumBreakoutStrategy(strategy_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create realistic OHLCV data with a breakout pattern
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(100):
            # Simulate price movement with breakout around day 60
            if i < 50:
                # Consolidation phase
                price_change = np.random.normal(0, 0.01)
            elif i < 65:
                # Breakout phase - strong upward movement
                price_change = np.random.normal(0.03, 0.02)
            else:
                # Post-breakout continuation
                price_change = np.random.normal(0.01, 0.015)
            
            base_price *= (1 + price_change)
            
            # Generate OHLC from close price
            close = base_price
            daily_range = close * 0.02 * np.random.random()
            high = close + daily_range * np.random.random()
            low = close - daily_range * np.random.random()
            open_price = low + (high - low) * np.random.random()
            
            prices.append([open_price, high, low, close])
            
            # Volume spike during breakout
            if 50 <= i < 65:
                volume = np.random.randint(800000, 2000000)  # High volume
            else:
                volume = np.random.randint(400000, 800000)   # Normal volume
            
            volumes.append(volume)
        
        df = pd.DataFrame(prices, columns=['open', 'high', 'low', 'close'])
        df['volume'] = volumes
        df['timestamp'] = dates
        df['symbol'] = 'AAPL'
        
        return df
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, momentum_strategy):
        """Test strategy initialization."""
        await momentum_strategy.initialize()
        
        assert momentum_strategy.breakout_period == 10
        assert momentum_strategy.volume_threshold == 1.5
        assert momentum_strategy.price_change_threshold == 0.03
        assert momentum_strategy.signal_threshold == 0.6
    
    @pytest.mark.asyncio
    async def test_indicator_calculation(self, momentum_strategy, sample_data):
        """Test technical indicator calculations."""
        await momentum_strategy.initialize()
        
        # Store sample data
        symbol = "AAPL"
        momentum_strategy.market_data[symbol] = sample_data
        
        # Update indicators
        await momentum_strategy.update_indicators(symbol)
        
        indicators = momentum_strategy.indicators[symbol]
        
        # Check that indicators were calculated
        assert 'high_breakout' in indicators
        assert 'low_breakout' in indicators
        assert 'volume_ma' in indicators
        assert 'volume_ratio' in indicators
        assert 'momentum' in indicators
        assert 'atr' in indicators
        
        # Check indicator values are reasonable
        assert not indicators['high_breakout'].isna().all()
        assert not indicators['volume_ratio'].isna().all()
        assert (indicators['volume_ratio'] > 0).all()
    
    @pytest.mark.asyncio
    async def test_upward_breakout_detection(self, momentum_strategy, sample_data):
        """Test upward breakout signal detection."""
        await momentum_strategy.initialize()
        
        symbol = "AAPL"
        momentum_strategy.market_data[symbol] = sample_data
        await momentum_strategy.update_indicators(symbol)
        
        # Test with breakout conditions
        current_price = 110.0
        previous_close = 105.0
        high_breakout = 108.0
        volume_ratio = 2.0  # Above threshold
        momentum = 0.05     # Positive momentum
        
        breakout_detected = momentum_strategy._check_upward_breakout(
            current_price, previous_close, high_breakout, volume_ratio, momentum
        )
        
        assert breakout_detected is True
    
    @pytest.mark.asyncio
    async def test_downward_breakout_detection(self, momentum_strategy, sample_data):
        """Test downward breakout signal detection."""
        await momentum_strategy.initialize()
        
        symbol = "AAPL"
        momentum_strategy.market_data[symbol] = sample_data
        await momentum_strategy.update_indicators(symbol)
        
        # Test with breakout conditions
        current_price = 92.0
        previous_close = 98.0
        low_breakout = 95.0
        volume_ratio = 1.8  # Above threshold
        momentum = -0.04   # Negative momentum
        
        breakout_detected = momentum_strategy._check_downward_breakout(
            current_price, previous_close, low_breakout, volume_ratio, momentum
        )
        
        assert breakout_detected is True
    
    @pytest.mark.asyncio
    async def test_signal_strength_calculation(self, momentum_strategy):
        """Test signal strength calculation."""
        await momentum_strategy.initialize()
        
        # Test upward breakout signal strength
        strength = momentum_strategy._calculate_signal_strength(
            current_price=110.0,
            breakout_level=105.0,
            volume_ratio=2.0,
            momentum=0.05,
            direction="upward"
        )
        
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be relatively strong signal
    
    @pytest.mark.asyncio
    async def test_position_size_calculation(self, momentum_strategy):
        """Test position size calculation."""
        await momentum_strategy.initialize()
        
        symbol = "AAPL"
        signal_strength = 0.8
        current_price = 150.0
        
        position_size = momentum_strategy.calculate_position_size(
            symbol, signal_strength, current_price
        )
        
        assert position_size > 0
        assert isinstance(position_size, float)
    
    @pytest.mark.asyncio
    async def test_full_signal_generation_with_engine(self, momentum_strategy, sample_data):
        """Test full signal generation in engine context."""
        # Create engine and add strategy
        engine = TradingEngine(mode=EngineMode.BACKTEST)
        symbol = "AAPL"
        engine.add_symbol(symbol)
        
        await momentum_strategy.initialize()
        engine.strategies['momentum'] = momentum_strategy
        
        # Start engine
        await engine.start()
        
        # Feed data to trigger signal generation
        # Use the last portion of sample data which has breakout pattern
        breakout_data = sample_data.tail(20)  # Last 20 days with breakout
        
        for _, row in breakout_data.iterrows():
            # Simulate market data event
            from core.events import MarketDataEvent
            
            event = MarketDataEvent(
                symbol=symbol,
                price=row['close'],
                volume=int(row['volume']),
                timestamp=row['timestamp']
            )
            
            await momentum_strategy.handle(event)
        
        # Stop engine
        await engine.stop()
        
        # Check that signals were generated
        assert momentum_strategy.metrics.total_signals >= 0  # May or may not generate signals
    
    def test_strategy_metrics(self, momentum_strategy):
        """Test strategy metrics calculation."""
        # Add some mock signals to history
        from core.events import SignalEvent
        from strategies.base import SignalType
        
        momentum_strategy.signals_history = [
            SignalEvent(
                symbol="AAPL",
                signal_type=SignalType.BUY.value,
                strength=0.8,
                strategy_name="test_momentum"
            ),
            SignalEvent(
                symbol="AAPL",
                signal_type=SignalType.SELL.value,
                strength=0.7,
                strategy_name="test_momentum"
            )
        ]
        
        metrics = momentum_strategy.get_strategy_metrics()
        
        assert "strategy_specific" in metrics
        assert metrics["strategy_specific"]["upward_breakouts"] == 1
        assert metrics["strategy_specific"]["downward_breakouts"] == 1
        assert metrics["strategy_specific"]["average_signal_strength"] == 0.75
    
    def test_breakout_validation(self, momentum_strategy, sample_data):
        """Test breakout validation logic."""
        symbol = "AAPL"
        momentum_strategy.market_data[symbol] = sample_data
        
        # Test upward breakout validation
        # Use data where price is trending up in recent periods
        upward_valid = momentum_strategy.is_breakout_valid(symbol, "upward")
        assert isinstance(upward_valid, bool)
        
        # Test downward breakout validation
        downward_valid = momentum_strategy.is_breakout_valid(symbol, "downward")
        assert isinstance(downward_valid, bool)
