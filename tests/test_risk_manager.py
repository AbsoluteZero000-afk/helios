"""
Helios Risk Manager Tests v3

Comprehensive tests for risk management functionality including
position limits, portfolio constraints, and VAR calculations.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from risk.manager import RiskManager, Position
from config.settings import Settings


class TestRiskManager:
    """Test suite for RiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create RiskManager instance for testing."""
        return RiskManager()
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.max_portfolio_risk = 0.02
        settings.max_open_positions = 5
        settings.risk_position_limit = 0.05
        settings.risk_correlation_limit = 0.3
        return settings
    
    def test_max_position_value_calculation(self, risk_manager):
        """Test maximum position value calculation."""
        portfolio_value = 100000.0
        expected_max_value = portfolio_value * risk_manager.settings.max_portfolio_risk
        
        actual_max_value = risk_manager.max_position_value(portfolio_value)
        
        assert actual_max_value == expected_max_value
    
    def test_allow_trade_within_limits(self, risk_manager):
        """Test trade approval when within risk limits."""
        symbol = "AAPL"
        quantity = 50.0
        price = 150.0
        portfolio_value = 100000.0
        open_positions = {}
        
        result = risk_manager.allow_trade(symbol, quantity, price, portfolio_value, open_positions)
        
        assert result is True
    
    def test_reject_trade_exceeding_position_value(self, risk_manager):
        """Test trade rejection when position value exceeds limit."""
        symbol = "AAPL"
        quantity = 200.0  # Large quantity
        price = 150.0     # $30,000 position (exceeds 2% of $100k portfolio)
        portfolio_value = 100000.0
        open_positions = {}
        
        result = risk_manager.allow_trade(symbol, quantity, price, portfolio_value, open_positions)
        
        assert result is False
    
    def test_reject_trade_exceeding_max_positions(self, risk_manager):
        """Test trade rejection when max open positions exceeded."""
        symbol = "AAPL"
        quantity = 10.0
        price = 150.0
        portfolio_value = 100000.0
        
        # Create positions at the maximum limit
        open_positions = {
            f"STOCK{i}": Position(symbol=f"STOCK{i}", quantity=10.0, avg_price=100.0)
            for i in range(risk_manager.settings.max_open_positions)
        }
        
        result = risk_manager.allow_trade(symbol, quantity, price, portfolio_value, open_positions)
        
        assert result is False
    
    def test_allow_trade_for_existing_position(self, risk_manager):
        """Test trade approval for existing position (doesn't count against max positions)."""
        symbol = "AAPL"
        quantity = 10.0
        price = 150.0
        portfolio_value = 100000.0
        
        # Create positions at maximum limit, including the target symbol
        open_positions = {
            symbol: Position(symbol=symbol, quantity=50.0, avg_price=140.0)
        }
        open_positions.update({
            f"STOCK{i}": Position(symbol=f"STOCK{i}", quantity=10.0, avg_price=100.0)
            for i in range(risk_manager.settings.max_open_positions - 1)
        })
        
        result = risk_manager.allow_trade(symbol, quantity, price, portfolio_value, open_positions)
        
        assert result is True
    
    @patch('risk.manager.get_settings')
    def test_risk_manager_with_custom_settings(self, mock_get_settings, mock_settings):
        """Test RiskManager with custom settings."""
        mock_get_settings.return_value = mock_settings
        
        risk_manager = RiskManager()
        portfolio_value = 50000.0
        
        expected_max_value = portfolio_value * mock_settings.max_portfolio_risk
        actual_max_value = risk_manager.max_position_value(portfolio_value)
        
        assert actual_max_value == expected_max_value
    
    def test_portfolio_risk_assessment(self, risk_manager):
        """Test portfolio-level risk assessment."""
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=100.0, avg_price=150.0),
            "MSFT": Position(symbol="MSFT", quantity=50.0, avg_price=200.0),
            "GOOGL": Position(symbol="GOOGL", quantity=25.0, avg_price=120.0),
        }
        
        portfolio_value = 100000.0
        
        risk_assessment = risk_manager.assess_portfolio_risk(positions, portfolio_value)
        
        assert "total_exposure" in risk_assessment
        assert "position_count" in risk_assessment
        assert "largest_position_pct" in risk_assessment
        assert risk_assessment["position_count"] == len(positions)
    
    def test_position_sizing_recommendation(self, risk_manager):
        """Test position sizing recommendations."""
        symbol = "AAPL"
        current_price = 150.0
        portfolio_value = 100000.0
        volatility = 0.25  # 25% annual volatility
        
        recommended_size = risk_manager.calculate_position_size(
            symbol, current_price, portfolio_value, volatility
        )
        
        # Should not exceed maximum position limit
        max_position_value = portfolio_value * risk_manager.settings.risk_position_limit
        max_shares = max_position_value / current_price
        
        assert recommended_size <= max_shares
        assert recommended_size > 0
    
    def test_stop_loss_calculation(self, risk_manager):
        """Test stop loss calculation based on ATR and volatility."""
        entry_price = 100.0
        atr = 2.5
        volatility = 0.20
        
        stop_loss = risk_manager.calculate_stop_loss(entry_price, atr, volatility)
        
        # Stop loss should be below entry price for long position
        assert stop_loss < entry_price
        # Should be reasonable percentage
        stop_loss_pct = (entry_price - stop_loss) / entry_price
        assert 0.01 <= stop_loss_pct <= 0.10  # Between 1% and 10%
