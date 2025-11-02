"""
Helios Momentum Breakout Strategy v3

Implements a momentum-based breakout strategy using price and volume analysis.
Triggers on significant price moves with above-average volume confirmation.
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from strategies.base import (
    TechnicalAnalysisStrategy, StrategyConfig, SignalEvent, SignalType
)
from utils.logger import strategy_logger as logger


class MomentumBreakoutStrategy(TechnicalAnalysisStrategy):
    """
    Momentum breakout strategy based on price and volume analysis.
    
    Strategy Logic:
    1. Identify price breakouts above/below recent high/low
    2. Confirm with above-average volume
    3. Check for sustained momentum over confirmation period
    4. Generate BUY/SELL signals with appropriate stop losses
    """
    
    async def initialize(self) -> None:
        """
        Initialize momentum breakout strategy parameters.
        """
        params = self.config.parameters
        
        # Breakout detection parameters
        self.breakout_period = int(params.get('breakout_period', 20))
        self.volume_threshold = float(params.get('volume_threshold', 1.5))
        self.price_change_threshold = float(params.get('price_change_threshold', 0.05))
        self.confirmation_period = int(params.get('confirmation_period', 3))
        
        # Signal generation parameters
        self.signal_threshold = float(params.get('signal_threshold', 0.8))
        self.min_data_length = int(params.get('min_data_length', 100))
        
        # Risk management parameters
        self.stop_loss_pct = float(params.get('stop_loss_pct', 0.03))
        self.take_profit_pct = float(params.get('take_profit_pct', 0.10))
        self.max_holding_days = int(params.get('max_holding_days', 15))
        
        logger.info(
            f"MomentumBreakout initialized: breakout_period={self.breakout_period}, "
            f"volume_threshold={self.volume_threshold}, price_threshold={self.price_change_threshold}"
        )
    
    async def cleanup(self) -> None:
        """
        Cleanup momentum breakout strategy resources.
        """
        # Clear any cached calculations
        for symbol in self.config.symbols:
            if symbol in self.indicators:
                self.indicators[symbol].clear()
        
        logger.info("MomentumBreakout strategy cleaned up")
    
    async def update_indicators(self, symbol: str) -> None:
        """
        Update technical indicators for momentum analysis.
        
        Args:
            symbol: Symbol to update indicators for
        """
        df = self.market_data.get(symbol)
        if df is None or df.empty or len(df) < self.min_data_length:
            return
        
        # Initialize indicators dict for symbol
        if symbol not in self.indicators:
            self.indicators[symbol] = {}
        
        indicators = self.indicators[symbol]
        
        # Calculate price-based indicators
        indicators['high_breakout'] = df['high'].rolling(window=self.breakout_period).max()
        indicators['low_breakout'] = df['low'].rolling(window=self.breakout_period).min()
        
        # Volume indicators
        indicators['volume_ma'] = df['volume'].rolling(window=self.breakout_period).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_ma']
        
        # Price change indicators
        indicators['price_change'] = df['close'].pct_change()
        indicators['price_change_ma'] = indicators['price_change'].rolling(window=5).mean()
        
        # Momentum indicators
        indicators['momentum'] = df['close'] / df['close'].shift(self.confirmation_period) - 1
        indicators['momentum_ma'] = indicators['momentum'].rolling(window=5).mean()
        
        # Volatility indicators
        indicators['volatility'] = indicators['price_change'].rolling(window=20).std()
        indicators['atr'] = self._calculate_atr(df)
        
        # Support and resistance levels
        indicators['resistance'] = df['high'].rolling(window=self.breakout_period).max()
        indicators['support'] = df['low'].rolling(window=self.breakout_period).min()
        
        logger.debug(f"Updated momentum indicators for {symbol}")
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: OHLCV DataFrame
            period: ATR calculation period
            
        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    async def generate_signal(self, symbol: str) -> Optional[SignalEvent]:
        """
        Generate momentum breakout signals.
        
        Args:
            symbol: Symbol to generate signal for
            
        Returns:
            Optional[SignalEvent]: Generated signal or None
        """
        indicators = self.indicators.get(symbol, {})
        df = self.market_data.get(symbol)
        
        if (not indicators or df is None or df.empty or 
            len(df) < self.min_data_length):
            return None
        
        # Get latest values
        try:
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            previous_close = df['close'].iloc[-2]
            
            high_breakout = indicators['high_breakout'].iloc[-2]  # Previous breakout level
            low_breakout = indicators['low_breakout'].iloc[-2]   # Previous breakout level
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            momentum = indicators['momentum'].iloc[-1]
            atr = indicators['atr'].iloc[-1]
            
            # Check for breakout conditions
            upward_breakout = self._check_upward_breakout(
                current_price, previous_close, high_breakout, volume_ratio, momentum
            )
            
            downward_breakout = self._check_downward_breakout(
                current_price, previous_close, low_breakout, volume_ratio, momentum
            )
            
            # Generate signals based on breakouts
            if upward_breakout:
                signal_strength = min(self._calculate_signal_strength(
                    current_price, high_breakout, volume_ratio, momentum, "upward"
                ), 1.0)
                
                if signal_strength >= self.signal_threshold:
                    # Calculate stop loss and take profit
                    stop_loss = current_price * (1 - self.stop_loss_pct)
                    take_profit = current_price * (1 + self.take_profit_pct)
                    
                    logger.info(
                        f"Upward breakout signal for {symbol}: price={current_price:.2f}, "
                        f"breakout_level={high_breakout:.2f}, volume_ratio={volume_ratio:.2f}"
                    )
                    
                    return SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.BUY.value,
                        strength=signal_strength,
                        strategy_name=self.config.name,
                        target_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            "breakout_type": "upward",
                            "breakout_level": high_breakout,
                            "volume_ratio": volume_ratio,
                            "momentum": momentum,
                            "atr": atr
                        }
                    )
            
            elif downward_breakout:
                signal_strength = min(self._calculate_signal_strength(
                    current_price, low_breakout, volume_ratio, momentum, "downward"
                ), 1.0)
                
                if signal_strength >= self.signal_threshold:
                    # Calculate stop loss and take profit for short
                    stop_loss = current_price * (1 + self.stop_loss_pct)
                    take_profit = current_price * (1 - self.take_profit_pct)
                    
                    logger.info(
                        f"Downward breakout signal for {symbol}: price={current_price:.2f}, "
                        f"breakout_level={low_breakout:.2f}, volume_ratio={volume_ratio:.2f}"
                    )
                    
                    return SignalEvent(
                        symbol=symbol,
                        signal_type=SignalType.SELL.value,
                        strength=signal_strength,
                        strategy_name=self.config.name,
                        target_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={
                            "breakout_type": "downward",
                            "breakout_level": low_breakout,
                            "volume_ratio": volume_ratio,
                            "momentum": momentum,
                            "atr": atr
                        }
                    )
            
            return None
            
        except (IndexError, KeyError) as e:
            logger.warning(f"Insufficient data for momentum signal generation in {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")
            return None
    
    def _check_upward_breakout(
        self,
        current_price: float,
        previous_close: float,
        high_breakout: float,
        volume_ratio: float,
        momentum: float
    ) -> bool:
        """
        Check for upward breakout conditions.
        
        Args:
            current_price: Current closing price
            previous_close: Previous closing price
            high_breakout: Breakout resistance level
            volume_ratio: Current volume vs average ratio
            momentum: Price momentum indicator
            
        Returns:
            bool: True if upward breakout detected
        """
        # Price must break above resistance level
        price_breakout = current_price > high_breakout
        
        # Volume must be above threshold
        volume_confirmation = volume_ratio > self.volume_threshold
        
        # Price change must be significant
        price_change = (current_price - previous_close) / previous_close
        significant_move = price_change > self.price_change_threshold
        
        # Positive momentum
        positive_momentum = momentum > 0
        
        return price_breakout and volume_confirmation and significant_move and positive_momentum
    
    def _check_downward_breakout(
        self,
        current_price: float,
        previous_close: float,
        low_breakout: float,
        volume_ratio: float,
        momentum: float
    ) -> bool:
        """
        Check for downward breakout conditions.
        
        Args:
            current_price: Current closing price
            previous_close: Previous closing price
            low_breakout: Breakout support level
            volume_ratio: Current volume vs average ratio
            momentum: Price momentum indicator
            
        Returns:
            bool: True if downward breakout detected
        """
        # Price must break below support level
        price_breakout = current_price < low_breakout
        
        # Volume must be above threshold
        volume_confirmation = volume_ratio > self.volume_threshold
        
        # Price change must be significant (negative)
        price_change = (current_price - previous_close) / previous_close
        significant_move = price_change < -self.price_change_threshold
        
        # Negative momentum
        negative_momentum = momentum < 0
        
        return price_breakout and volume_confirmation and significant_move and negative_momentum
    
    def _calculate_signal_strength(
        self,
        current_price: float,
        breakout_level: float,
        volume_ratio: float,
        momentum: float,
        direction: str
    ) -> float:
        """
        Calculate signal strength based on breakout characteristics.
        
        Args:
            current_price: Current price
            breakout_level: Breakout level (support/resistance)
            volume_ratio: Volume confirmation ratio
            momentum: Price momentum
            direction: Breakout direction ('upward' or 'downward')
            
        Returns:
            float: Signal strength (0.0 to 1.0)
        """
        # Base strength from price distance from breakout level
        if direction == "upward":
            price_strength = min((current_price - breakout_level) / breakout_level, 0.3)
        else:
            price_strength = min((breakout_level - current_price) / breakout_level, 0.3)
        
        # Volume confirmation strength
        volume_strength = min((volume_ratio - 1.0) * 0.2, 0.3)  # Cap at 0.3
        
        # Momentum strength
        momentum_strength = min(abs(momentum) * 2.0, 0.4)  # Cap at 0.4
        
        # Combined strength
        total_strength = price_strength + volume_strength + momentum_strength
        
        # Normalize to 0-1 range
        normalized_strength = min(max(total_strength, 0.0), 1.0)
        
        logger.debug(
            f"Signal strength calculation for {direction} breakout: "
            f"price={price_strength:.3f}, volume={volume_strength:.3f}, "
            f"momentum={momentum_strength:.3f}, total={normalized_strength:.3f}"
        )
        
        return normalized_strength
    
    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        current_price: float
    ) -> float:
        """
        Calculate position size for momentum breakout strategy.
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0.0 to 1.0)
            current_price: Current market price
            
        Returns:
            float: Position size
        """
        # Get position sizing configuration
        base_size = self.config.position_sizing.get('base_size', 0.01)
        max_size = self.config.position_sizing.get('max_size', 0.03)
        scale_with_signal = self.config.position_sizing.get('scale_with_signal', False)
        
        if scale_with_signal:
            # Scale position size with signal strength
            position_size = base_size + (max_size - base_size) * signal_strength
        else:
            # Use fixed base size
            position_size = base_size
        
        # Apply portfolio value to get dollar amount
        # This would typically be calculated based on current portfolio value
        portfolio_value = 100000.0  # Placeholder - would get from portfolio manager
        dollar_amount = portfolio_value * position_size
        
        # Convert to share quantity
        share_quantity = dollar_amount / current_price
        
        logger.debug(
            f"Position sizing for {symbol}: {position_size:.3f} of portfolio "
            f"(${dollar_amount:.2f}, {share_quantity:.0f} shares)"
        )
        
        return share_quantity
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """
        Get momentum strategy-specific metrics.
        
        Returns:
            Dict[str, Any]: Strategy metrics
        """
        base_metrics = self.get_metrics()
        
        # Calculate momentum-specific metrics
        if self.signals_history:
            upward_signals = sum(1 for s in self.signals_history if s.signal_type == SignalType.BUY.value)
            downward_signals = sum(1 for s in self.signals_history if s.signal_type == SignalType.SELL.value)
            
            avg_signal_strength = sum(s.strength for s in self.signals_history) / len(self.signals_history)
            
            momentum_metrics = {
                "upward_breakouts": upward_signals,
                "downward_breakouts": downward_signals,
                "average_signal_strength": avg_signal_strength,
                "breakout_success_rate": self.metrics.win_rate,
            }
        else:
            momentum_metrics = {
                "upward_breakouts": 0,
                "downward_breakouts": 0,
                "average_signal_strength": 0.0,
                "breakout_success_rate": 0.0,
            }
        
        return {
            **base_metrics.__dict__,
            "strategy_specific": momentum_metrics,
            "parameters": {
                "breakout_period": self.breakout_period,
                "volume_threshold": self.volume_threshold,
                "price_change_threshold": self.price_change_threshold,
                "confirmation_period": self.confirmation_period,
                "signal_threshold": self.signal_threshold,
            }
        }
    
    def is_breakout_valid(self, symbol: str, breakout_type: str) -> bool:
        """
        Validate if a breakout signal is still valid.
        
        Args:
            symbol: Symbol to check
            breakout_type: 'upward' or 'downward'
            
        Returns:
            bool: True if breakout is still valid
        """
        df = self.market_data.get(symbol)
        if df is None or df.empty:
            return False
        
        # Check if recent price action confirms the breakout
        recent_prices = df['close'].tail(self.confirmation_period)
        
        if breakout_type == "upward":
            # For upward breakout, recent prices should generally trend higher
            return recent_prices.iloc[-1] >= recent_prices.iloc[0]
        else:
            # For downward breakout, recent prices should generally trend lower
            return recent_prices.iloc[-1] <= recent_prices.iloc[0]
