"""
Helios Data Processing v3

Advanced data processing utilities for market data cleaning,
transformation, and technical indicator computation using TA-Lib exclusively.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger("data_processor")


@dataclass
class OHLCVData:
    """OHLCV data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    symbol: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol
        }


class DataProcessor:
    """
    Advanced data processing for financial market data.
    
    Provides cleaning, validation, transformation, and technical analysis
    using TA-Lib for reliable and consistent indicator calculations.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = logger
        self.talib_available = self._check_talib_availability()
    
    def _check_talib_availability(self) -> bool:
        """Check if TA-Lib is available for import."""
        try:
            import talib
            return True
        except ImportError:
            logger.warning("TA-Lib not available. Install with: pip install TA-Lib")
            logger.info("macOS users: brew install ta-lib")
            return False
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            return df
        
        # Remove invalid data points
        df = df.dropna()
        
        # Ensure OHLC relationships are valid
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['volume'] < 0)
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Removing {invalid_count} invalid OHLCV records")
            df = df[~invalid_mask]
        
        return df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        timeframe: str = "1H",
        timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: DataFrame with timestamp index or column
            timeframe: Target timeframe (1T, 5T, 1H, 1D, etc.)
            timestamp_col: Name of timestamp column
            
        Returns:
            pd.DataFrame: Resampled data
        """
        if df.empty:
            return df
        
        # Ensure timestamp column is datetime
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        
        # Define aggregation rules for OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Keep only columns that exist in the DataFrame
        agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
        
        resampled = df.resample(timeframe).agg(agg_rules).dropna()
        
        logger.debug(f"Resampled data from {len(df)} to {len(resampled)} records ({timeframe})")
        
        return resampled
    
    def calculate_returns(self, df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """
        Calculate various return metrics.
        
        Args:
            df: DataFrame with price data
            price_col: Column containing prices
            
        Returns:
            pd.DataFrame: DataFrame with return columns added
        """
        if df.empty or price_col not in df.columns:
            return df
        
        df = df.copy()
        
        # Simple returns
        df['returns'] = df[price_col].pct_change()
        
        # Log returns
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Rolling volatility (20-period)
        df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        return df
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        
        Args:
            df: DataFrame to check
            columns: Columns to check for outliers
            method: Method for outlier detection (iqr, zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outlier information
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            df[f'{col}_outlier'] = outlier_mask
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                logger.info(f"Detected {outlier_count} outliers in column '{col}'")
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators using TA-Lib when available,
        fallback to pandas calculations otherwise.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        if self.talib_available:
            return self._add_talib_indicators(df)
        else:
            return self._add_pandas_indicators(df)
    
    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using TA-Lib.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with TA-Lib indicators added
        """
        try:
            import talib
            
            # Convert to numpy arrays for TA-Lib
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values if 'volume' in df.columns else None
            
            # Moving Averages using TA-Lib
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # MACD using TA-Lib
            df['macd'], df['macd_signal'], df['macd_histogram'] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # RSI using TA-Lib
            df['rsi'] = talib.RSI(close, timeperiod=14)
            
            # Bollinger Bands using TA-Lib
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Additional TA-Lib indicators
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # Momentum indicators
            df['momentum'] = talib.MOM(close, timeperiod=10)
            df['roc'] = talib.ROC(close, timeperiod=10)
            
            # Volume indicators (if volume data available)
            if volume is not None:
                df['ad'] = talib.AD(high, low, close, volume)  # Accumulation/Distribution
                df['obv'] = talib.OBV(close, volume)  # On-Balance Volume
                
                # Volume-based simple indicators
                df['volume_sma'] = talib.SMA(volume, timeperiod=20)
                df['volume_ratio'] = volume / df['volume_sma'].values
            
            # Pattern Recognition (select useful patterns)
            if 'open' in df.columns:
                open_prices = df['open'].values
                df['doji'] = talib.CDLDOJI(open_prices, high, low, close)
                df['hammer'] = talib.CDLHAMMER(open_prices, high, low, close)
                df['shooting_star'] = talib.CDLSHOOTINGSTAR(open_prices, high, low, close)
                df['engulfing'] = talib.CDLENGULFING(open_prices, high, low, close)
            
            logger.debug(f"Added TA-Lib technical indicators to {len(df)} records")
            
        except Exception as e:
            logger.error(f"Failed to add TA-Lib indicators: {e}")
            logger.warning("Falling back to pandas-based indicators")
            df = self._add_pandas_indicators(df)
        
        return df
    
    def _add_pandas_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using pandas calculations (fallback).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with pandas-based indicators added
        """
        try:
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ATR (Average True Range)
            if all(col in df.columns for col in ['high', 'low', 'open']):
                tr1 = df['high'] - df['low']
                tr2 = abs(df['high'] - df['close'].shift(1))
                tr3 = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # On-Balance Volume (simplified)
                volume_direction = np.where(df['close'] > df['close'].shift(1), 1, -1)
                df['obv'] = (df['volume'] * volume_direction).cumsum()
            
            logger.debug(f"Added pandas-based technical indicators to {len(df)} records")
            
        except Exception as e:
            logger.error(f"Failed to add pandas indicators: {e}")
        
        return df
    
    def calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom indicators not available in standard libraries.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with custom indicators added
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            # Custom Stochastic Oscillator
            if all(col in df.columns for col in ['high', 'low', 'close']):
                # Calculate %K
                lowest_low = df['low'].rolling(window=14).min()
                highest_high = df['high'].rolling(window=14).max()
                df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                
                # Calculate %D (3-period SMA of %K)
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Custom volatility indicators
            df['realized_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            df['price_channel_high'] = df['high'].rolling(window=20).max()
            df['price_channel_low'] = df['low'].rolling(window=20).min()
            df['price_channel_mid'] = (df['price_channel_high'] + df['price_channel_low']) / 2
            
            # Custom trend strength indicator
            sma_10 = df['close'].rolling(window=10).mean()
            sma_30 = df['close'].rolling(window=30).mean()
            df['trend_strength'] = abs(sma_10 - sma_30) / df['close']
            
            logger.debug(f"Added custom technical indicators to {len(df)} records")
            
        except Exception as e:
            logger.error(f"Failed to add custom indicators: {e}")
        
        return df


def process_ohlcv_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw market data into standardized OHLCV format.
    
    Args:
        market_data: Raw market data dictionary
        
    Returns:
        Dict[str, Any]: Processed market data
    """
    try:
        # Extract and validate required fields
        symbol = market_data.get("symbol")
        if not symbol:
            raise ValueError("Missing symbol in market data")
        
        # Handle different data formats
        if "price" in market_data:
            # Simple price update
            processed = {
                "symbol": symbol,
                "timestamp": market_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "close": market_data["price"],
                "volume": market_data.get("volume", 0),
                "data_type": "tick"
            }
        else:
            # Full OHLCV data
            processed = {
                "symbol": symbol,
                "timestamp": market_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                "open": market_data.get("open"),
                "high": market_data.get("high"),
                "low": market_data.get("low"),
                "close": market_data.get("close"),
                "volume": market_data.get("volume", 0),
                "data_type": "ohlcv"
            }
        
        # Add processing metadata
        processed["processed_at"] = datetime.now(timezone.utc).isoformat()
        processed["data_quality"] = "good"  # Would be determined by validation logic
        
        return processed
        
    except Exception as e:
        logger.error(f"Failed to process market data: {e}")
        raise


# Global processor instance
data_processor = DataProcessor()
