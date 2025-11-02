"""
Helios Data Processing v3

Advanced data processing utilities for market data cleaning,
transformation, and technical indicator computation.
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
    
    Provides cleaning, validation, transformation, and technical analysis.
    """
    
    def __init__(self):
        """Initialize data processor."""
        self.logger = logger
    
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
        Add common technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
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
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            logger.debug(f"Added technical indicators to {len(df)} records")
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
        
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
