"""
Data preprocessing utilities for football match data.

Handles normalization, encoding, and time-indexed DataFrame creation.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """
    Preprocesses football match data for model training.
    
    Features:
    - Outcome encoding (Home Win=0, Draw=1, Away Win=2)
    - Feature normalization (StandardScaler for ratings, MinMaxScaler for percentages)
    - Time-indexed DataFrame creation
    - Missing value handling
    """
    
    def __init__(self) -> None:
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self._fitted = False
        
        # Columns to scale with each scaler
        self._standard_scale_cols: list[str] = []
        self._minmax_scale_cols: list[str] = []
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the scalers on the training data.
        
        Args:
            df: Training DataFrame with features
            
        Returns:
            Self for method chaining
        """
        # Identify columns for each scaler type
        self._standard_scale_cols = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["elo", "rating", "diff", "position"])
        ]
        
        self._minmax_scale_cols = [
            col for col in df.columns
            if any(keyword in col.lower() for keyword in ["pct", "percent", "possession", "rate"])
        ]
        
        # Fit standard scaler
        if self._standard_scale_cols:
            standard_data = df[self._standard_scale_cols].fillna(0)
            self.standard_scaler.fit(standard_data)
        
        # Fit minmax scaler
        if self._minmax_scale_cols:
            minmax_data = df[self._minmax_scale_cols].fillna(0)
            self.minmax_scaler.fit(minmax_data)
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            df: DataFrame with features to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        result = df.copy()
        
        # Apply standard scaler
        if self._standard_scale_cols:
            cols_present = [c for c in self._standard_scale_cols if c in result.columns]
            if cols_present:
                scaled = self.standard_scaler.transform(result[cols_present].fillna(0))
                result[cols_present] = scaled
        
        # Apply minmax scaler
        if self._minmax_scale_cols:
            cols_present = [c for c in self._minmax_scale_cols if c in result.columns]
            if cols_present:
                scaled = self.minmax_scaler.transform(result[cols_present].fillna(0))
                result[cols_present] = scaled
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    @staticmethod
    def encode_outcome(df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode match outcome from goals.
        
        Encoding:
        - 0: Home Win
        - 1: Draw
        - 2: Away Win
        
        Args:
            df: DataFrame with home_goals and away_goals columns
            
        Returns:
            DataFrame with added 'outcome' column
        """
        result = df.copy()
        
        if "home_goals" in result.columns and "away_goals" in result.columns:
            conditions = [
                result["home_goals"] > result["away_goals"],
                result["home_goals"] == result["away_goals"],
                result["home_goals"] < result["away_goals"],
            ]
            choices = [0, 1, 2]
            result["outcome"] = np.select(conditions, choices, default=np.nan)
        
        return result
    
    @staticmethod
    def decode_outcome(outcome: int) -> str:
        """Decode numeric outcome to string label."""
        mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        return mapping.get(outcome, "Unknown")
    
    @staticmethod
    def create_time_indexed(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a time-indexed DataFrame for temporal operations.
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            DataFrame with DatetimeIndex
        """
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date").reset_index(drop=True)
        return result
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Literal["drop", "mean", "median", "zero"] = "mean",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy
                - "drop": Drop rows with missing values
                - "mean": Fill with column mean
                - "median": Fill with column median
                - "zero": Fill with zeros
            columns: Specific columns to process (None = all numeric)
            
        Returns:
            DataFrame with handled missing values
        """
        result = df.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        if strategy == "drop":
            result = result.dropna(subset=columns)
        elif strategy == "mean":
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].fillna(result[col].mean())
        elif strategy == "median":
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].fillna(result[col].median())
        elif strategy == "zero":
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].fillna(0)
        
        return result
    
    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calendar-based features from match date.
        
        Adds:
        - day_of_week: 0-6 (Monday-Sunday)
        - month: 1-12
        - is_weekend: Boolean
        - week_of_season: Week number within the season
        
        Args:
            df: DataFrame with 'date' column
            
        Returns:
            DataFrame with added calendar features
        """
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        
        result["day_of_week"] = result["date"].dt.dayofweek
        result["month"] = result["date"].dt.month
        result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
        
        # Week of season (assuming season starts in August)
        def get_season_week(date: pd.Timestamp) -> int:
            # Season start is typically around August 1
            if date.month >= 8:
                season_start = pd.Timestamp(year=date.year, month=8, day=1)
            else:
                season_start = pd.Timestamp(year=date.year - 1, month=8, day=1)
            
            weeks = (date - season_start).days // 7
            return max(1, min(52, weeks + 1))
        
        result["week_of_season"] = result["date"].apply(get_season_week)
        
        return result
    
    @staticmethod
    def split_by_date(
        df: pd.DataFrame,
        split_date: str | pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by date for temporal validation.
        
        Args:
            df: DataFrame with 'date' column
            split_date: Date to split on (train < split_date <= test)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        split = pd.to_datetime(split_date)
        
        train = df[df["date"] < split].reset_index(drop=True)
        test = df[df["date"] >= split].reset_index(drop=True)
        
        return train, test
