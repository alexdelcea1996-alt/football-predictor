"""
Temporal cross-validation strategy for time-series data.

Ensures no look-ahead bias by training on past data only.
"""

from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class TemporalCrossValidator(BaseCrossValidator):
    """
    Temporal cross-validation with expanding or sliding windows.
    
    Respects chronological order: always train on earlier data,
    validate on later data to prevent look-ahead bias.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 100,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None:
        """
        Args:
            n_splits: Number of CV folds
            min_train_size: Minimum training set size
            test_size: Size of each test fold (None = auto)
            gap: Gap between train and test (prevents leakage)
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for temporal CV."""
        n_samples = len(X)
        
        # Calculate test size if not specified
        if self.test_size is None:
            available = n_samples - self.min_train_size - self.gap
            test_size = max(1, available // self.n_splits)
        else:
            test_size = self.test_size
        
        # Generate folds
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test end is at the back, moving forward for each split
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # Ensure minimum training size
            train_end = test_start - self.gap
            if train_end < self.min_train_size:
                continue
            
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            if len(test_indices) > 0 and len(train_indices) >= self.min_train_size:
                yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits


class DateBasedCV:
    """
    Cross-validation based on date boundaries.
    
    Splits data at specific date cutoffs for more realistic evaluation.
    """
    
    def __init__(
        self,
        date_column: str = "date",
        n_splits: int = 5,
        min_train_months: int = 6,
        test_months: int = 2,
    ) -> None:
        self.date_column = date_column
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        self.test_months = test_months
    
    def split(
        self,
        df: pd.DataFrame,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate train/test DataFrames based on date splits."""
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column)
        
        min_date = df[self.date_column].min()
        max_date = df[self.date_column].max()
        
        # Calculate split points
        total_months = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month)
        
        for i in range(self.n_splits):
            # Work backward from the end
            test_end_offset = i * self.test_months
            test_end = max_date - pd.DateOffset(months=test_end_offset)
            test_start = test_end - pd.DateOffset(months=self.test_months)
            train_end = test_start - pd.DateOffset(days=1)
            
            # Ensure minimum training period
            train_start = min_date
            if (train_end - train_start).days < self.min_train_months * 30:
                continue
            
            train_mask = df[self.date_column] <= train_end
            test_mask = (df[self.date_column] > test_start) & (df[self.date_column] <= test_end)
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(train_df) > 0 and len(test_df) > 0:
                yield train_df.reset_index(drop=True), test_df.reset_index(drop=True)
