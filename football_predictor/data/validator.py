"""
Data validation utilities for football match data.

Ensures data quality and consistency before feature engineering.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from football_predictor.config import get_settings


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.is_valid


class DataValidator:
    """
    Validates football match data for quality and consistency.
    
    Checks include:
    - Schema validation (required columns, types)
    - Range validation (goals >= 0, dates reasonable)
    - Consistency checks (no duplicates, valid team names)
    - Temporal validation (chronological order)
    """
    
    REQUIRED_COLUMNS = ["date", "home_team", "away_team"]
    OPTIONAL_COLUMNS = [
        "home_goals", "away_goals", "league", "season", "status",
        "home_xg", "away_xg", "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_possession", "away_possession",
        "home_corners", "away_corners",
    ]
    
    def __init__(self) -> None:
        self.settings = get_settings()
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on a DataFrame.
        
        Args:
            df: DataFrame with match data
            
        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        errors: list[str] = []
        warnings: list[str] = []
        stats: dict[str, Any] = {}
        
        # Schema validation
        schema_result = self._validate_schema(df)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        
        if errors:
            # Can't continue with other validations if schema is invalid
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Range validation
        range_result = self._validate_ranges(df)
        errors.extend(range_result.errors)
        warnings.extend(range_result.warnings)
        
        # Duplicate detection
        dup_result = self._validate_duplicates(df)
        errors.extend(dup_result.errors)
        warnings.extend(dup_result.warnings)
        
        # Temporal validation
        temp_result = self._validate_temporal(df)
        warnings.extend(temp_result.warnings)
        
        # Collect statistics
        stats = self._compute_stats(df)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )
    
    def _validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate column schema."""
        errors = []
        warnings = []
        
        # Check required columns
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        # Check for empty DataFrame
        if len(df) == 0:
            errors.append("DataFrame is empty")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Validate date column type
        if "date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                try:
                    pd.to_datetime(df["date"])
                except Exception:
                    errors.append("Date column cannot be parsed as datetime")
        
        # Warn about missing optional columns
        present_optional = [col for col in self.OPTIONAL_COLUMNS if col in df.columns]
        missing_optional = [col for col in self.OPTIONAL_COLUMNS if col not in df.columns]
        if missing_optional:
            warnings.append(f"Missing optional columns (may reduce prediction accuracy): {missing_optional[:5]}...")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_ranges(self, df: pd.DataFrame) -> ValidationResult:
        """Validate value ranges."""
        errors = []
        warnings = []
        max_goals = self.settings.data.max_goals_per_match
        
        # Goals validation
        for col in ["home_goals", "away_goals"]:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    if (non_null < 0).any():
                        errors.append(f"{col} contains negative values")
                    if (non_null > max_goals).any():
                        warnings.append(f"{col} contains unusually high values (>{max_goals})")
        
        # xG validation (should be between 0 and ~10)
        for col in ["home_xg", "away_xg"]:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    if (non_null < 0).any():
                        errors.append(f"{col} contains negative values")
                    if (non_null > 10).any():
                        warnings.append(f"{col} contains values > 10, which is unusual")
        
        # Possession validation (should be 0-100)
        for col in ["home_possession", "away_possession"]:
            if col in df.columns:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    if ((non_null < 0) | (non_null > 100)).any():
                        errors.append(f"{col} contains values outside 0-100 range")
        
        # Date validation
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            min_date = datetime(1990, 1, 1)
            max_date = datetime.now()
            
            if (dates < min_date).any():
                warnings.append(f"Data contains matches before {min_date.year}")
            if (dates > max_date).any():
                # Future matches are OK if they're scheduled
                future_count = (dates > max_date).sum()
                warnings.append(f"{future_count} matches are scheduled for the future")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """Check for duplicate matches."""
        errors = []
        warnings = []
        
        # Check for exact duplicates
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            errors.append(f"Found {dup_count} exact duplicate rows")
        
        # Check for logical duplicates (same date, teams)
        if all(col in df.columns for col in ["date", "home_team", "away_team"]):
            date_str = pd.to_datetime(df["date"]).dt.date.astype(str)
            match_key = date_str + "_" + df["home_team"] + "_" + df["away_team"]
            
            if match_key.duplicated().any():
                dup_count = match_key.duplicated().sum()
                warnings.append(f"Found {dup_count} matches with same date and teams (possible double entries)")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_temporal(self, df: pd.DataFrame) -> ValidationResult:
        """Validate temporal consistency."""
        warnings = []
        
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            
            # Check for large gaps
            sorted_dates = dates.sort_values()
            gaps = sorted_dates.diff()
            
            large_gaps = gaps[gaps > pd.Timedelta(days=60)]
            if len(large_gaps) > 0:
                warnings.append(
                    f"Found {len(large_gaps)} gaps > 60 days in the data "
                    "(may indicate missing data or season breaks)"
                )
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def _compute_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute summary statistics for the data."""
        stats: dict[str, Any] = {
            "total_matches": len(df),
        }
        
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            stats["date_range"] = {
                "min": dates.min().isoformat(),
                "max": dates.max().isoformat(),
            }
        
        if "home_team" in df.columns:
            stats["unique_home_teams"] = df["home_team"].nunique()
        
        if "away_team" in df.columns:
            stats["unique_away_teams"] = df["away_team"].nunique()
        
        if all(col in df.columns for col in ["home_team", "away_team"]):
            all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
            stats["total_teams"] = len(all_teams)
        
        if "league" in df.columns:
            stats["leagues"] = df["league"].unique().tolist()
        
        if "season" in df.columns:
            stats["seasons"] = df["season"].unique().tolist()
        
        # Outcome distribution (if goals available)
        if all(col in df.columns for col in ["home_goals", "away_goals"]):
            finished = df[df["home_goals"].notna() & df["away_goals"].notna()]
            if len(finished) > 0:
                home_wins = (finished["home_goals"] > finished["away_goals"]).sum()
                draws = (finished["home_goals"] == finished["away_goals"]).sum()
                away_wins = (finished["home_goals"] < finished["away_goals"]).sum()
                
                stats["outcome_distribution"] = {
                    "home_wins": int(home_wins),
                    "draws": int(draws),
                    "away_wins": int(away_wins),
                    "home_win_pct": round(home_wins / len(finished) * 100, 1),
                    "draw_pct": round(draws / len(finished) * 100, 1),
                    "away_win_pct": round(away_wins / len(finished) * 100, 1),
                }
        
        # Missing data summary
        missing = {}
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                missing[col] = int(null_count)
        if missing:
            stats["missing_values"] = missing
        
        return stats
    
    def clean(
        self,
        df: pd.DataFrame,
        drop_duplicates: bool = True,
        drop_missing_required: bool = True,
    ) -> pd.DataFrame:
        """
        Clean the DataFrame by removing invalid rows.
        
        Args:
            df: Input DataFrame
            drop_duplicates: Remove duplicate rows
            drop_missing_required: Remove rows missing required columns
            
        Returns:
            Cleaned DataFrame
        """
        result = df.copy()
        
        # Ensure date is datetime
        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])
        
        # Drop duplicates
        if drop_duplicates:
            result = result.drop_duplicates()
        
        # Drop rows missing required columns
        if drop_missing_required:
            for col in self.REQUIRED_COLUMNS:
                if col in result.columns:
                    result = result[result[col].notna()]
        
        # Sort by date
        if "date" in result.columns:
            result = result.sort_values("date").reset_index(drop=True)
        
        return result
