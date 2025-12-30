"""
Configuration management for Football Predictor.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API configuration for data providers."""
    
    model_config = SettingsConfigDict(env_prefix="FP_API_")
    
    # Sportmonks API
    sportmonks_api_key: str = Field(default="", description="Sportmonks API key")
    sportmonks_base_url: str = Field(
        default="https://api.sportmonks.com/v3/football",
        description="Sportmonks API base URL"
    )
    
    # API-Football (fallback)
    api_football_key: str = Field(default="", description="API-Football key")
    api_football_base_url: str = Field(
        default="https://v3.football.api-sports.io",
        description="API-Football base URL"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(default=60, ge=1, le=300)
    request_timeout: int = Field(default=30, ge=5, le=120)


class FeatureSettings(BaseSettings):
    """Feature engineering configuration."""
    
    model_config = SettingsConfigDict(env_prefix="FP_FEATURE_")
    
    # Rolling window sizes
    rolling_windows: list[int] = Field(default=[5, 10, 20])
    
    # Elo rating parameters
    elo_k_factor: float = Field(default=20.0, ge=10.0, le=40.0)
    elo_k_factor_important: float = Field(default=30.0, ge=20.0, le=50.0)
    elo_home_advantage: float = Field(default=100.0, ge=50.0, le=150.0)
    elo_initial_rating: float = Field(default=1500.0)
    
    # Exponential decay for weighted averages
    decay_factor: float = Field(default=0.95, ge=0.8, le=0.99)
    
    # Head-to-head lookback
    h2h_lookback_matches: int = Field(default=5, ge=3, le=10)


class ModelSettings(BaseSettings):
    """Model training configuration."""
    
    model_config = SettingsConfigDict(env_prefix="FP_MODEL_")
    
    # Cross-validation
    cv_folds: int = Field(default=5, ge=3, le=10)
    
    # Early stopping
    early_stopping_rounds: int = Field(default=50, ge=10, le=200)
    
    # CatBoost defaults
    catboost_iterations: int = Field(default=1000, ge=100, le=5000)
    catboost_learning_rate: float = Field(default=0.05, ge=0.01, le=0.3)
    catboost_depth: int = Field(default=6, ge=4, le=10)
    catboost_l2_leaf_reg: float = Field(default=3.0, ge=1.0, le=10.0)
    
    # XGBoost defaults
    xgboost_n_estimators: int = Field(default=1000, ge=100, le=5000)
    xgboost_learning_rate: float = Field(default=0.05, ge=0.01, le=0.3)
    xgboost_max_depth: int = Field(default=6, ge=3, le=12)
    xgboost_reg_lambda: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Logistic Regression
    logreg_c: float = Field(default=1.0, ge=0.01, le=10.0)
    
    # Ensemble weights (CatBoost, XGBoost, LogReg)
    ensemble_weights: list[float] = Field(default=[0.4, 0.4, 0.2])
    
    # Optuna
    optuna_n_trials: int = Field(default=100, ge=10, le=500)


class DataSettings(BaseSettings):
    """Data processing configuration."""
    
    model_config = SettingsConfigDict(env_prefix="FP_DATA_")
    
    # Supported leagues (Top 5 European)
    leagues: list[str] = Field(
        default=[
            "premier_league",      # England
            "la_liga",             # Spain
            "bundesliga",          # Germany
            "serie_a",             # Italy
            "ligue_1",             # France
        ]
    )
    
    # League IDs for APIs
    league_ids_sportmonks: dict[str, int] = Field(
        default={
            "premier_league": 8,
            "la_liga": 564,
            "bundesliga": 82,
            "serie_a": 384,
            "ligue_1": 301,
        }
    )
    
    league_ids_api_football: dict[str, int] = Field(
        default={
            "premier_league": 39,
            "la_liga": 140,
            "bundesliga": 78,
            "serie_a": 135,
            "ligue_1": 61,
        }
    )
    
    # Minimum matches for training
    min_training_matches: int = Field(default=500, ge=100)
    
    # Data validation
    max_goals_per_match: int = Field(default=15, ge=10)


class Settings(BaseSettings):
    """Main settings aggregator."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Project paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / "data")
    models_dir: Path = Field(default_factory=lambda: Path.cwd() / "models")
    cache_dir: Path = Field(default_factory=lambda: Path.cwd() / ".cache")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    # Sub-settings
    api: APISettings = Field(default_factory=APISettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()
