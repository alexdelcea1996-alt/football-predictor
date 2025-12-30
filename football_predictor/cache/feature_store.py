"""
Feature store for caching computed features.
"""

from pathlib import Path
from typing import Any
import json
import hashlib
import pickle


class FeatureStore:
    """
    Caches computed features for fast recomputation.
    
    Stores:
    - Elo ratings
    - Rolling statistics
    - H2H records
    - Feature aggregator state
    """
    
    def __init__(self, cache_dir: str | Path | None = None) -> None:
        from football_predictor.config import get_settings
        
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key hash."""
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def save(self, key: str, data: Any) -> None:
        """Save data to cache."""
        cache_key = self._get_cache_key(key)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        
        # Save metadata
        meta_path = self.cache_dir / f"{cache_key}.meta.json"
        with open(meta_path, "w") as f:
            json.dump({"key": key}, f)
    
    def load(self, key: str) -> Any | None:
        """Load data from cache."""
        cache_key = self._get_cache_key(key)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        cache_key = self._get_cache_key(key)
        return (self.cache_dir / f"{cache_key}.pkl").exists()
    
    def delete(self, key: str) -> None:
        """Delete from cache."""
        cache_key = self._get_cache_key(key)
        pkl_path = self.cache_dir / f"{cache_key}.pkl"
        meta_path = self.cache_dir / f"{cache_key}.meta.json"
        
        if pkl_path.exists():
            pkl_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
    
    def clear(self) -> None:
        """Clear all cache."""
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
        for f in self.cache_dir.glob("*.meta.json"):
            f.unlink()
    
    def list_keys(self) -> list[str]:
        """List all cached keys."""
        keys = []
        for meta_path in self.cache_dir.glob("*.meta.json"):
            with open(meta_path) as f:
                data = json.load(f)
                keys.append(data["key"])
        return keys
