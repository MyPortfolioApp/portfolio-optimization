from pathlib import Path
import hashlib

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

def key_path(prefix: str, key: str, suffix: str = ".csv") -> Path:
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{h}{suffix}"
