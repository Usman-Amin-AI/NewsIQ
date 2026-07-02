import json
import os
from pathlib import Path
from typing import Dict, Optional

from langchain_core.documents import Document


class DocumentCacheError(Exception):
    pass


class DocumentCache:
    def __init__(self, cache_path: str) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.index: Dict[str, Dict[str, str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, str]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            raise DocumentCacheError(f"Could not load document cache: {exc}") from exc

    def _save_cache(self) -> None:
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self.index, handle, indent=2)

    def get(self, url: str) -> Optional[Dict[str, str]]:
        return self.index.get(url)

    def set(self, url: str, content_hash: str, metadata: Dict[str, str]) -> None:
        self.index[url] = {"content_hash": content_hash, **metadata}
        self._save_cache()

    def should_skip(self, url: str, content_hash: str) -> bool:
        existing = self.get(url)
        return bool(existing and existing.get("content_hash") == content_hash)

    def urls(self):
        return list(self.index.keys())
