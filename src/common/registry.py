import json
from pathlib import Path


class CompanyRegistry:
    def __init__(self, path: str):
        self.path = Path(path)
        self._data = self._load()
        # Build index for O(1) lookup
        self._index: dict[str, dict] = {c["ticker"]: c for c in self._data}

    def _load(self) -> list[dict]:
        with open(self.path) as f:
            return json.load(f)

    def list(self) -> list[dict]:
        return self._data

    def get(self, ticker: str) -> dict:
        if ticker not in self._index:
            raise ValueError(f"Company not found: {ticker}")
        return self._index[ticker]
