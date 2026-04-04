import json
from pathlib import Path


class CompanyRegistry:
    def __init__(self, path: str):
        self.path = Path(path)
        self._data = self._load()

    def _load(self):
        with open(self.path) as f:
            return json.load(f)

    def list(self):
        return self._data

    def get(self, ticker: str):
        for c in self._data:
            if c["ticker"] == ticker:
                return c
        raise ValueError(f"Company not found: {ticker}")
