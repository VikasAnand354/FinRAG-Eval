import re
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text.

    Collapses multiple spaces. Preserves all text content including
    numbers, percentages, and currency values.
    """
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
