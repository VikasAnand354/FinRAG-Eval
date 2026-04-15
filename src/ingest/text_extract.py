import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "section", "article", "header", "footer", "table",
}
_SKIP_TAGS = {"script", "style", "head"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "br":
            self._parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text with paragraph breaks preserved.

    Block elements (p, div, h1-h6, tr, li, section) become double newlines.
    Inline content is joined as-is. XBRL head metadata is skipped.
    """
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    # Collapse runs of spaces/tabs on each line
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
