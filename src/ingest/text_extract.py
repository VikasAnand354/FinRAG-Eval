import re
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p",
    "div",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "section",
    "article",
    "header",
    "footer",
    "table",
}

# Tags whose full subtree should be skipped (content + children).
# Note: ix:header (the XBRL taxonomy block in SEC filings) is intentionally NOT
# skipped here. Skipping it would remove the XBRL paragraphs from paragraph
# counting, shifting all subsequent chunk indices and breaking gold citations.
# Instead, is_prose_paragraph() in is_interesting_chunk() filters XBRL-dominated
# chunks at QA generation time only.
_SKIP_TAGS = {"script", "style", "head"}

# Regex to detect XBRL namespace-prefixed tokens (e.g. "aapl:SomeMember").
_XBRL_NAMESPACE = re.compile(r"\b[a-z]{2,10}:[A-Z][A-Za-z]+")


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

    def handle_entityref(self, name: str) -> None:
        """Decode named HTML entities (e.g. &nbsp; → space)."""
        import html

        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        """Decode numeric HTML entities (e.g. &#160; → non-breaking space)."""
        import html

        if self._skip_depth == 0:
            self._parts.append(html.unescape(f"&#{name};"))

    def get_text(self) -> str:
        return "".join(self._parts)


def is_prose_paragraph(text: str) -> bool:
    """Return False if the paragraph is dominated by XBRL namespace tokens.

    A paragraph where more than 20% of whitespace-delimited tokens look like
    XBRL namespace references (e.g. 'aapl:DeirdreOBrienMember') is not readable
    financial prose and should be discarded.
    """
    tokens = text.split()
    if not tokens:
        return False
    xbrl_count = sum(1 for t in tokens if _XBRL_NAMESPACE.search(t))
    return (xbrl_count / len(tokens)) < 0.20


def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text with paragraph breaks preserved.

    - Block elements (p, div, h1-h6, tr, li, section) become double newlines.
    - XBRL <ix:header> blocks are skipped entirely.
    - HTML entities (&nbsp;, &#160;, etc.) are decoded.

    Note: XBRL paragraph filtering is intentionally NOT done here. Filtering
    paragraphs before chunking shifts paragraph indices and breaks gold citations.
    Use is_prose_paragraph() in is_interesting_chunk() to skip XBRL-dominated
    chunks at QA generation time only.
    """
    if not html:
        return ""
    extractor = _TextExtractor()
    extractor.feed(html)
    raw = extractor.get_text()

    # Collapse runs of spaces/tabs within lines
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    # Collapse 3+ newlines to 2
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    return raw.strip()
