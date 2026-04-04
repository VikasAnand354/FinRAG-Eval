def test_strip_html_removes_tags():
    from src.ingest.text_extract import strip_html

    result = strip_html("<p>Hello <b>world</b></p>")
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result
    assert ">" not in result


def test_strip_html_empty_string():
    from src.ingest.text_extract import strip_html

    assert strip_html("") == ""


def test_strip_html_plain_text_unchanged():
    from src.ingest.text_extract import strip_html

    text = "Net sales increased 2% to $391.0 billion."
    assert strip_html(text) == text


def test_strip_html_collapses_extra_spaces():
    from src.ingest.text_extract import strip_html

    result = strip_html("<p>Net   sales</p>")
    assert "  " not in result


def test_strip_html_malformed():
    from src.ingest.text_extract import strip_html

    # Should not raise even with malformed HTML
    result = strip_html("<p>Unclosed tag <b>bold")
    assert "Unclosed tag" in result
    assert "bold" in result
    assert "<" not in result


def test_strip_html_preserves_financial_content():
    from src.ingest.text_extract import strip_html

    html = "<td>Revenue</td><td>$391.0 billion</td>"
    result = strip_html(html)
    assert "Revenue" in result
    assert "$391.0 billion" in result
