import pytest
from langgraph_logic.utils import chunk_by_paragraph, chunk_by_lines

sample_text = "Para 1.\n\nPara 2.\n\nPara 3."

def test_chunk_by_paragraph():
    chunks = chunk_by_paragraph(sample_text)
    assert len(chunks) == 3
    assert chunks[0] == "Para 1."

def test_chunk_by_lines():
    text = "line1\nline2\nline3\nline4\nline5"
    chunks = chunk_by_lines(text, lines_per_chunk=2)
    assert len(chunks) == 3
    assert chunks[1] == "line3\nline4"
