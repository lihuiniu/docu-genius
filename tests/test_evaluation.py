import pytest
from langgraph_logic.evaluate import should_resummarize

def test_should_resummarize_below_threshold():
    result = should_resummarize({
        "summary_scores": [0.1, 0.4, 0.6],
        "threshold": 0.5
    })
    assert result == "resummarize"

def test_should_resummarize_above_threshold():
    result = should_resummarize({
        "summary_scores": [0.8, 0.9, 0.7],
        "threshold": 0.5
    })
    assert result == "index"
