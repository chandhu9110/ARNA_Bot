from unittest.mock import patch
from Backend.src.LLM import gemini_response


@patch("Backend.src.LLM.gemini_response")
def test_rag_search(mock_search):

    mock_search.return_value = "Use NPK fertilizer"

    result = gemini_response(
        "What fertilizer for potato disease?"
    )

    assert result is not None