from unittest.mock import MagicMock, patch

from specguard.main import HelloLLMResponse, call_hello_llm


@patch("specguard.main.instructor.from_litellm")
def test_call_hello_llm_returns_validated_response(mock_from_litellm) -> None:
    mock_client = MagicMock()
    mock_client.create.return_value = HelloLLMResponse(message="Hello LLM", is_valid=True)
    mock_from_litellm.return_value = mock_client

    result = call_hello_llm()

    assert isinstance(result, HelloLLMResponse)
    assert result.is_valid is True
    assert result.message

    mock_client.create.assert_called_once()
    call_kwargs = mock_client.create.call_args.kwargs
    assert call_kwargs["response_model"] is HelloLLMResponse
    assert len(call_kwargs["messages"]) == 2
