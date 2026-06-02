"""Tests for LLM provider implementations."""

from unittest.mock import MagicMock, patch

import pytest

from git_cm.llm import AnthropicProvider, LLMResponse, OpenAIProvider, ToolResult, create_provider


class TestOpenAIProvider:
    """Test OpenAI provider."""

    def test_init(self):
        """Test OpenAI provider initialization."""
        with patch("openai.OpenAI") as mock_openai:
            provider = OpenAIProvider("test-key", "gpt-4")
            
            mock_openai.assert_called_once_with(api_key="test-key")
            assert provider.api_key == "test-key"
            assert provider.model == "gpt-4"
            assert provider.api_base is None

    def test_init_with_base_url(self):
        """Test OpenAI provider with custom base URL."""
        with patch("openai.OpenAI") as mock_openai:
            provider = OpenAIProvider("test-key", "gpt-4", "https://custom.api.com")
            
            mock_openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.com",
            )

    def test_openai_generate_done(self):
        """Test OpenAI generate with no tool calls."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "feat: add feature"
            mock_response.choices[0].message.tool_calls = None
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider("test-key", "gpt-4")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is True
            assert result.message == "feat: add feature"
            assert result.tool_calls == []

    def test_openai_generate_tool_call(self):
        """Test OpenAI generate with read_file tool call."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_tool_call = MagicMock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "read_file"
            mock_tool_call.function.arguments = '{"path": "src/main.py"}'
            
            mock_response = MagicMock()
            mock_response.choices[0].message.content = None
            mock_response.choices[0].message.tool_calls = [mock_tool_call]
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider("test-key", "gpt-4")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is False
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0]["id"] == "call_123"
            assert result.tool_calls[0]["name"] == "read_file"
            assert result.tool_calls[0]["arguments"]["path"] == "src/main.py"

    def test_openai_generate_with_reasoning_content(self):
        """Test OpenAI generate extracts reasoning_content from DeepSeek."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "feat: add feature"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].message.reasoning_content = "1. Analyze diff scope.\n2. Check conventional commits."
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider("test-key", "gpt-4")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is True
            assert result.message == "feat: add feature"
            assert result.reasoning_content == "1. Analyze diff scope.\n2. Check conventional commits."

    def test_openai_generate_without_reasoning_content(self):
        """Test OpenAI generate when model does not return reasoning_content."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "fix: resolve bug"
            mock_response.choices[0].message.tool_calls = None
            # No reasoning_content attribute (e.g., standard OpenAI model)
            del mock_response.choices[0].message.reasoning_content
            mock_client.chat.completions.create.return_value = mock_response
            
            provider = OpenAIProvider("test-key", "gpt-4")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is True
            assert result.message == "fix: resolve bug"
            assert result.reasoning_content is None


class TestAnthropicProvider:
    """Test Anthropic provider."""

    def test_init(self):
        """Test Anthropic provider initialization."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            provider = AnthropicProvider("test-key", "claude-3")
            
            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert provider.api_key == "test-key"
            assert provider.model == "claude-3"

    def test_init_with_base_url(self):
        """Test Anthropic provider with custom base URL."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            provider = AnthropicProvider("test-key", "claude-3", "https://custom.api.com")
            
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.com",
            )

    def test_anthropic_generate_done(self):
        """Test Anthropic generate with no tool calls."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="feat: add feature", type="text")]
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider("test-key", "claude-3")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is True
            assert result.message == "feat: add feature"
            assert result.tool_calls == []

    def test_anthropic_generate_tool_call(self):
        """Test Anthropic generate with tool_use block."""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            
            mock_response = MagicMock()
            tool_use_mock = MagicMock()
            tool_use_mock.type = "tool_use"
            tool_use_mock.id = "tool_123"
            tool_use_mock.name = "read_file"
            tool_use_mock.input = {"path": "src/config.py"}
            mock_response.content = [tool_use_mock]
            mock_client.messages.create.return_value = mock_response
            
            provider = AnthropicProvider("test-key", "claude-3")
            result = provider.generate("system prompt", [{"role": "user", "content": "test"}])
            
            assert result.is_done is False
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0]["id"] == "tool_123"
            assert result.tool_calls[0]["name"] == "read_file"
            assert result.tool_calls[0]["arguments"]["path"] == "src/config.py"


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_done(self):
        """Test LLMResponse for done state."""
        response = LLMResponse(message="feat: test", is_done=True)
        assert response.message == "feat: test"
        assert response.is_done is True
        assert response.tool_calls == []

    def test_llm_response_tool_call(self):
        """Test LLMResponse for tool call state."""
        response = LLMResponse(tool_calls=[{"id": "1", "name": "read_file"}])
        assert response.is_done is False
        assert response.message == ""
        assert len(response.tool_calls) == 1


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_tool_result(self):
        """Test ToolResult creation."""
        result = ToolResult("call_123", "read_file", "file content")
        assert result.tool_call_id == "call_123"
        assert result.name == "read_file"
        assert result.content == "file content"


class TestCreateProvider:
    """Test provider factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch("openai.OpenAI"):
            provider = create_provider("openai", "key", "gpt-4")
            assert isinstance(provider, OpenAIProvider)

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        with patch("anthropic.Anthropic"):
            provider = create_provider("anthropic", "key", "claude-3")
            assert isinstance(provider, AnthropicProvider)

    def test_create_provider_case_insensitive(self):
        """Test provider name is case insensitive."""
        with patch("openai.OpenAI"):
            provider = create_provider("OPENAI", "key", "gpt-4")
            assert isinstance(provider, OpenAIProvider)

    def test_create_unsupported_provider(self):
        """Test creating unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider("unsupported", "key", "model")

    def test_provider_unsupported_no_fallback(self):
        """Test that unsupported provider directly raises error without fallback."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider("unsupported", "key", "model")
