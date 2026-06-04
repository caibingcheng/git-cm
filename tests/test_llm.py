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

    def test_openai_generate_stream_text_only(self):
        """Test OpenAI stream with plain text response."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Simulate streaming chunks
            def make_chunk(content):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = content
                chunk.choices[0].delta.reasoning_content = None
                chunk.choices[0].delta.tool_calls = None
                return chunk

            # Final usage chunk (choices is empty)
            usage_chunk = MagicMock()
            usage_chunk.choices = []
            usage_chunk.usage = MagicMock(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

            mock_client.chat.completions.create.return_value = [
                make_chunk("feat: "),
                make_chunk("add "),
                make_chunk("feature"),
                usage_chunk,
            ]

            provider = OpenAIProvider("test-key", "gpt-4")
            chunks = list(provider.generate_stream("system", [{"role": "user", "content": "test"}]))

            assert len(chunks) == 4  # 3 text + 1 done
            assert chunks[0].type == "text_delta"
            assert chunks[0].text_delta == "feat: "
            assert chunks[1].type == "text_delta"
            assert chunks[1].text_delta == "add "
            assert chunks[2].type == "text_delta"
            assert chunks[2].text_delta == "feature"
            assert chunks[3].type == "done"
            assert chunks[3].usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_openai_generate_stream_with_reasoning(self):
        """Test OpenAI stream with reasoning content (DeepSeek)."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            def make_chunk(content, reasoning=None):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = content
                chunk.choices[0].delta.reasoning_content = reasoning
                chunk.choices[0].delta.tool_calls = None
                return chunk

            usage_chunk = MagicMock()
            usage_chunk.choices = []
            usage_chunk.usage = None

            mock_client.chat.completions.create.return_value = [
                make_chunk(None, "Thinking..."),
                make_chunk("feat: ", None),
                make_chunk("fix bug", None),
                usage_chunk,
            ]

            provider = OpenAIProvider("test-key", "gpt-4")
            chunks = list(provider.generate_stream("system", [{"role": "user", "content": "test"}]))

            assert len(chunks) == 4
            assert chunks[0].type == "reasoning_delta"
            assert chunks[0].reasoning_delta == "Thinking..."
            assert chunks[1].type == "text_delta"
            assert chunks[1].text_delta == "feat: "
            assert chunks[2].type == "text_delta"
            assert chunks[2].text_delta == "fix bug"
            assert chunks[3].type == "done"

    def test_openai_generate_stream_tool_call(self):
        """Test OpenAI stream with tool call fragments."""
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            def make_text_chunk(content):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = content
                chunk.choices[0].delta.reasoning_content = None
                chunk.choices[0].delta.tool_calls = None
                return chunk

            def make_tool_chunk(idx, tid=None, name=None, args=None):
                chunk = MagicMock()
                chunk.choices = [MagicMock()]
                chunk.choices[0].delta = MagicMock()
                chunk.choices[0].delta.content = None
                chunk.choices[0].delta.reasoning_content = None

                tc = MagicMock()
                tc.index = idx
                tc.id = tid or ""
                tc.function = MagicMock()
                tc.function.name = name or ""
                tc.function.arguments = args or ""
                chunk.choices[0].delta.tool_calls = [tc]
                return chunk

            usage_chunk = MagicMock()
            usage_chunk.choices = []
            usage_chunk.usage = None

            mock_client.chat.completions.create.return_value = [
                make_text_chunk("Let me "),
                make_text_chunk("read the file."),
                make_tool_chunk(0, tid="call_123", name="read_file"),
                make_tool_chunk(0, args='{"path": "'),
                make_tool_chunk(0, args='src/main.py"}'),
                usage_chunk,
            ]

            provider = OpenAIProvider("test-key", "gpt-4")
            chunks = list(provider.generate_stream("system", [{"role": "user", "content": "test"}]))

            assert len(chunks) == 3  # 2 text + 1 tool_calls
            assert chunks[0].type == "text_delta"
            assert chunks[0].text_delta == "Let me "
            assert chunks[1].type == "text_delta"
            assert chunks[1].text_delta == "read the file."
            assert chunks[2].type == "tool_calls"
            assert len(chunks[2].tool_calls) == 1
            assert chunks[2].tool_calls[0]["id"] == "call_123"
            assert chunks[2].tool_calls[0]["name"] == "read_file"
            assert chunks[2].tool_calls[0]["arguments"]["path"] == "src/main.py"

    def test_provider_supports_streaming(self):
        """Test that real providers declare streaming support."""
        with patch("openai.OpenAI"):
            openai_provider = OpenAIProvider("test-key", "gpt-4")
            assert openai_provider.supports_streaming is True

        # Base class does not support streaming
        from git_cm.llm import LLMProvider
        assert LLMProvider.supports_streaming is False


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
