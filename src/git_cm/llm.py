"""LLM provider implementations for git-cm."""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import click

READ_FILE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read content from a file in the repository for additional context. You can call this multiple times to read different files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file from repository root. Supports subdirectories (e.g., 'src/main.py').",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-based). Optional, defaults to 1.",
                    "default": 1,
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (1-based). Optional, defaults to end of file.",
                },
            },
            "required": ["path"],
        },
    },
}

GREP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "Search for a pattern across all files in the repository using grep. Returns a list of relative file paths that match. Useful for finding where specific functions, classes, or patterns are used.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The grep pattern to search for (supports regex).",
                },
                "include": {
                    "type": "string",
                    "description": "Optional file glob pattern to limit search (e.g., '*.py', '*.{js,ts}'). Uses grep --include.",
                },
            },
            "required": ["pattern"],
        },
    },
}

DIFF_MORE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "diff_more",
        "description": "Get additional diff content that was truncated. Call this when the diff shown in the prompt was truncated and you need to see more changes to understand the full scope.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

MESSAGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "message",
        "description": "Submit the final commit message when you are ready. This tool should be called when you have analyzed the diff and are ready to propose a commit message. The message will be shown to the user for confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The commit message to propose. Should be concise, descriptive, and follow the repository's style conventions.",
                }
            },
            "required": ["message"],
        },
    },
}

TOOLS = [READ_FILE_TOOL_SCHEMA, GREP_TOOL_SCHEMA, DIFF_MORE_TOOL_SCHEMA, MESSAGE_TOOL_SCHEMA]


class ToolResult:
    """Result of a tool call."""

    def __init__(self, tool_call_id: str, name: str, content: str):
        self.tool_call_id = tool_call_id
        self.name = name
        self.content = content


class LLMResponse:
    """Parsed LLM response."""

    def __init__(
        self,
        message: str = "",
        tool_calls: Optional[List[Dict]] = None,
        is_done: bool = False,
        reasoning_content: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        context_window: Optional[int] = None,
    ):
        self.message = message
        self.tool_calls = tool_calls or []
        self.is_done = is_done
        self.reasoning_content = reasoning_content
        self.usage = usage or {}
        self.context_window = context_window


class LLMProvider(ABC):
    """Abstract base class for LLM providers with tool calling support."""

    def __init__(self, api_key: str, model: str, api_base: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base

    @abstractmethod
    def _get_context_window(self) -> Optional[int]:
        """Try to fetch context window from API, fallback to manual config."""
        if self._fetched_context_window is not None:
            return self._fetched_context_window
        
        # Try automatic fetch first
        try:
            model_info = self.client.models.retrieve(self.model)
            if hasattr(model_info, 'context_window'):
                self._fetched_context_window = model_info.context_window
                return self._fetched_context_window
        except Exception:
            pass  # API might not support this or key lacks permission
        
        # Fallback to manual config
        if self._context_window is not None:
            return self._context_window
        
        return None

    def generate(self, system_prompt: str, messages: List[Dict]) -> LLMResponse:
        """Generate response with possible tool calls."""
        pass




class OpenAIProvider(LLMProvider):
    """OpenAI-compatible API provider with tool calling."""

    def __init__(self, api_key: str, model: str, api_base: Optional[str] = None, context_window: Optional[int] = None):
        super().__init__(api_key, model, api_base)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install it with: pip install openai"
            )

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = OpenAI(**client_kwargs)
        self._context_window = context_window
        self._fetched_context_window = None

    def _get_context_window(self) -> Optional[int]:
        """Try to fetch context window from API, fallback to manual config."""
        if self._fetched_context_window is not None:
            return self._fetched_context_window
        
        # Try automatic fetch first
        try:
            model_info = self.client.models.retrieve(self.model)
            if hasattr(model_info, 'context_window'):
                self._fetched_context_window = model_info.context_window
                return self._fetched_context_window
        except Exception:
            pass  # API might not support this or key lacks permission
        
        # Fallback to manual config
        if self._context_window is not None:
            return self._context_window
        
        return None

    def generate(self, system_prompt: str, messages: List[Dict]) -> LLMResponse:
        """Generate response using OpenAI API with tool support."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=8192,
            )

            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)

            # Check for tool calls
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    })
                usage = {}
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(response.usage, 'total_tokens', 0),
                    }
                
                return LLMResponse(
                    message=message.content or "",
                    tool_calls=tool_calls,
                    reasoning_content=reasoning,
                    usage=usage,
                    context_window=self._get_context_window(),
                )

            # No tool calls - this is the final answer
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0),
                }
            
            return LLMResponse(
                message=message.content.strip() if message.content else "",
                is_done=True,
                reasoning_content=reasoning,
                usage=usage,
                context_window=self._get_context_window(),
            )

        except Exception as e:
            click.echo(f"Error calling OpenAI API: {e}", err=True)
            raise click.ClickException(f"LLM API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider with tool calling."""

    def __init__(self, api_key: str, model: str, api_base: Optional[str] = None, context_window: Optional[int] = None):
        super().__init__(api_key, model, api_base)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install it with: pip install anthropic"
            )

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        self.client = Anthropic(**client_kwargs)
        self._context_window = context_window

    def _get_context_window(self) -> Optional[int]:
        """Try to fetch context window from API, fallback to manual config."""
        if self._fetched_context_window is not None:
            return self._fetched_context_window
        
        # Try automatic fetch first
        try:
            model_info = self.client.models.retrieve(self.model)
            if hasattr(model_info, 'context_window'):
                self._fetched_context_window = model_info.context_window
                return self._fetched_context_window
        except Exception:
            pass  # API might not support this or key lacks permission
        
        # Fallback to manual config
        if self._context_window is not None:
            return self._context_window
        
        return None

    def generate(self, system_prompt: str, messages: List[Dict]) -> LLMResponse:
        """Generate response using Anthropic API with tool support."""
        try:
            # Convert OpenAI-format messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg.get("role") == "tool":
                    # Tool results in Anthropic format
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    })
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Assistant with tool calls
                    content_blocks = []
                    for tc in msg.get("tool_calls", []):
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": json.loads(tc.get("function", {}).get("arguments", "{}")),
                        })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })
                elif msg.get("content"):
                    anthropic_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })

            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=0.7,
                system=system_prompt,
                tools=TOOLS,
                messages=anthropic_messages,
            )

            # Check for tool_use blocks
            tool_calls = []
            final_text = ""
            reasoning = ""

            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    })
                elif block.type == "text":
                    final_text += block.text
                elif hasattr(block, "type") and getattr(block, "type", None) == "thinking":
                    # Anthropic thinking block (reasoning)
                    reasoning += getattr(block, "thinking", "")

            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                    "total_tokens": getattr(response.usage, 'input_tokens', 0) + getattr(response.usage, 'output_tokens', 0),
                }
            
            if tool_calls:
                return LLMResponse(
                    tool_calls=tool_calls,
                    reasoning_content=reasoning if reasoning else None,
                    usage=usage,
                    context_window=self._context_window,
                )

            return LLMResponse(
                message=final_text.strip(),
                is_done=True,
                reasoning_content=reasoning if reasoning else None,
                usage=usage,
                context_window=self._context_window,
            )

        except Exception as e:
            click.echo(f"Error calling Anthropic API: {e}", err=True)
            raise click.ClickException(f"LLM API error: {e}")


def create_provider(provider_type: str, api_key: str, model: str, api_base: Optional[str] = None, context_window: Optional[int] = None) -> LLMProvider:
    """Factory function to create the appropriate LLM provider."""
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider(api_key, model, api_base, context_window)
    elif provider_type == "anthropic":
        return AnthropicProvider(api_key, model, api_base, context_window)
    else:
        raise ValueError(f"Unsupported provider: {provider_type}. Use 'openai' or 'anthropic'.")
