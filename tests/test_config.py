"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from git_cm.config import Config, DEFAULT_SYSTEM_PROMPT, interactive_setup


class TestConfig:
    """Test configuration management."""

    def test_default_config(self, temp_dir, monkeypatch):
        """Test default configuration values."""
        # Mock config directory
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_dir / "config.toml"
        
        data = config.load()
        
        assert data["provider"] == ""
        assert data["api_key"] == ""
        assert data["api_base"] == ""
        assert data["model"] == ""
        assert data["system_prompt"] == DEFAULT_SYSTEM_PROMPT

    def test_load_from_file(self, temp_dir, monkeypatch):
        """Test loading configuration from file."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        # Write a test config file
        config_file.write_text('''
provider = "openai"
api_key = "test-key"
model = "gpt-4"
''')
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        assert data["provider"] == "openai"
        assert data["api_key"] == "test-key"
        assert data["model"] == "gpt-4"

    def test_cli_override(self, temp_dir, monkeypatch):
        """Test CLI parameter overrides."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        # Write a test config file
        config_file.write_text('''
provider = "openai"
api_key = "file-key"
model = "gpt-4"
''')
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        config.set_cli_override("api_key", "cli-key")
        config.set_cli_override("model", "gpt-4o")
        
        data = config.load()
        
        assert data["provider"] == "openai"
        assert data["api_key"] == "cli-key"  # CLI overrides file
        assert data["model"] == "gpt-4o"      # CLI overrides file

    def test_env_override(self, temp_dir, monkeypatch):
        """Test environment variable overrides."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        # Write a test config file
        config_file.write_text('''
provider = "openai"
api_key = "file-key"
model = "gpt-4"
''')
        
        monkeypatch.setenv("GIT_CM_API_KEY", "env-key")
        monkeypatch.setenv("GIT_CM_MODEL", "gpt-4o-mini")
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        assert data["api_key"] == "env-key"
        assert data["model"] == "gpt-4o-mini"

    def test_save_and_load(self, temp_dir, monkeypatch):
        """Test saving and loading configuration."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        config.set("provider", "anthropic")
        config.set("api_key", "test-key")
        config.set("model", "claude-3")
        config.save()
        
        # Load in a new instance
        config2 = Config()
        config2.config_dir = config_dir
        config2.config_file = config_file
        data = config2.load()
        
        assert data["provider"] == "anthropic"
        assert data["api_key"] == "test-key"
        assert data["model"] == "claude-3"

    def test_is_configured(self, temp_dir, monkeypatch):
        """Test configuration completeness check."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        config.load()
        
        # Not configured initially
        assert not config.is_configured()
        
        # Set required fields
        config.set("provider", "openai")
        config.set("api_key", "test-key")
        config.set("model", "gpt-4")
        
        assert config.is_configured()

    def test_system_prompt_property(self, temp_dir, monkeypatch):
        """Test system_prompt property returns default when not set."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        config.load()
        
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestInteractiveSetup:
    """Test interactive configuration setup."""

    def test_interactive_setup_openai(self, temp_dir, monkeypatch):
        """Test interactive setup with OpenAI provider."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        # Mock click.prompt to return test values
        prompt_responses = {
            "Select LLM provider": "openai",
            "Enter your API key": "sk-test123",
            "Custom API base URL (optional, press Enter to skip)": "",
            "Enter model name": "gpt-4o-mini",
        }
        
        def mock_prompt(*args, **kwargs):
            text = args[0] if args else kwargs.get("text", "")
            for key, value in prompt_responses.items():
                if key in text:
                    return value
            return kwargs.get("default", "")
        
        monkeypatch.setattr("click.prompt", mock_prompt)
        monkeypatch.setattr("click.confirm", lambda *args, **kwargs: False)
        
        interactive_setup(config)
        
        assert config.provider == "openai"
        assert config.api_key == "sk-test123"
        assert config.model == "gpt-4o-mini"

    def test_interactive_setup_anthropic(self, temp_dir, monkeypatch):
        """Test interactive setup with Anthropic provider."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        prompt_responses = {
            "Select LLM provider": "anthropic",
            "Enter your API key": "sk-ant-test123",
            "Custom API base URL (optional, press Enter to skip)": "",
            "Enter model name": "claude-3-5-sonnet-20241022",
        }
        
        def mock_prompt(*args, **kwargs):
            text = args[0] if args else kwargs.get("text", "")
            for key, value in prompt_responses.items():
                if key in text:
                    return value
            return kwargs.get("default", "")
        
        monkeypatch.setattr("click.prompt", mock_prompt)
        monkeypatch.setattr("click.confirm", lambda *args, **kwargs: False)
        
        interactive_setup(config)
        
        assert config.provider == "anthropic"
        assert config.api_key == "sk-ant-test123"
        assert config.model == "claude-3-5-sonnet-20241022"
