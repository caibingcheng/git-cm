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


class TestMultiProviderConfig:
    """Test multi-provider configuration support."""

    def test_load_multi_provider_config(self, temp_dir, monkeypatch):
        """Test loading configuration with multiple providers."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        # Write a multi-provider config file
        config_file.write_text('''
default = "openai"

[[providers]]
name = "openai"
provider_type = "openai"
api_key = "sk-openai"
api_base = ""
model = "gpt-4o-mini"

[[providers]]
name = "anthropic"
provider_type = "anthropic"
api_key = "sk-ant"
api_base = ""
model = "claude-3-5-sonnet-20241022"
''')
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        # Should use default provider (openai)
        assert data["provider"] == "openai"
        assert data["api_key"] == "sk-openai"
        assert data["model"] == "gpt-4o-mini"
        assert config.active_provider_name == "openai"
        assert config.default_provider_name == "openai"
        assert len(config.providers) == 2

    def test_active_provider_env_var(self, temp_dir, monkeypatch):
        """Test GIT_CM_ACTIVE_PROVIDER environment variable."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config_file.write_text('''
default = "openai"

[[providers]]
name = "openai"
provider_type = "openai"
api_key = "sk-openai"
model = "gpt-4o-mini"

[[providers]]
name = "anthropic"
provider_type = "anthropic"
api_key = "sk-ant"
model = "claude-3"
''')
        
        monkeypatch.setenv("GIT_CM_ACTIVE_PROVIDER", "anthropic")
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        # Should use anthropic due to env var
        assert data["provider"] == "anthropic"
        assert data["api_key"] == "sk-ant"
        assert data["model"] == "claude-3"
        assert config.active_provider_name == "anthropic"

    def test_default_provider_fallback(self, temp_dir, monkeypatch):
        """Test fallback to first provider when no default is set."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config_file.write_text('''
[[providers]]
name = "openai"
provider_type = "openai"
api_key = "sk-openai"
model = "gpt-4"

[[providers]]
name = "anthropic"
provider_type = "anthropic"
api_key = "sk-ant"
model = "claude-3"
''')
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        # Should fallback to first provider
        assert data["provider"] == "openai"
        assert config.active_provider_name == "openai"

    def test_migrate_old_config(self, temp_dir, monkeypatch):
        """Test automatic migration of old config format."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        # Write old format config
        config_file.write_text('''
provider = "openai"
api_key = "sk-old"
model = "gpt-4"
''')
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        data = config.load()
        
        # Should be migrated and loaded
        assert data["provider"] == "openai"
        assert data["api_key"] == "sk-old"
        assert data["model"] == "gpt-4"
        assert config.active_provider_name == "openai"
        
        # Check that file was migrated to new format
        content = config_file.read_text()
        assert "[[providers]]" in content
        assert 'default = "openai"' in content

    def test_provider_selection_priority(self, temp_dir, monkeypatch):
        """Test provider selection priority: CLI > env > default > first."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config_file.write_text('''
default = "openai"

[[providers]]
name = "openai"
provider_type = "openai"
api_key = "sk-openai"
model = "gpt-4"

[[providers]]
name = "anthropic"
provider_type = "anthropic"
api_key = "sk-ant"
model = "claude-3"
''')
        
        # Priority 1: CLI override should override everything
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        config.set_cli_override("api_key", "cli-key")
        
        data = config.load()
        assert data["api_key"] == "cli-key"
        assert data["provider"] == "openai"  # default provider
        
        # Priority 2: GIT_CM_ACTIVE_PROVIDER env var
        config2 = Config()
        config2.config_dir = config_dir
        config2.config_file = config_file
        monkeypatch.setenv("GIT_CM_ACTIVE_PROVIDER", "anthropic")
        
        data2 = config2.load()
        assert data2["provider"] == "anthropic"
        assert data2["api_key"] == "sk-ant"
        
        monkeypatch.delenv("GIT_CM_ACTIVE_PROVIDER")
        
        # Priority 3: default provider
        config3 = Config()
        config3.config_dir = config_dir
        config3.config_file = config_file
        
        data3 = config3.load()
        assert data3["provider"] == "openai"  # default
        
        # Priority 4: first provider (when no default)
        config_file.write_text('''
[[providers]]
name = "anthropic"
provider_type = "anthropic"
api_key = "sk-ant"
model = "claude-3"
''')
        
        config4 = Config()
        config4.config_dir = config_dir
        config4.config_file = config_file
        
        data4 = config4.load()
        assert data4["provider"] == "anthropic"  # first provider

    def test_add_and_remove_provider(self, temp_dir, monkeypatch):
        """Test adding and removing providers."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        config.add_provider("openai", "openai", "sk-1", "gpt-4", set_default=True)
        config.add_provider("anthropic", "anthropic", "sk-2", "claude-3")
        
        assert len(config.providers) == 2
        assert config.list_providers() == ["openai", "anthropic"]
        assert config.default_provider_name == "openai"
        
        config.remove_provider("openai")
        assert len(config.providers) == 1
        assert config.default_provider_name == "anthropic"  # fallback to first

    def test_active_provider_not_found_warning(self, temp_dir, monkeypatch, capsys):
        """Test warning when active provider is not found."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config_file.write_text('''
[[providers]]
name = "openai"
provider_type = "openai"
api_key = "sk-openai"
model = "gpt-4"
''')
        
        monkeypatch.setenv("GIT_CM_ACTIVE_PROVIDER", "nonexistent")
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        config.load()
        
        # Should fallback to default/first provider
        assert config.active_provider_name == "openai"

    def test_add_provider_default_system_prompt_not_saved(self, temp_dir, monkeypatch):
        """Test that default system_prompt is NOT written to config file."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        # Add provider without custom system_prompt
        config.add_provider("openai", "openai", "sk-1", "gpt-4", set_default=True)
        config.save()
        
        # Check that system_prompt is NOT in the saved config
        content = config_file.read_text()
        assert "system_prompt" not in content
        
        # But runtime should still return the default
        config2 = Config()
        config2.config_dir = config_dir
        config2.config_file = config_file
        config2.load()
        assert config2.system_prompt == DEFAULT_SYSTEM_PROMPT

    def test_add_provider_custom_system_prompt_saved(self, temp_dir, monkeypatch):
        """Test that custom system_prompt IS written to config file."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        custom_prompt = "You are a specialized commit message generator."
        # Add provider with custom system_prompt
        config.add_provider("openai", "openai", "sk-1", "gpt-4", 
                          system_prompt=custom_prompt, set_default=True)
        config.save()
        
        # Check that system_prompt IS in the saved config
        content = config_file.read_text()
        assert "system_prompt" in content
        assert custom_prompt in content
        
        # And runtime should return the custom prompt
        config2 = Config()
        config2.config_dir = config_dir
        config2.config_file = config_file
        config2.load()
        assert config2.system_prompt == custom_prompt

    def test_interactive_setup_no_custom_prompt_not_saved(self, temp_dir, monkeypatch):
        """Test interactive setup without custom system_prompt does not save it."""
        config_dir = temp_dir / ".config" / "git-cm"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.toml"
        
        config = Config()
        config.config_dir = config_dir
        config.config_file = config_file
        
        # Mock click.prompt to return test values
        prompt_responses = {
            "Enter provider name": "openai",
            "Select LLM provider type": "openai",
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
        config.load()  # Reload to get the saved config
        
        # Config should work normally
        assert config.provider == "openai"
        assert config.api_key == "sk-test123"
        assert config.model == "gpt-4o-mini"
        # system_prompt should still return default at runtime
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT
        
        # But config file should NOT contain system_prompt
        content = config_file.read_text()
        assert "system_prompt" not in content


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
            "Enter provider name": "openai",
            "Select LLM provider type": "openai",
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
        config.load()  # Reload to get the saved config
        
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
            "Enter provider name": "anthropic",
            "Select LLM provider type": "anthropic",
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
        config.load()  # Reload to get the saved config
        
        assert config.provider == "anthropic"
        assert config.api_key == "sk-ant-test123"
        assert config.model == "claude-3-5-sonnet-20241022"
