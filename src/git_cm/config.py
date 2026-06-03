"""Configuration management for git-cm."""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

import click

# Handle tomli import for Python < 3.11
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import tomli_w

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that generates meaningful git commit messages.
Analyze the provided git diff and recent commit history to generate a commit message
that matches the style and conventions used in the repository.

You have access to three tools:
1. `read_file` — read content from a file in the repository for additional context.
2. `grep` — search for a pattern across all files in the repository, returning matching file paths.
3. `message` — submit the final commit message when you are ready.

If the diff doesn't provide enough context to write a good commit message, use the `read_file` and `grep` tools.

When you are ready to provide the final commit message, you MUST call the `message` tool.
Do not return the commit message as plain text without calling the tool.

Commit Message Format Rules:
- The commit message consists of a title (first line) and optional body (subsequent lines)
- Title must be a SINGLE LINE, under 80 characters total
- Body is OPTIONAL — whether to include it should be guided by the style of recent commits in history
- If recent commits mostly have title-only, default to title-only; if they frequently include bodies, follow that convention
- Only add body when the change genuinely needs explanation beyond the title, or when repository convention (per history) favors it
- When body is present, keep it concise and to the point
- When body is present, leave a blank line after the title
- Body can be multiple lines with no length limit
- Use imperative mood (e.g., "add feature" not "added feature")
- Do not include markdown formatting or quotes in the output
- Output ONLY the commit message via the `message` tool, nothing else
- You MUST generate a non-empty commit message

Title Format:
1. First, check if the current branch name contains a ticket ID (e.g., JIRA-123, TICKET-456, #789)
2. If found, use format: `{TICKET-ID}: {description}` (ticket ID replaces the type prefix)
   Example: `JIRA-123: add user authentication`
3. If NO ticket ID found, use conventional commits format: `type: description`
   Example: `feat: add user authentication`
4. Title MUST be under 80 characters including the ticket ID prefix"""

DEFAULT_CONFIG = {
    "provider": "",
    "api_key": "",
    "api_base": "",
    "model": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}

DEFAULT_PROVIDER_TYPE = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
}


class Config:
    """Manages git-cm configuration with multi-provider support."""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "git-cm"
        self.config_file = self.config_dir / "config.toml"
        self._data = {}
        self._cli_overrides = {}
        self._providers = []
        self._default_provider_name = None
        self._active_provider_name = None

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _migrate_old_config(self, file_config: dict) -> dict:
        """Migrate old flat config to new multi-provider format."""
        provider_name = file_config.get("provider", "default")
        if not provider_name:
            provider_name = "default"
        
        migrated = {
            "default": provider_name,
            "providers": [
                {
                    "name": provider_name,
                    "provider_type": file_config.get("provider", DEFAULT_PROVIDER_TYPE),
                    "api_key": file_config.get("api_key", ""),
                    "api_base": file_config.get("api_base", ""),
                    "model": file_config.get("model", ""),
                    "system_prompt": file_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
                }
            ]
        }
        return migrated

    def _is_new_format(self, file_config: dict) -> bool:
        """Check if config is in new multi-provider format."""
        return "providers" in file_config

    def _get_provider_by_name(self, name: str) -> Optional[dict]:
        """Get provider config by name."""
        for provider in self._providers:
            if provider.get("name") == name:
                return provider
        return None

    def _resolve_active_provider(self) -> Optional[dict]:
        """Resolve which provider should be active based on priority."""
        # Priority 1: GIT_CM_ACTIVE_PROVIDER environment variable
        active_name = os.environ.get("GIT_CM_ACTIVE_PROVIDER")
        if active_name:
            provider = self._get_provider_by_name(active_name)
            if provider:
                self._active_provider_name = active_name
                return provider
            click.echo(f"Warning: Provider '{active_name}' not found in config.", err=True)

        # Priority 2: Config file default
        if self._default_provider_name:
            provider = self._get_provider_by_name(self._default_provider_name)
            if provider:
                self._active_provider_name = self._default_provider_name
                return provider

        # Priority 3: First provider in list
        if self._providers:
            self._active_provider_name = self._providers[0].get("name")
            return self._providers[0]

        return None

    def load(self) -> dict:
        """Load configuration from file and environment."""
        self._data = DEFAULT_CONFIG.copy()
        self._providers = []
        self._default_provider_name = None
        self._active_provider_name = None

        # Load from config file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "rb") as f:
                    file_config = tomllib.load(f)
                    
                    # Migrate old format to new format if needed
                    if not self._is_new_format(file_config):
                        file_config = self._migrate_old_config(file_config)
                        # Save migrated config back to file
                        self._save_config(file_config)
                        click.echo(f"Configuration migrated to new format.")
                    
                    self._providers = file_config.get("providers", [])
                    self._default_provider_name = file_config.get("default")
            except Exception as e:
                click.echo(f"Warning: Failed to load config file: {e}", err=True)

        # Resolve active provider
        active_provider = self._resolve_active_provider()
        
        if active_provider:
            self._data = {
                "provider": active_provider.get("provider_type", ""),
                "api_key": active_provider.get("api_key", ""),
                "api_base": active_provider.get("api_base", ""),
                "model": active_provider.get("model", ""),
                "system_prompt": active_provider.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            }

        # Environment variable overrides (legacy, for backward compatibility)
        env_mapping = {
            "GIT_CM_PROVIDER": "provider",
            "GIT_CM_API_KEY": "api_key",
            "GIT_CM_API_BASE": "api_base",
            "GIT_CM_MODEL": "model",
        }
        for env_var, key in env_mapping.items():
            value = os.environ.get(env_var)
            if value:
                self._data[key] = value

        # CLI overrides (highest priority)
        self._data.update(self._cli_overrides)

        # If system_prompt is a file path, read content from it
        raw_prompt = self._data.get("system_prompt", "")
        if raw_prompt and os.path.isfile(raw_prompt):
            try:
                with open(raw_prompt, "r", encoding="utf-8") as f:
                    self._data["system_prompt"] = f.read()
            except Exception as e:
                click.echo(
                    f"Warning: Failed to read system prompt file '{raw_prompt}': {e}",
                    err=True,
                )

        return self._data

    def _save_config(self, config_data: dict) -> None:
        """Save configuration dict to file."""
        self._ensure_config_dir()
        with open(self.config_file, "wb") as f:
            tomli_w.dump(config_data, f)

    def save(self) -> None:
        """Save current configuration to file in new format."""
        config_data = {
            "providers": self._providers,
        }
        if self._default_provider_name:
            config_data["default"] = self._default_provider_name
        self._save_config(config_data)

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set configuration value on active provider."""
        self._data[key] = value
        # Also update the active provider in providers list
        if self._active_provider_name:
            provider = self._get_provider_by_name(self._active_provider_name)
            if provider and key in ["provider", "api_key", "api_base", "model", "system_prompt"]:
                # Map flat key to provider key
                provider_key = "provider_type" if key == "provider" else key
                provider[provider_key] = value
        elif key in ["provider", "api_key", "api_base", "model", "system_prompt"]:
            # No active provider yet, create one
            provider_name = "default"
            provider_type = self._data.get("provider", DEFAULT_PROVIDER_TYPE) if key != "provider" else value
            provider = {
                "name": provider_name,
                "provider_type": provider_type,
                "api_key": self._data.get("api_key", ""),
                "api_base": self._data.get("api_base", ""),
                "model": self._data.get("model", ""),
                "system_prompt": self._data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            }
            # Update the specific key
            provider_key = "provider_type" if key == "provider" else key
            provider[provider_key] = value
            self._providers.append(provider)
            self._active_provider_name = provider_name
            if not self._default_provider_name:
                self._default_provider_name = provider_name

    def set_cli_override(self, key: str, value: Optional[str]) -> None:
        """Set a CLI parameter override."""
        if value is not None:
            self._cli_overrides[key] = value

    def is_configured(self) -> bool:
        """Check if required configuration is present."""
        return bool(self._data.get("provider") and self._data.get("api_key") and self._data.get("model"))

    @property
    def provider(self) -> str:
        return self._data.get("provider", "")

    @property
    def api_key(self) -> str:
        return self._data.get("api_key", "")

    @property
    def api_base(self) -> str:
        return self._data.get("api_base", "")

    @property
    def model(self) -> str:
        return self._data.get("model", "")

    @property
    def system_prompt(self) -> str:
        return self._data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    # Multi-provider management methods
    
    @property
    def providers(self) -> List[dict]:
        """Get all configured providers."""
        return self._providers.copy()

    @property
    def active_provider_name(self) -> Optional[str]:
        """Get the name of the currently active provider."""
        return self._active_provider_name

    @property
    def default_provider_name(self) -> Optional[str]:
        """Get the name of the default provider."""
        return self._default_provider_name

    def add_provider(self, name: str, provider_type: str, api_key: str, model: str,
                     api_base: str = "", system_prompt: str = "", set_default: bool = False) -> None:
        """Add a new provider configuration."""
        # Check if name already exists
        if self._get_provider_by_name(name):
            raise ValueError(f"Provider '{name}' already exists")
        
        provider = {
            "name": name,
            "provider_type": provider_type,
            "api_key": api_key,
            "api_base": api_base,
            "model": model,
            "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
        }
        self._providers.append(provider)
        
        if set_default or not self._default_provider_name:
            self._default_provider_name = name

    def remove_provider(self, name: str) -> None:
        """Remove a provider by name."""
        for i, provider in enumerate(self._providers):
            if provider.get("name") == name:
                self._providers.pop(i)
                # Update default if needed
                if self._default_provider_name == name:
                    if self._providers:
                        self._default_provider_name = self._providers[0].get("name")
                    else:
                        self._default_provider_name = None
                return
        raise ValueError(f"Provider '{name}' not found")

    def set_default(self, name: str) -> None:
        """Set the default provider."""
        if not self._get_provider_by_name(name):
            raise ValueError(f"Provider '{name}' not found")
        self._default_provider_name = name

    def list_providers(self) -> List[str]:
        """List all provider names."""
        return [p.get("name", "") for p in self._providers]

    def get_provider_config(self, name: Optional[str] = None) -> Optional[dict]:
        """Get configuration for a specific provider."""
        if name is None:
            name = self._active_provider_name
        return self._get_provider_by_name(name)


def interactive_setup(config: Config) -> None:
    """Interactive configuration setup for first-time users."""
    click.echo("Welcome to git-cm! Let's set up your configuration.")
    click.echo()

    # If we already have providers, show them and ask what to do
    if config.providers:
        click.echo("Existing providers:")
        for i, provider in enumerate(config.providers, 1):
            is_default = " (default)" if provider.get("name") == config.default_provider_name else ""
            click.echo(f"  {i}. {provider.get('name')} ({provider.get('provider_type')}){is_default}")
        click.echo()
        
        action = click.prompt(
            "What would you like to do?",
            type=click.Choice(["add", "edit", "remove", "done"], case_sensitive=False),
            default="add",
        )
        
        if action.lower() == "done":
            return
        elif action.lower() == "remove":
            name = click.prompt("Enter provider name to remove")
            try:
                config.remove_provider(name)
                config.save()
                click.echo(f"Provider '{name}' removed.")
            except ValueError as e:
                click.echo(f"Error: {e}", err=True)
            return
        elif action.lower() == "edit":
            click.echo("To edit a provider, remove it and add a new one.")
            return
    
    # Add new provider
    click.echo("Add a new provider:")
    click.echo()
    
    # Provider name
    if config.providers:
        name = click.prompt("Enter provider name (unique identifier)")
    else:
        name = click.prompt("Enter provider name", default="default")
    
    # Provider type
    provider_type = click.prompt(
        "Select LLM provider type",
        type=click.Choice(["openai", "anthropic"], case_sensitive=False),
        show_choices=True,
    )
    
    # API Key
    api_key = click.prompt("Enter your API key", hide_input=True)
    
    # API Base (optional)
    api_base = click.prompt(
        "Custom API base URL (optional, press Enter to skip)",
        default="",
        show_default=False,
    )
    
    # Model
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    model = click.prompt(
        "Enter model name",
        default=default_models.get(provider_type.lower(), ""),
    )
    
    # System prompt (optional)
    system_prompt = ""
    if click.confirm("Would you like to customize the system prompt?", default=False):
        click.echo("Enter your custom system prompt (press Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        custom_prompt = "\n".join(lines)
        if custom_prompt.strip():
            system_prompt = custom_prompt
    
    # Set as default?
    set_default = True
    if config.providers:
        set_default = click.confirm("Set as default provider?", default=False)
    
    try:
        config.add_provider(
            name=name,
            provider_type=provider_type.lower(),
            api_key=api_key,
            model=model,
            api_base=api_base,
            system_prompt=system_prompt,
            set_default=set_default,
        )
        config.save()
        click.echo()
        click.echo(f"Provider '{name}' saved to {config.config_file}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
