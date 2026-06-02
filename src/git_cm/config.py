"""Configuration management for git-cm."""

import os
import sys
from pathlib import Path
from typing import Optional

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


class Config:
    """Manages git-cm configuration."""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "git-cm"
        self.config_file = self.config_dir / "config.toml"
        self._data = {}
        self._cli_overrides = {}

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        """Load configuration from file and environment."""
        self._data = DEFAULT_CONFIG.copy()

        # Load from config file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "rb") as f:
                    file_config = tomllib.load(f)
                    self._data.update(file_config)
            except Exception as e:
                click.echo(f"Warning: Failed to load config file: {e}", err=True)

        # Environment variable overrides
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

    def save(self) -> None:
        """Save current configuration to file."""
        self._ensure_config_dir()
        with open(self.config_file, "wb") as f:
            tomli_w.dump(self._data, f)

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._data.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set configuration value."""
        self._data[key] = value

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


def interactive_setup(config: Config) -> None:
    """Interactive configuration setup for first-time users."""
    click.echo("Welcome to git-cm! Let's set up your configuration.")
    click.echo()

    # Provider selection
    provider = click.prompt(
        "Select LLM provider",
        type=click.Choice(["openai", "anthropic"], case_sensitive=False),
        show_choices=True,
    )
    config.set("provider", provider.lower())

    # API Key
    api_key = click.prompt("Enter your API key", hide_input=True)
    config.set("api_key", api_key)

    # API Base (optional)
    api_base = click.prompt(
        "Custom API base URL (optional, press Enter to skip)",
        default="",
        show_default=False,
    )
    if api_base:
        config.set("api_base", api_base)

    # Model
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    model = click.prompt(
        "Enter model name",
        default=default_models.get(provider.lower(), ""),
    )
    config.set("model", model)

    # System prompt (optional)
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
            config.set("system_prompt", custom_prompt)

    config.save()
    click.echo()
    click.echo(f"Configuration saved to {config.config_file}")
