# git-cm

AI-powered git commit message generator.

`git-cm` analyzes your staged changes and recent commit history to automatically generate meaningful commit messages that match your repository's style and conventions.

## Features

- **Smart Style Analysis**: Matches your commit style (conventional commits, emoji, formatting, etc.)
- **Multiple LLM Providers**: Supports OpenAI, Anthropic Claude, and any OpenAI-compatible API
- **Interactive Setup**: First-time configuration wizard
- **Flexible Configuration**: TOML config file with CLI override support
- **Safety Checks**: Validates git user configuration against commit history
- **Code-Aware**: LLM can read project files and search the codebase to generate more accurate messages

## Installation

### From GitHub

```bash
pip install git+https://github.com/caibingcheng/git-cm.git
```

### For Development

```bash
git clone https://github.com/caibingcheng/git-cm.git
cd git-cm
pip install -e ".[dev]"
```

## Usage

```bash
# Stage your changes
git add .

# Generate commit message (interactive)
git-cm

# Skip confirmation and commit directly
git-cm --yes
```

After installation, you can also use `git cm` as a git subcommand.

## Configuration

Configuration is read from `~/.config/git-cm/config.toml`.

### Single Provider (Legacy Format)

```toml
provider = "openai"
api_key = "your-api-key"
model = "gpt-4o-mini"
api_base = "https://custom.api.endpoint"  # optional, for OpenAI-compatible APIs
system_prompt = "..."  # optional
```

### Multiple Providers

You can configure multiple providers and switch between them:

```toml
default = "work"

[[providers]]
name = "work"
provider_type = "openai"
api_key = "sk-work-key"
model = "gpt-4o"

[[providers]]
name = "personal"
provider_type = "anthropic"
api_key = "sk-personal-key"
model = "claude-3-5-sonnet-20241022"
```

Switch active provider via environment variable:

```bash
export GIT_CM_ACTIVE_PROVIDER=personal
git-cm
```

Supported providers: `openai`, `anthropic`. Any OpenAI-compatible API (DeepSeek, etc.) works via `api_base`.

**Environment variables** (take precedence over config file):

```bash
export GIT_CM_PROVIDER=openai
export GIT_CM_API_KEY=your-api-key
export GIT_CM_API_BASE=https://custom.api.endpoint
export GIT_CM_MODEL=gpt-4o
```

**CLI options** (highest precedence):

```bash
git-cm --provider openai --model gpt-4o --api-key sk-...
```

| Option | Description |
|--------|-------------|
| `--provider` | LLM provider (`openai` or `anthropic`) |
| `--model` | Model name |
| `--api-key` | API key |
| `--api-base` | Custom API base URL |
| `--yes`, `-y` | Skip confirmation and commit directly |
| `--verbose` | Enable verbose output for debugging |
| `--version` | Show version |

## Requirements

- Python 3.9+
- Git
- API key for your chosen LLM provider

## Development

```bash
pytest tests/ -v
```

## License

MIT
