# git-cm

AI-powered git commit message generator.

`git-cm` analyzes your staged changes and recent commit history to automatically generate meaningful commit messages that match your repository's style and conventions.

## Features

- **Smart Style Analysis**: Matches your commit style (conventional commits, emoji, formatting, etc.)
- **Multiple LLM Providers**: Supports OpenAI, Anthropic Claude, and any OpenAI-compatible API
- **Interactive Setup**: First-time configuration wizard
- **Flexible Configuration**: TOML config file with CLI override support
- **Safety Checks**: Validates git user configuration against commit history

## Installation

### From GitHub

```bash
pip install git+https://github.com/yourusername/git-cm.git
```

### For Development

```bash
git clone https://github.com/yourusername/git-cm.git
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

Configuration is read from `~/.config/git-cm/config.toml`:

```toml
provider = "openai"
api_key = "your-api-key"
model = "gpt-4o-mini"
system_prompt = "..."  # optional
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
