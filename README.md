# git-cm

AI-powered git commit message generator.

`git-cm` analyzes your staged changes and recent commit history to automatically generate meaningful commit messages that match your repository's style and conventions.

## Features

- **Smart Style Analysis**: Analyzes your recent commit history to match the style (conventional commits, emoji usage, formatting, etc.)
- **Multiple LLM Providers**: Supports OpenAI, Anthropic Claude, and any compatible APIs
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

### Basic Usage

```bash
# Stage your changes
git add .

# Generate commit message
git-cm
```

### Skip Confirmation

```bash
git-cm --yes
# or
git-cm -y
```

### Use as Git Subcommand

After installation, you can also use:

```bash
git cm
```

## Configuration

### First Run

On first run, `git-cm` will interactively guide you through configuration:

```bash
$ git-cm
No configuration found.
Welcome to git-cm! Let's set up your configuration.

Select LLM provider (openai, anthropic): openai
Enter your API key: [hidden]
Custom API base URL (optional, press Enter to skip):
Enter model name [gpt-4o-mini]:
Would you like to customize the system prompt? [y/N]: n

Configuration saved to ~/.config/git-cm/config.toml
```

### Configuration File

Configuration is stored in `~/.config/git-cm/config.toml`:

```toml
provider = "openai"
api_key = "your-api-key"
api_base = ""  # Optional: custom API endpoint
model = "gpt-4o-mini"
system_prompt = "..."
```

### Environment Variables

You can also set configuration via environment variables:

```bash
export GIT_CM_PROVIDER=openai
export GIT_CM_API_KEY=your-api-key
export GIT_CM_API_BASE=https://custom.api.endpoint
export GIT_CM_MODEL=gpt-4o
```

Environment variables take precedence over config file values.

### CLI Options

CLI parameters take the highest precedence:

```bash
git-cm --provider openai --model gpt-4o --api-key sk-...
```

**Available options:**

| Option | Description |
|--------|-------------|
| `--provider` | LLM provider (`openai` or `anthropic`) |
| `--model` | Model name |
| `--api-key` | API key |
| `--api-base` | Custom API base URL |
| `--yes`, `-y` | Skip confirmation and commit directly |
| `--version` | Show version |

## Supported Providers

### OpenAI

```toml
provider = "openai"
api_key = "sk-..."
model = "gpt-4o-mini"
```

Supports any OpenAI-compatible API (DeepSeek, Silicon Flow, Azure, etc.) via `api_base`.

### Anthropic Claude

```toml
provider = "anthropic"
api_key = "sk-ant-..."
model = "claude-3-5-sonnet-20241022"
```

## How It Works

1. **Repository Check**: Verifies you're in a git repository
2. **Staged Changes**: Checks for staged changes
3. **User Validation**: Displays git user config and warns if not found in history
4. **History Analysis**: Fetches last 5 commits and analyzes style
5. **Diff Analysis**: Gets staged diff
6. **Message Generation**: Sends prompt to LLM with style context
7. **Confirmation**: Shows generated message for confirmation
8. **Commit**: Executes `git commit` with the message

## Style Analysis

`git-cm` analyzes your commit history for:

- **Conventional Commits**: Detects `feat:`, `fix:`, `docs:`, etc.
- **Scope Usage**: Detects `feat(auth):` pattern
- **Emoji Usage**: Detects emoji in messages
- **Message Length**: Calculates average length
- **Capitalization**: Detects uppercase/lowercase start
- **Punctuation**: Detects period at end

The generated message will match these conventions.

## Requirements

- Python 3.9+
- Git
- API key for your chosen LLM provider

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Project Structure

```
git-cm/
├── src/git_cm/
│   ├── __init__.py      # Version
│   ├── cli.py           # CLI entry point
│   ├── config.py        # Configuration management
│   ├── git_utils.py     # Git operations
│   ├── llm.py           # LLM provider implementations
│   ├── prompt.py        # Prompt generation
│   └── style.py         # Commit style analysis
└── tests/
    ├── conftest.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_git_utils.py
    ├── test_llm.py
    ├── test_prompt.py
    └── test_style.py
```

## License

MIT
