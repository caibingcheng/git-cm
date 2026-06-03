"""CLI entry point for git-cm."""

"""CLI entry point for git-cm."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import click

from git_cm.config import Config, interactive_setup
from git_cm.git_utils import (
    check_user_in_history,
    commit_changes,
    find_agents_md,
    get_current_branch,
    get_recent_commits,
    get_repo,
    get_staged_diff,
    get_user_config,
    grep_repo,
    has_staged_changes,
    is_git_repo,
    read_files_batch,
)
from git_cm.llm import ToolResult, create_provider
from git_cm.prompt import build_prompt



class Spinner:
    """Simple CLI spinner for showing progress."""

    def __init__(self, text="Thinking", interval=0.1):
        self.text = text
        self.interval = interval
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _spin(self):
        idx = 0
        while self.running:
            frame = self.frames[idx % len(self.frames)]
            sys.stdout.write(f"\r{frame} {self.text}...")
            sys.stdout.flush()
            idx += 1
            time.sleep(self.interval)
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.text) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


def show_reasoning(content: str) -> None:
    """Display reasoning content in terminal."""
    click.echo(click.style("Thought: ", fg="cyan") + click.style(f"{content}\n", fg="bright_black"))


def verbose_echo(enabled: bool, message: str, **kwargs) -> None:
    """Print message only when verbose mode is enabled."""
    if enabled:
        click.echo(click.style("[verbose] ", fg="magenta") + message, **kwargs)


@click.command()
@click.option("--provider", help="LLM provider (openai or anthropic)")
@click.option("--model", help="Model name")
@click.option("--api-key", help="API key")
@click.option("--api-base", help="Custom API base URL")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation and commit directly")
@click.option("--verbose", is_flag=True, help="Enable verbose output for debugging")
@click.version_option(version="0.1.0")
def main(provider, model, api_key, api_base, yes, verbose):
    """AI-powered git commit message generator."""
    # Initialize configuration
    config = Config()

    # Set CLI overrides
    config.set_cli_override("provider", provider)
    config.set_cli_override("model", model)
    config.set_cli_override("api_key", api_key)
    config.set_cli_override("api_base", api_base)

    # Load configuration
    config.load()

    # Interactive setup if not configured
    if not config.is_configured():
        click.echo("No configuration found.")
        interactive_setup(config)
        config.load()  # Reload after setup

    # Check if we're in a git repository
    current_dir = Path.cwd()
    if not is_git_repo(current_dir):
        click.echo("Error: Not a git repository.", err=True)
        raise click.ClickException("Not a git repository")

    # Get repository
    repo = get_repo(current_dir)

    # Check for staged changes
    if not has_staged_changes(repo):
        click.echo("Error: No staged changes found.", err=True)
        click.echo("Use 'git add' to stage changes before running git-cm.", err=True)
        raise click.ClickException("No staged changes")

    # Get user config
    user_config = get_user_config(repo)
    user_name = user_config["name"]
    user_email = user_config["email"]

    click.echo(f"Git user: {user_name} <{user_email}>")
    verbose_echo(verbose, f"Working directory: {current_dir}")
    verbose_echo(verbose, f"Repo path: {repo.working_tree_dir}")

    # Check if user appears in history
    if not check_user_in_history(repo, user_name, user_email):
        click.echo()
        click.echo(
            "Warning: Your git user name or email does not appear in recent commit history."
        )
        click.echo("This might indicate incorrect git configuration.")

        if not click.confirm("Do you want to continue?"):
            click.echo("Aborted.")
            return

    # Get recent commits
    recent_commits = get_recent_commits(repo, n=5)
    click.echo(f"Found {len(recent_commits)} recent commits")
    if verbose and recent_commits:
        for i, commit in enumerate(recent_commits, 1):
            first_line = commit.split("\n")[0]
            verbose_echo(verbose, f"  Commit {i}: {first_line}")

    # Analyze style


    # Analyze style

    # Get staged diff
    diff = get_staged_diff(repo)
    verbose_echo(verbose, f"Diff length: {len(diff)} characters")

    if not diff.strip():
        click.echo("Error: Could not retrieve staged diff.", err=True)
        raise click.ClickException("Failed to get staged diff")

    # Check for AGENTS.md
    agents_md_content = find_agents_md(repo)
    if agents_md_content:
        click.echo("Found AGENTS.md, including project conventions.")
        verbose_echo(verbose, f"AGENTS.md length: {len(agents_md_content)} characters")
    else:
        verbose_echo(verbose, "No AGENTS.md found")

    # Append AGENTS.md to system prompt
    system_prompt = config.system_prompt
    if agents_md_content:
        system_prompt = system_prompt + "\n\nProject conventions (from AGENTS.md):\n" + agents_md_content

    # Get current branch
    current_branch = get_current_branch(repo)
    if current_branch:
        verbose_echo(verbose, f"Current branch: {current_branch}")
    else:
        verbose_echo(verbose, "No current branch (new repo without commits)")

    # Generate prompt
    prompt = build_prompt(diff, recent_commits)
    if current_branch:
        prompt = f"Current branch: {current_branch}\n\n" + prompt

    verbose_echo(verbose, f"User prompt length: {len(prompt)} characters")
    if verbose:
        click.echo(click.style("[verbose] System prompt:", fg="magenta"))
        click.echo("-" * 40)
        click.echo(system_prompt)
        click.echo("-" * 40)
        click.echo(click.style("[verbose] User prompt:", fg="magenta"))
        click.echo("-" * 40)
        click.echo(prompt)
        click.echo("-" * 40)

    # Create LLM provider
    try:
        llm_provider = create_provider(
            config.provider,
            config.api_key,
            config.model,
            config.api_base or None,
        )
    except Exception as e:
        click.echo(f"Error creating LLM provider: {e}", err=True)
        raise click.ClickException(str(e))

    # Build initial messages
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Tool call loop
    max_tool_calls = 32
    tool_call_count = 0
    retry_count = 0
    max_retries = 3

    click.echo()
    spinner = Spinner(text="Thinking...")
    spinner.start()

    try:
        while tool_call_count < max_tool_calls:
            verbose_echo(verbose, f"Sending request to {config.provider} ({config.model})...")
            if verbose and messages:
                verbose_echo(verbose, f"Messages count: {len(messages)}")

            response = llm_provider.generate(system_prompt, messages)

            spinner.stop()
            verbose_echo(verbose, f"Response received. Tool calls: {len(response.tool_calls)}")
            if response.reasoning_content:
                show_reasoning(response.reasoning_content)
            if response.message:
                click.echo(f"  {click.style('└─', fg='cyan')} {response.message}")
            spinner.start()

            # Check if LLM returned plain text without tool calls
            if not response.tool_calls:
                verbose_echo(verbose, "No tool calls in response, requesting tool use")
                messages.append({
                    "role": "user",
                    "content": "Please use the `message` tool to submit your commit message. Do not return plain text.",
                })
                continue

            tool_call_count += len(response.tool_calls)
            verbose_echo(verbose, f"Total tool calls so far: {tool_call_count}/{max_tool_calls}")

            # Build tool results
            tool_results = []
            read_requests = []

            for tc in response.tool_calls:
                if tc["name"] == "message":
                    message = tc["arguments"].get("message", "")

                    # Validate message is not empty
                    if not message or not message.strip():
                        tool_results.append(
                            ToolResult(
                                tc["id"],
                                "message",
                                "Error: Commit message cannot be empty. Please provide a meaningful commit message.",
                            )
                        )
                        continue

                    # Display the commit message
                    spinner.stop()
                    click.echo("-" * 40)
                    click.echo(message)
                    click.echo("-" * 40)
                    click.echo()

                    if yes:
                        click.echo("Auto-committing (--yes flag set)...")
                        commit_changes(repo, message)
                        return

                    if click.confirm("Do you want to commit with this message?", default=True):
                        commit_changes(repo, message)
                        return
                    else:
                        retry_count += 1
                        if retry_count > max_retries:
                            click.echo("Max retries reached. Commit cancelled.")
                            return

                        feedback = click.prompt(
                            "Why don't you like this message? (Press Enter to skip)",
                            default="",
                            show_default=False,
                        )
                        if not feedback.strip():
                            feedback = "I refuse current commit message"

                        tool_results.append(
                            ToolResult(
                                tc["id"],
                                "message",
                                f"User refused this commit message. Feedback: {feedback}. Please generate a new commit message based on the feedback.",
                            )
                        )

                elif tc["name"] == "read_file":
                    args = tc["arguments"]
                    read_requests.append(
                        {
                            "path": args.get("path"),
                            "start_line": args.get("start_line", 1),
                            "end_line": args.get("end_line"),
                        }
                    )
                    spinner.stop()
                    click.echo(click.style(f"⚙️ Read {args}", fg="bright_black"))
                    spinner.start()

                elif tc["name"] == "grep":
                    args = tc["arguments"]
                    pattern = args.get("pattern", "")
                    include = args.get("include")
                    spinner.stop()
                    click.echo(click.style(f"⚙️ Grep {args}", fg="bright_black"))
                    spinner.start()
                    result = grep_repo(repo, pattern, include)
                    verbose_echo(verbose, f"Grep result length: {len(result)} characters")
                    tool_results.append(
                        ToolResult(
                            tool_call_id=tc["id"],
                            name="grep",
                            content=result,
                        )
                    )

                else:
                    tool_results.append(
                        ToolResult(
                            tool_call_id=tc["id"],
                            name=tc["name"],
                            content=f"Error: Unknown tool '{tc['name']}'. Please use one of the available tools.",
                        )
                    )

            spinner.stop()
            click.echo()
            spinner.start()

            # Read files with budget
            if read_requests:
                results = read_files_batch(repo, read_requests)
                for tc in response.tool_calls:
                    if tc["name"] == "read_file":
                        path = tc["arguments"].get("path")
                        result = results.get(
                            path,
                            {
                                "content": "[Error: unknown file]",
                                "status": "error",
                            },
                        )
                        tool_results.append(
                            ToolResult(
                                tool_call_id=tc["id"],
                                name="read_file",
                                content=result["content"],
                            )
                        )

            # Append assistant's tool calls to messages (preserve message text)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.message,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            # Append tool results
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )

        spinner.stop()
    except click.ClickException:
        spinner.stop()
        raise
    except Exception as e:
        spinner.stop()
        click.echo(f"Error generating commit message: {e}", err=True)
        raise click.ClickException(str(e))

    click.echo("Error: Max tool calls reached without a valid commit message.", err=True)
    raise click.ClickException("Failed to generate commit message")


if __name__ == "__main__":
    main()
