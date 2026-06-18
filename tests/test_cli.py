"""Tests for CLI entry point."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from git import Repo

from git_cm.cli import main, show_reasoning
from git_cm.llm import LLMResponse


def message_tool_response(message: str) -> LLMResponse:
    """Helper to create a message tool call response."""
    return LLMResponse(
        tool_calls=[{
            "id": "call_1",
            "name": "message",
            "arguments": {"message": message},
        }],
    )


@pytest.fixture
def cli_runner():
    """Provide a Click CLI runner."""
    return CliRunner()


@pytest.fixture
def repo_with_changes(temp_git_repo):
    """Create repo with staged changes."""
    repo = Repo(temp_git_repo)

    # Modify file and stage
    test_file = temp_git_repo / "test.txt"
    test_file.write_text("modified content")
    repo.index.add([str(test_file)])

    return temp_git_repo


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary git repository for testing."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    
    repo = Repo.init(repo_path)
    
    # Configure git user
    with repo.config_writer() as config:
        config.set_value("user", "name", "Test User")
        config.set_value("user", "email", "test@example.com")
    
    # Create initial commit
    test_file = repo_path / "test.txt"
    test_file.write_text("initial content")
    repo.index.add([str(test_file)])
    repo.index.commit("initial commit")
    
    return repo_path


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = MagicMock()
    config.is_configured.return_value = True
    config.provider = "openai"
    config.api_key = "test-key"
    config.model = "gpt-4"
    config.api_base = ""
    config.system_prompt = "test prompt"
    return config


class TestCLIBasicChecks:
    """Test basic CLI checks."""

    def test_not_git_repo(self, cli_runner, tmp_path, mock_config):
        """Test error when not in a git repo."""
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = mock_config
            
            with cli_runner.isolated_filesystem(temp_dir=tmp_path):
                result = cli_runner.invoke(main, [])
                
                assert result.exit_code == 1
                assert "Not a git repository" in result.output

    def test_no_staged_changes(self, cli_runner, temp_git_repo, mock_config):
        """Test error when no staged changes."""
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = mock_config
            
            # Change to the repo directory
            old_cwd = os.getcwd()
            os.chdir(str(temp_git_repo))
            try:
                result = cli_runner.invoke(main, [])
                
                assert result.exit_code == 1
                assert "No staged changes" in result.output
            finally:
                os.chdir(old_cwd)


class TestCLIWithStagedChanges:
    """Test CLI with staged changes."""

    def test_shows_user_config(self, cli_runner, repo_with_changes, mock_config):
        """Test that user config is displayed."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: test commit")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "Test User" in result.output
                    assert "test@example.com" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_user_in_history_no_prompt(self, cli_runner, repo_with_changes, mock_config):
        """Test that user in history doesn't prompt for confirmation."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: test commit")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    # Should not show warning about user not in history
                    assert "does not appear in recent commit history" not in result.output
                    assert result.exit_code == 0
                finally:
                    os.chdir(old_cwd)

    def test_user_not_in_history_prompts(self, cli_runner, repo_with_changes, mock_config):
        """Test that user not in history prompts for confirmation."""
        # Change git config to different user
        repo = Repo(repo_with_changes)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Different User")
            config.set_value("user", "email", "different@example.com")
        
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = mock_config
            
            old_cwd = os.getcwd()
            os.chdir(str(repo_with_changes))
            try:
                result = cli_runner.invoke(main, [], input="n\n")
                
                assert "does not appear in recent commit history" in result.output
                assert "Aborted" in result.output
            finally:
                os.chdir(old_cwd)

    def test_generates_message(self, cli_runner, repo_with_changes, mock_config):
        """Test that commit message is generated."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: add new feature")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: add new feature" in result.output
                    assert "Committed:" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_yes_flag_skips_confirmation(self, cli_runner, repo_with_changes, mock_config):
        """Test --yes flag skips confirmation prompt."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: test")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "Auto-committing" in result.output
                    # Should not ask for confirmation
                    assert "Do you want to commit" not in result.output
                finally:
                    os.chdir(old_cwd)

    def test_cancel_commit(self, cli_runner, repo_with_changes, mock_config):
        """Test cancelling commit after retries."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # All calls return message tool
            mock_provider.generate.return_value = message_tool_response("feat: message")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    # Reject 4 times to reach max retries (3 retries = 4 rejections total)
                    result = cli_runner.invoke(main, [], input="n\nn\nn\nn\n")
                    
                    assert result.exit_code == 0
                    assert "Max retries reached" in result.output
                    assert "Commit cancelled" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_retry_max_retries_reached(self, cli_runner, repo_with_changes, mock_config):
        """Test max retries reached cancels commit."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: message")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    # Reject 4 times to reach max retries (3 retries = 4 rejections total)
                    result = cli_runner.invoke(main, [], input="n\nn\nn\nn\n")
                    
                    assert result.exit_code == 0
                    assert "Max retries reached" in result.output
                    assert "Commit cancelled" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_cli_params_override_config(self, cli_runner, repo_with_changes, mock_config):
        """Test CLI parameters override config."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("test")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(
                        main,
                        [
                            "--yes",
                            "--provider", "openai",
                            "--model", "gpt-4o",
                            "--api-key", "test-key",
                        ],
                    )
                    
                    assert result.exit_code == 0
                    # Verify that CLI overrides were set on the config
                    mock_config.set_cli_override.assert_any_call("provider", "openai")
                    mock_config.set_cli_override.assert_any_call("model", "gpt-4o")
                    mock_config.set_cli_override.assert_any_call("api_key", "test-key")
                finally:
                    os.chdir(old_cwd)

    def test_retry_with_feedback(self, cli_runner, repo_with_changes, mock_config):
        """Test retry with user feedback."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: message tool, Second call: improved message
            mock_provider.generate.side_effect = [
                message_tool_response("feat: initial message"),
                message_tool_response("feat: improved message"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    # Input: feedback (reject), y (accept)
                    result = cli_runner.invoke(main, [], input="Make it more detailed\ny\n")
                    
                    assert result.exit_code == 0
                    assert "feat: initial message" in result.output
                    assert "feat: improved message" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_retry_default_feedback(self, cli_runner, repo_with_changes, mock_config):
        """Test retry with default feedback (empty input)."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.side_effect = [
                message_tool_response("feat: initial message"),
                message_tool_response("feat: retry message"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    # Input: n (reject with default feedback), y (accept)
                    result = cli_runner.invoke(main, [], input="n\ny\n")
                    
                    assert result.exit_code == 0
                    assert "feat: retry message" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_tool_call_loop_direct_done(self, cli_runner, repo_with_changes, mock_config):
        """Test LLM directly returns message without tool calls."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: direct text (no tool), Second call: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(message="feat: add feature"),
                message_tool_response("feat: add feature"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: add feature" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_tool_call_loop_read_then_done(self, cli_runner, repo_with_changes, mock_config):
        """Test LLM calls read_file then returns message via tool."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: read_file tool, Second call: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(
                    tool_calls=[{
                        "id": "call_1",
                        "name": "read_file",
                        "arguments": {"path": "test.txt"},
                    }],
                ),
                message_tool_response("feat: update test file"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: update test file" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_show_reasoning_default(self, capsys):
        """Test reasoning content is displayed by default."""
        show_reasoning("analyze diff scope")
        captured = capsys.readouterr()
        assert "Thought:" in captured.out
        assert "analyze diff scope" in captured.out

    def test_show_reasoning_expanded(self, capsys):
        """Test reasoning content expands when env var is set."""
        old_env = os.environ.get("GIT_CM_SHOW_REASONING")
        os.environ["GIT_CM_SHOW_REASONING"] = "1"
        try:
            show_reasoning("analyze diff scope")
            captured = capsys.readouterr()
            assert "analyze diff scope" in captured.out
            assert "Thought:" in captured.out
        finally:
            if old_env is None:
                os.environ.pop("GIT_CM_SHOW_REASONING", None)
            else:
                os.environ["GIT_CM_SHOW_REASONING"] = old_env

    def test_tool_call_message_prefix(self, cli_runner, repo_with_changes, mock_config):
        """Test tool call response message has tree prefix."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: tool call with message, Second call: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(
                    message="Let me check the files.",
                    tool_calls=[{
                        "id": "call_1",
                        "name": "read_file",
                        "arguments": {"path": "test.txt"},
                    }],
                ),
                message_tool_response("feat: update test file"),
            ]
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])

                    assert result.exit_code == 0
                    assert "Response:" in result.output
                    assert "Let me check the files." in result.output
                    assert "feat: update test file" in result.output
                    assert "⚙️ Read" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_interactive_setup_when_not_configured(self, cli_runner, repo_with_changes):
        """Test interactive setup when no config exists."""
        unconfigured_mock = MagicMock()
        unconfigured_mock.is_configured.return_value = False
        unconfigured_mock.provider = "openai"
        unconfigured_mock.api_key = "key"
        unconfigured_mock.model = "gpt-4"
        unconfigured_mock.api_base = ""
        unconfigured_mock.system_prompt = "test prompt"
        
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = unconfigured_mock
            
            with patch("git_cm.cli.interactive_setup") as mock_setup:
                    with patch("git_cm.cli.create_provider") as mock_create:
                        mock_provider = MagicMock()
                        mock_provider.generate.return_value = message_tool_response("feat: test")
                        mock_create.return_value = mock_provider
                    
                    old_cwd = os.getcwd()
                    os.chdir(str(repo_with_changes))
                    try:
                        result = cli_runner.invoke(main, ["--yes"])
                        
                        mock_setup.assert_called_once()
                    finally:
                        os.chdir(old_cwd)

    def test_llm_direct_text_response(self, cli_runner, repo_with_changes, mock_config):
        """Test LLM returns plain text without tool calls gets corrected."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First: direct text, Second: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(message="feat: direct text"),
                message_tool_response("feat: correct message"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: correct message" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_message_tool_empty_message(self, cli_runner, repo_with_changes, mock_config):
        """Test message tool with empty message returns error."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First: empty message, Second: valid message
            mock_provider.generate.side_effect = [
                LLMResponse(
                    tool_calls=[{
                        "id": "call_1",
                        "name": "message",
                        "arguments": {"message": ""},
                    }],
                ),
                message_tool_response("feat: valid message"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: valid message" in result.output
                    assert mock_provider.generate.call_count == 2
                finally:
                    os.chdir(old_cwd)

    def test_unknown_tool_call(self, cli_runner, repo_with_changes, mock_config):
        """Test LLM calls an unknown tool gets an error response."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First: unknown tool, Second: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(
                    tool_calls=[{
                        "id": "call_1",
                        "name": "unknown_tool",
                        "arguments": {"foo": "bar"},
                    }],
                ),
                message_tool_response("feat: valid message"),
            ]
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 0
                    assert "feat: valid message" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2
                    
                    # Verify the error was passed back to LLM
                    # generate(system_prompt, messages) uses positional args
                    second_call_messages = mock_provider.generate.call_args_list[1][0][1]
                    tool_result_msg = next(
                        (m for m in second_call_messages if m.get("role") == "tool"),
                        None,
                    )
                    assert tool_result_msg is not None
                    assert "Error: Unknown tool 'unknown_tool'" in tool_result_msg["content"]
                finally:
                    os.chdir(old_cwd)

class TestCLIErrors:
    """Test CLI error handling."""

    def test_provider_creation_error(self, cli_runner, temp_git_repo, mock_config):
        """Test handling provider creation error."""
        repo = Repo(temp_git_repo)
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified")
        repo.index.add([str(test_file)])
        
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_create.side_effect = ValueError("Invalid provider")
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 1
                    assert "Error creating LLM provider" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_llm_generation_error(self, cli_runner, temp_git_repo, mock_config):
        """Test handling LLM generation error."""
        repo = Repo(temp_git_repo)
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified")
        repo.index.add([str(test_file)])
        
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.side_effect = Exception("API Error")
            mock_create.return_value = mock_provider
            
            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config
                
                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["--yes"])
                    
                    assert result.exit_code == 1
                    assert "Error generating commit message" in result.output
                finally:
                    os.chdir(old_cwd)

    def test_empty_diff(self, cli_runner, temp_git_repo, mock_config):
        """Test handling empty diff."""
        # Create an empty file and stage it
        repo = Repo(temp_git_repo)
        empty_file = temp_git_repo / "empty.txt"
        empty_file.write_text("")
        repo.index.add([str(empty_file)])

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: add empty file")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["--yes"])

                    # Empty file might still have a diff
                    assert result.exit_code in [0, 1]
                finally:
                    os.chdir(old_cwd)

    def test_stash_restore_failure_raises(self, cli_runner, temp_git_repo, mock_config):
        """Test stash restore failure raises ClickException with recovery instructions."""
        repo = Repo(temp_git_repo)
        new_file = temp_git_repo / "feature.txt"
        new_file.write_text("feature content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")
        new_file.write_text("uncommitted change")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten head")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                with patch("git_cm.cli.pop_stash") as mock_pop_stash:
                    mock_pop_stash.side_effect = RuntimeError("stash top changed")

                    old_cwd = os.getcwd()
                    os.chdir(str(temp_git_repo))
                    try:
                        result = cli_runner.invoke(main, ["-r", "--yes"])

                        assert result.exit_code != 0
                        assert "Failed to restore stash" in result.output
                        assert "git stash list" in result.output
                        assert "git stash pop" in result.output
                    finally:
                        os.chdir(old_cwd)


class TestDiffMore:
    """Test diff_more tool functionality."""

    def test_diff_more_returns_next_chunk(self, cli_runner, repo_with_changes, mock_config):
        """Test diff_more returns the next chunk of diff."""
        # Create a large diff that will be chunked
        repo = Repo(repo_with_changes)
        for i in range(2000):
            (repo_with_changes / f"file_{i}.txt").write_text(f"content {i}\n")
            repo.index.add([str(repo_with_changes / f"file_{i}.txt")])

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: diff_more tool, Second call: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(
                    tool_calls=[{
                        "id": "call_1",
                        "name": "diff_more",
                        "arguments": {},
                    }],
                ),
                message_tool_response("feat: add many files"),
            ]
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])

                    assert result.exit_code == 0
                    assert "feat: add many files" in result.output
                    assert "Committed:" in result.output
                    assert mock_provider.generate.call_count == 2

                    # Verify chunk 0 was in initial messages
                    first_call_messages = mock_provider.generate.call_args_list[0][0][1]
                    chunk0_msg = next(
                        (m for m in first_call_messages if m.get("role") == "user" and "[Diff chunk:" in m.get("content", "")),
                        None,
                    )
                    assert chunk0_msg is not None
                    assert "[Diff chunk: total=" in chunk0_msg["content"]
                    assert "current_index=0" in chunk0_msg["content"]

                    # Verify the diff_more result (chunk 1) was passed to LLM
                    second_call_messages = mock_provider.generate.call_args_list[1][0][1]
                    tool_result_msg = next(
                        (m for m in second_call_messages if m.get("role") == "tool"),
                        None,
                    )
                    assert tool_result_msg is not None
                    assert "[Diff chunk: total=" in tool_result_msg["content"]
                    assert "current_index=1" in tool_result_msg["content"]
                finally:
                    os.chdir(old_cwd)

    def test_diff_more_exhausted(self, cli_runner, repo_with_changes, mock_config):
        """Test diff_more when no more content is available."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            # First call: diff_more tool (exhausted), Second call: message tool
            mock_provider.generate.side_effect = [
                LLMResponse(
                    tool_calls=[{
                        "id": "call_1",
                        "name": "diff_more",
                        "arguments": {},
                    }],
                ),
                message_tool_response("feat: test commit"),
            ]
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])

                    assert result.exit_code == 0
                    assert "feat: test commit" in result.output
                    # Second call should get "No more diff content available"
                    second_call_messages = mock_provider.generate.call_args_list[1][0][1]
                    tool_result_msg = next(
                        (m for m in second_call_messages if m.get("role") == "tool"),
                        None,
                    )
                    assert tool_result_msg is not None
                    assert "No more diff content available" in tool_result_msg["content"]
                    assert "Total chunks:" in tool_result_msg["content"]
                finally:
                    os.chdir(old_cwd)

    def test_large_diff_warning(self, cli_runner, repo_with_changes, mock_config):
        """Test warning for very large diffs (>100000 chars)."""
        # Create a very large diff
        repo = Repo(repo_with_changes)
        large_content = "x" * 200000
        (repo_with_changes / "large_file.txt").write_text(large_content)
        repo.index.add([str(repo_with_changes / "large_file.txt")])

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: add large file")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes"])

                    assert result.exit_code == 0
                    assert "Warning: Staged diff is very large" in result.output
                    assert "Consider splitting into smaller commits" in result.output
                finally:
                    os.chdir(old_cwd)


class TestCLIRewriteAndHint:
    """Test rewrite mode and user hint features."""

    def test_hint_in_prompt(self, cli_runner, repo_with_changes, mock_config):
        """Test user hint is included in the prompt."""
        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: hinted")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(repo_with_changes))
                try:
                    result = cli_runner.invoke(main, ["--yes", "some hint"])

                    assert result.exit_code == 0
                    first_call_messages = mock_provider.generate.call_args_list[0][0][1]
                    user_msg = first_call_messages[0]["content"]
                    assert "<user_hint>" in user_msg
                    assert "some hint" in user_msg
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_head(self, cli_runner, temp_git_repo, mock_config):
        """Test rewriting HEAD commit message."""
        repo = Repo(temp_git_repo)
        new_file = temp_git_repo / "feature.txt"
        new_file.write_text("feature content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten head")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["-r", "--yes"])

                    assert result.exit_code == 0
                    assert "Rewrote commit message: feat: rewritten head" in result.output
                    assert repo.head.commit.message.strip() == "feat: rewritten head"
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_head_stashes_and_restores_changes(self, cli_runner, temp_git_repo, mock_config):
        """Test rewriting HEAD stashes and restores uncommitted changes."""
        repo = Repo(temp_git_repo)
        new_file = temp_git_repo / "feature.txt"
        new_file.write_text("feature content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")
        new_file.write_text("uncommitted change")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten head")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["-r", "--yes"])

                    assert result.exit_code == 0
                    assert new_file.read_text() == "uncommitted change"
                    assert repo.is_dirty(untracked_files=True)
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_specific_commit_warning(self, cli_runner, temp_git_repo, mock_config):
        """Test rewriting a historical commit shows warning and rewrites."""
        repo = Repo(temp_git_repo)
        for i in range(2):
            f = temp_git_repo / f"file{i}.txt"
            f.write_text(f"content {i}")
            repo.index.add([str(f)])
            repo.index.commit(f"commit {i}")

        commits = list(repo.iter_commits("HEAD", max_count=10))
        target = commits[1]

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["-r", target.hexsha, "--yes"], input="y\n")

                    assert result.exit_code == 0
                    assert "will rewrite a historical commit" in result.output
                    new_commits = list(repo.iter_commits("HEAD", max_count=10))
                    messages = [c.message.strip() for c in new_commits]
                    assert "feat: rewritten" in messages
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_pushed_commit_warning(self, cli_runner, temp_git_repo, mock_config):
        """Test rewriting a pushed commit shows warning."""
        repo = Repo(temp_git_repo)
        remote_path = temp_git_repo.parent / "remote.git"
        remote_path.mkdir()
        Repo.init(remote_path, bare=True)
        repo.create_remote("origin", str(remote_path))
        repo.git.push("-u", "origin", "master")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten pushed")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["-r", "--yes"], input="y\n")

                    assert result.exit_code == 0
                    assert "appears to have been pushed" in result.output
                    assert repo.head.commit.message.strip() == "feat: rewritten pushed"
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_with_hint(self, cli_runner, temp_git_repo, mock_config):
        """Test rewrite mode accepts a hint and includes it in prompt."""
        repo = Repo(temp_git_repo)
        new_file = temp_git_repo / "feature.txt"
        new_file.write_text("feature content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: concise")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(main, ["-r", "--yes", "make it concise"])

                    assert result.exit_code == 0
                    first_call_messages = mock_provider.generate.call_args_list[0][0][1]
                    user_msg = first_call_messages[0]["content"]
                    assert "make it concise" in user_msg
                    assert "<rewrite_context>" in user_msg
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_with_to_option(self, cli_runner, temp_git_repo, mock_config):
        """Test --to option explicitly selects the commit to rewrite."""
        repo = Repo(temp_git_repo)
        for i in range(2):
            f = temp_git_repo / f"file{i}.txt"
            f.write_text(f"content {i}")
            repo.index.add([str(f)])
            repo.index.commit(f"commit {i}")

        commits = list(repo.iter_commits("HEAD", max_count=10))
        target = commits[1]

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(
                        main, ["-r", "-t", target.hexsha, "--yes"], input="y\n"
                    )

                    assert result.exit_code == 0
                    assert "will rewrite a historical commit" in result.output
                    new_commits = list(repo.iter_commits("HEAD", max_count=10))
                    messages = [c.message.strip() for c in new_commits]
                    assert "feat: rewritten" in messages
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_with_hint_option(self, cli_runner, temp_git_repo, mock_config):
        """Test --hint option provides a user hint in rewrite mode."""
        repo = Repo(temp_git_repo)
        new_file = temp_git_repo / "feature.txt"
        new_file.write_text("feature content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: concise")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(
                        main, ["-r", "-m", "make it concise", "--yes"]
                    )

                    assert result.exit_code == 0
                    first_call_messages = mock_provider.generate.call_args_list[0][0][1]
                    user_msg = first_call_messages[0]["content"]
                    assert "make it concise" in user_msg
                    assert "<rewrite_context>" in user_msg
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_to_and_positional_hint(self, cli_runner, temp_git_repo, mock_config):
        """Test --to with a positional hint rewrites target and includes hint."""
        repo = Repo(temp_git_repo)
        for i in range(2):
            f = temp_git_repo / f"file{i}.txt"
            f.write_text(f"content {i}")
            repo.index.add([str(f)])
            repo.index.commit(f"commit {i}")

        commits = list(repo.iter_commits("HEAD", max_count=10))
        target = commits[1]

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(
                        main,
                        ["-r", "-t", target.hexsha, "--yes", "some hint"],
                        input="y\n",
                    )

                    assert result.exit_code == 0
                    new_commits = list(repo.iter_commits("HEAD", max_count=10))
                    messages = [c.message.strip() for c in new_commits]
                    assert "feat: rewritten" in messages

                    first_call_messages = mock_provider.generate.call_args_list[0][0][1]
                    user_msg = first_call_messages[0]["content"]
                    assert "some hint" in user_msg
                    assert "<rewrite_context>" in user_msg
                finally:
                    os.chdir(old_cwd)

    def test_rewrite_invalid_commit_id(self, cli_runner, temp_git_repo, mock_config):
        """Test --to with an invalid commit id exits with an error."""
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = mock_config

            old_cwd = os.getcwd()
            os.chdir(str(temp_git_repo))
            try:
                result = cli_runner.invoke(
                    main, ["-r", "-t", "notacommit", "--yes"]
                )

                assert result.exit_code != 0
                assert "Invalid commit" in result.output
            finally:
                os.chdir(old_cwd)

    def test_rewrite_positional_commit_id_backward_compat(
        self, cli_runner, temp_git_repo, mock_config
    ):
        """Test positional commit id is treated as rewrite target for backward compat."""
        repo = Repo(temp_git_repo)
        for i in range(2):
            f = temp_git_repo / f"file{i}.txt"
            f.write_text(f"content {i}")
            repo.index.add([str(f)])
            repo.index.commit(f"commit {i}")

        commits = list(repo.iter_commits("HEAD", max_count=10))
        target = commits[1]

        with patch("git_cm.cli.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.generate.return_value = message_tool_response("feat: rewritten")
            mock_create.return_value = mock_provider

            with patch("git_cm.cli.Config") as mock_config_class:
                mock_config_class.return_value = mock_config

                old_cwd = os.getcwd()
                os.chdir(str(temp_git_repo))
                try:
                    result = cli_runner.invoke(
                        main, ["-r", target.hexsha, "--yes"], input="y\n"
                    )

                    assert result.exit_code == 0
                    new_commits = list(repo.iter_commits("HEAD", max_count=10))
                    messages = [c.message.strip() for c in new_commits]
                    assert "feat: rewritten" in messages
                finally:
                    os.chdir(old_cwd)

    def test_hint_and_positional_hint_conflict(self, cli_runner, temp_git_repo, mock_config):
        """Test providing both --hint and a positional hint is rejected."""
        with patch("git_cm.cli.Config") as mock_config_class:
            mock_config_class.return_value = mock_config

            old_cwd = os.getcwd()
            os.chdir(str(temp_git_repo))
            try:
                result = cli_runner.invoke(
                    main, ["--yes", "-m", "hint", "positional"]
                )

                assert result.exit_code != 0
                assert "Cannot provide both --hint and a positional hint" in result.output
            finally:
                os.chdir(old_cwd)
