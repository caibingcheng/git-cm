"""Tests for prompt generation."""

import pytest

from git_cm.prompt import build_prompt, build_retry_prompt
from git_cm.style import analyze_style


class TestBuildPrompt:
    """Test prompt building functionality."""

    def test_basic_prompt(self):
        """Test building a basic prompt."""
        diff = "diff --git a/file.txt b/file.txt\n+new line"
        commits = ["feat: add feature"]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        assert "Recent commit history:" in prompt
        assert "feat: add feature" in prompt
        assert "Style notes:" in prompt
        assert "```diff" in prompt
        assert "new line" in prompt
        assert "```" in prompt

    def test_prompt_with_multiple_commits(self):
        """Test prompt with multiple recent commits."""
        diff = "+added line"
        commits = [
            "feat: add auth",
            "fix: resolve bug",
            "docs: update readme",
        ]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        # Should include all commits
        assert "feat: add auth" in prompt
        assert "fix: resolve bug" in prompt
        assert "docs: update readme" in prompt
        
        # Should mention conventional commits
        assert "Uses conventional commit prefixes" in prompt

    def test_prompt_limits_commits(self):
        """Test that prompt limits to 5 commits."""
        diff = "+change"
        commits = [f"commit {i}" for i in range(10)]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        # Should only include first 5 (each commit wrapped in ---)
        separator_count = prompt.count("---")
        # 5 commits = 5 opening --- + 1 closing --- = 6
        assert separator_count == 6

    def test_prompt_multiline_commits(self):
        """Test that multiline commits include full message body."""
        diff = "+change"
        commits = [
            "feat: add feature\n\nDetailed description here",
            "fix: bug fix\n\nMore details",
        ]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        # Should include full messages (including body)
        assert "feat: add feature" in prompt
        assert "Detailed description here" in prompt
        assert "fix: bug fix" in prompt
        assert "More details" in prompt

    def test_prompt_style_notes(self):
        """Test that style notes are included."""
        diff = "+change"
        commits = ["feat: add auth"]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        # Should have style-related notes
        assert "concise" in prompt or "length" in prompt
        assert "lowercase" in prompt or "uppercase" in prompt
        assert "period" in prompt

    def test_empty_diff(self):
        """Test prompt with empty diff."""
        diff = ""
        commits = ["feat: initial"]
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        assert "```diff" in prompt
        assert "Please generate a commit message" in prompt

    def test_empty_commits(self):
        """Test prompt with no commits."""
        diff = "+change"
        commits = []
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        # Should still work without commit history
        assert "Please generate a commit message" in prompt
        assert "```diff" in prompt
        # Should not have "Recent commit history" section
        assert "Recent commit history:" not in prompt


class TestBuildRetryPrompt:
    """Test retry prompt building."""

    def test_build_retry_prompt_includes_feedback(self):
        """Test that retry prompt includes feedback text."""
        original = "Original prompt"
        previous = "feat: old message"
        feedback = "Make it more detailed"
        
        result = build_retry_prompt(original, previous, feedback)
        
        assert original in result
        assert "Previous attempt:" in result
        assert previous in result
        assert f"User feedback: {feedback}" in result
        assert "Please generate a new commit message" in result

    def test_build_retry_prompt_default_feedback(self):
        """Test retry prompt with default feedback."""
        original = "Original prompt"
        previous = "feat: old message"
        feedback = "用户不接受当前的 commit message"
        
        result = build_retry_prompt(original, previous, feedback)
        
        assert f"User feedback: {feedback}" in result


class TestNewRepoPrompt:
    """Test new repository prompt notes."""

    def test_build_prompt_new_repo_notes(self):
        """Test that new repo prompt includes Conventional Commits hint."""
        diff = "+new line"
        commits = []
        style = {
            "avg_length": 50,
            "uses_prefixes": True,
            "prefix_pattern": "feat",
            "uses_emoji": False,
            "uses_scope": False,
            "uses_uppercase": False,
            "uses_period": False,
            "sample_commits": [],
            "is_new_repo": True,
        }
        
        prompt = build_prompt(diff, commits, style)
        
        assert "new repository" in prompt.lower()
        assert "Conventional Commits" in prompt
        assert "Style notes:" in prompt
        assert "Uses conventional commit prefixes" in prompt


class TestAgentsMdPrompt:
    """Test AGENTS.md prompt integration."""

    def test_build_prompt_with_agents_md(self):
        """Test that AGENTS.md content appears in prompt."""
        diff = "+change"
        commits = []
        style = analyze_style(commits)
        agents_md = "# Project Conventions\n\nUse semantic versioning."
        
        prompt = build_prompt(diff, commits, style, agents_md)
        
        assert "Project conventions (from AGENTS.md):" in prompt
        assert "Use semantic versioning" in prompt
        assert "```" in prompt

    def test_build_prompt_without_agents_md(self):
        """Test that empty AGENTS.md doesn't appear in prompt."""
        diff = "+change"
        commits = []
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style, "")
        
        assert "Project conventions (from AGENTS.md):" not in prompt

    def test_build_prompt_agents_md_default_empty(self):
        """Test that default agents_md_content is empty string."""
        diff = "+change"
        commits = []
        style = analyze_style(commits)
        
        prompt = build_prompt(diff, commits, style)
        
        assert "Project conventions (from AGENTS.md):" not in prompt
