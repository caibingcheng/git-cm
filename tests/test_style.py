"""Tests for commit message style analysis."""

import pytest

from git_cm.style import CONVENTIONAL_PREFIXES, analyze_style


class TestAnalyzeStyle:
    """Test style analysis functionality."""

    def test_empty_commits(self):
        """Test analyzing empty commit list returns default Conventional Commits style."""
        result = analyze_style([])
        
        assert result["avg_length"] == 50
        assert result["uses_prefixes"] is True
        assert result["prefix_pattern"] == "feat"
        assert result["uses_emoji"] is False
        assert result["uses_scope"] is False
        assert result["uses_uppercase"] is False
        assert result["uses_period"] is False
        assert result["sample_commits"] == []
        assert result["is_new_repo"] is True

    def test_conventional_commits(self):
        """Test analyzing conventional commit style."""
        commits = [
            "feat: add user authentication",
            "fix: resolve login redirect issue",
            "docs: update README with examples",
            "feat: implement password reset",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_prefixes"] is True
        assert result["prefix_pattern"] == "feat"
        assert result["uses_scope"] is False
        assert result["uses_emoji"] is False
        assert result["uses_uppercase"] is False
        assert result["uses_period"] is False
        assert result["avg_length"] > 0

    def test_conventional_with_scope(self):
        """Test analyzing commits with scope notation."""
        commits = [
            "feat(auth): add OAuth login",
            "fix(api): resolve rate limiting bug",
            "feat(ui): redesign login page",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_prefixes"] is True
        assert result["uses_scope"] is True
        assert result["prefix_pattern"] == "feat"

    def test_emoji_style(self):
        """Test analyzing emoji commit style."""
        commits = [
            "✨ add new feature",
            "🐛 fix critical bug",
            "📚 update documentation",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_emoji"] is True
        assert result["uses_prefixes"] is False

    def test_freeform_style(self):
        """Test analyzing free-form commit style."""
        commits = [
            "Added user login functionality",
            "Fixed the memory leak in cache",
            "Updated all dependencies to latest versions.",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_prefixes"] is False
        assert result["prefix_pattern"] is None
        assert result["uses_uppercase"] is True
        assert result["uses_period"] is True
        assert result["uses_emoji"] is False

    def test_mixed_style(self):
        """Test analyzing mixed commit styles."""
        commits = [
            "feat: add user authentication",
            "Fix critical bug in payment",
            "docs: update API reference",
            "🚀 deploy to production",
        ]
        
        result = analyze_style(commits)
        
        # Should detect prefixes since some use them
        assert result["uses_prefixes"] is True
        # Should detect emoji
        assert result["uses_emoji"] is True
        # Should detect uppercase (from "Fix critical bug...")
        assert result["uses_uppercase"] is True

    def test_average_length(self):
        """Test average length calculation."""
        commits = [
            "short",
            "medium length message here",
            "this is a much longer commit message with lots of words",
        ]
        
        result = analyze_style(commits)
        
        expected_avg = (5 + 26 + 51) / 3
        assert result["avg_length"] == pytest.approx(expected_avg, 0.1)

    def test_multiline_commits(self):
        """Test that multiline commits only use first line for analysis."""
        commits = [
            "feat: add authentication\n\nThis adds OAuth2 support for login.",
            "fix: resolve bug\n\nDetailed description of the fix.",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_prefixes"] is True
        # Average should be based on first lines only
        first_lines = [c.split("\n")[0] for c in commits]
        expected_avg = sum(len(line) for line in first_lines) / len(first_lines)
        assert result["avg_length"] == pytest.approx(expected_avg, 0.1)

    def test_unknown_prefix(self):
        """Test that unknown prefixes are not reported as pattern."""
        commits = [
            "custom: this is a custom prefix",
            "another: another custom prefix",
        ]
        
        result = analyze_style(commits)
        
        assert result["uses_prefixes"] is True
        # Unknown prefix should not be reported
        assert result["prefix_pattern"] is None

    def test_sample_commits(self):
        """Test that sample commits are included."""
        commits = [
            "feat: add feature 1",
            "fix: fix bug 1",
            "docs: update docs",
            "feat: add feature 2",
        ]
        
        result = analyze_style(commits)
        
        assert len(result["sample_commits"]) == 3
        assert result["sample_commits"][0] == "feat: add feature 1"

    def test_analyze_style_empty_commits_returns_default(self):
        """Test analyzing empty commit list returns default Conventional Commits style."""
        result = analyze_style([])
        
        assert result["avg_length"] == 50
        assert result["uses_prefixes"] is True
        assert result["prefix_pattern"] == "feat"
        assert result["uses_emoji"] is False
        assert result["uses_scope"] is False
        assert result["uses_uppercase"] is False
        assert result["uses_period"] is False
        assert result["sample_commits"] == []
        assert result["is_new_repo"] is True

    def test_analyze_style_non_empty_commits_not_new_repo(self):
        """Test that non-empty commits do not have is_new_repo flag."""
        commits = ["feat: add feature"]
        
        result = analyze_style(commits)
        
        assert result["is_new_repo"] is False
