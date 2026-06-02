"""Tests for prompt generation."""

from git_cm.prompt import build_prompt, build_retry_prompt


class TestBuildPrompt:
    """Test prompt building functionality."""

    def test_basic_prompt(self):
        """Test building a basic prompt."""
        diff = "diff --git a/file.txt b/file.txt\n+new line"
        commits = ["feat: add feature"]
        
        prompt = build_prompt(diff, commits)
        
        assert "Recent commit history:" in prompt
        assert "feat: add feature" in prompt
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
        
        prompt = build_prompt(diff, commits)
        
        # Should include all commits
        assert "feat: add auth" in prompt
        assert "fix: resolve bug" in prompt
        assert "docs: update readme" in prompt

    def test_prompt_limits_commits(self):
        """Test that prompt limits to 5 commits."""
        diff = "+change"
        commits = [f"commit {i}" for i in range(10)]
        
        prompt = build_prompt(diff, commits)
        
        # Should only include first 5 (each commit wrapped in ---)
        separator_count = prompt.count("---")
        # 5 commits = 5 opening --- + 1 closing --- = 6
        assert separator_count == 11

    def test_prompt_multiline_commits(self):
        """Test that multiline commits include full message body."""
        diff = "+change"
        commits = [
            "feat: add feature\n\nDetailed description here",
            "fix: bug fix\n\nMore details",
        ]
        
        prompt = build_prompt(diff, commits)
        
        # Should include full messages (including body)
        assert "feat: add feature" in prompt
        assert "Detailed description here" in prompt
        assert "fix: bug fix" in prompt
        assert "More details" in prompt

    def test_empty_diff(self):
        """Test prompt with empty diff."""
        diff = ""
        commits = ["feat: initial"]
        
        prompt = build_prompt(diff, commits)
        
        assert "```diff" in prompt
        assert "Please generate a commit message" in prompt

    def test_empty_commits(self):
        """Test prompt with no commits."""
        diff = "+change"
        commits = []
        
        prompt = build_prompt(diff, commits)
        
        # Should still work without commit history
        assert "Please generate a commit message" in prompt
        assert "```diff" in prompt
        # Should not have "Recent commit history" section
        assert "Recent commit history:" not in prompt
        # Should have new repo note
        assert "new repository" in prompt.lower()
        assert "Conventional Commits" in prompt


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
