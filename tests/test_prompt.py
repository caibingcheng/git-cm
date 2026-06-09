"""Tests for prompt generation."""

from git_cm.prompt import build_prompt, build_retry_prompt, chunk_diff, format_diff_chunk


class TestBuildPrompt:
    """Test prompt building functionality."""

    def test_basic_prompt(self):
        """Test building a basic prompt."""
        commits = ["feat: add feature"]

        prompt = build_prompt(commits)

        assert "<recent_commits>" in prompt
        assert '<commit index="1">' in prompt
        assert "feat: add feature" in prompt
        assert "<instruction>" in prompt
        assert "1 chunk" in prompt
        # Diff should NOT be in prompt
        assert "<diff>" not in prompt
        assert "<![CDATA[" not in prompt

    def test_prompt_with_multiple_commits(self):
        """Test prompt with multiple recent commits."""
        commits = [
            "feat: add auth",
            "fix: resolve bug",
            "docs: update readme",
        ]

        prompt = build_prompt(commits)

        # Should include all commits
        assert "feat: add auth" in prompt
        assert "fix: resolve bug" in prompt
        assert "docs: update readme" in prompt

    def test_prompt_limits_commits(self):
        """Test that prompt limits to 5 commits."""
        commits = [f"commit {i}" for i in range(10)]

        prompt = build_prompt(commits)

        # Should only include first 5
        commit_count = prompt.count("<commit ")
        assert commit_count == 5

    def test_prompt_multiline_commits(self):
        """Test that multiline commits include full message body."""
        commits = [
            "feat: add feature\n\nDetailed description here",
            "fix: bug fix\n\nMore details",
        ]

        prompt = build_prompt(commits)

        # Should include full messages (including body)
        assert "feat: add feature" in prompt
        assert "Detailed description here" in prompt
        assert "fix: bug fix" in prompt
        assert "More details" in prompt

    def test_empty_commits(self):
        """Test prompt with no commits."""
        commits = []

        prompt = build_prompt(commits)

        # Should still work without commit history
        assert "<instruction>" in prompt
        # Should not have commit tags
        assert "<commit " not in prompt
        # Should have new repo note
        assert "new repository" in prompt.lower()
        assert "Conventional Commits" in prompt

    def test_total_chunks_single(self):
        """Test that single chunk info is shown."""
        commits = ["feat: add feature"]

        prompt = build_prompt(commits, total_chunks=1)

        assert "1 chunk" in prompt
        assert "diff_more" not in prompt.lower()

    def test_total_chunks_multiple(self):
        """Test that multiple chunks info includes diff_more hint."""
        commits = ["feat: add feature"]

        prompt = build_prompt(commits, total_chunks=3)

        assert "3 chunk(s)" in prompt
        assert "diff_more" in prompt.lower()

    def test_files_info(self):
        """Test that files section is included when files_info is provided."""
        commits = []
        files_info = [
            {"path": "src/main.py", "status": "modified", "is_binary": "false"},
            {"path": "assets/logo.png", "status": "added", "is_binary": "true"},
        ]

        prompt = build_prompt(commits, files_info=files_info)

        assert "<files>" in prompt
        assert 'path="src/main.py"' in prompt
        assert 'status="modified"' in prompt
        assert 'type="text"' in prompt
        assert 'path="assets/logo.png"' in prompt
        assert 'status="added"' in prompt
        assert 'type="binary"' in prompt
        assert "</files>" in prompt

    def test_no_files_info(self):
        """Test that files section shows placeholder when no files_info."""
        commits = []

        prompt = build_prompt(commits)

        assert "<files>" in prompt
        assert "No file information available" in prompt
        assert "</files>" in prompt


class TestFormatDiffChunk:
    """Test diff chunk formatting."""

    def test_single_chunk(self):
        """Test formatting a single chunk."""
        chunk = "+line1\n+line2"
        result = format_diff_chunk(chunk, total_chunks=1, current_index=0)

        assert "[Diff chunk: total=1, current_index=0]" in result
        assert "```diff" in result
        assert "+line1" in result
        assert "+line2" in result
        assert "```" in result
        assert "[This is the last chunk.]" in result

    def test_first_of_multiple(self):
        """Test formatting the first chunk of multiple."""
        chunk = "+line1"
        result = format_diff_chunk(chunk, total_chunks=3, current_index=0)

        assert "[Diff chunk: total=3, current_index=0]" in result
        assert "[More chunks available" in result
        assert "This is the last chunk" not in result

    def test_last_of_multiple(self):
        """Test formatting the last chunk."""
        chunk = "+line3"
        result = format_diff_chunk(chunk, total_chunks=3, current_index=2)

        assert "[Diff chunk: total=3, current_index=2]" in result
        assert "[This is the last chunk.]" in result
        assert "More chunks available" not in result

    def test_middle_chunk(self):
        """Test formatting a middle chunk."""
        chunk = "+line2"
        result = format_diff_chunk(chunk, total_chunks=3, current_index=1)

        assert "[Diff chunk: total=3, current_index=1]" in result
        assert "[More chunks available" in result


class TestChunkDiff:
    """Test diff chunking functionality."""

    def test_small_diff_single_chunk(self):
        """Test that small diff fits in a single chunk."""
        diff = "+line1\n+line2\n+line3"
        chunks = chunk_diff(diff, chunk_size=100)

        assert len(chunks) == 1
        assert chunks[0] == diff + "\n"

    def test_large_diff_multiple_chunks(self):
        """Test that large diff is split into multiple chunks."""
        lines = [f"+line{i}" for i in range(100)]
        diff = "\n".join(lines)
        chunks = chunk_diff(diff, chunk_size=200)

        assert len(chunks) > 1

    def test_chunk_size_limit(self):
        """Test that no chunk exceeds the size limit."""
        lines = [f"+line{i}" for i in range(100)]
        diff = "\n".join(lines)
        chunks = chunk_diff(diff, chunk_size=200)

        for chunk in chunks:
            assert len(chunk) <= 200

    def test_chunks_are_contiguous(self):
        """Test that chunks are contiguous (no gaps, no overlap)."""
        lines = [f"+line{i}" for i in range(50)]
        diff = "\n".join(lines)
        chunks = chunk_diff(diff, chunk_size=150)

        # Reconstruct the original diff by joining chunks
        # Note: chunk_diff preserves \n at end of each line, so joining chunks
        # directly reconstructs the original with trailing newlines
        reconstructed = "".join(chunks)
        # Remove trailing newline that chunk_diff adds
        assert reconstructed.rstrip("\n") == diff

    def test_empty_diff(self):
        """Test chunking empty diff."""
        chunks = chunk_diff("")
        assert len(chunks) == 1
        # Empty string split by \n produces [''] which gets joined with \n
        assert chunks[0] == "\n"


class TestBuildRetryPrompt:
    """Test retry prompt building."""

    def test_build_retry_prompt_includes_feedback(self):
        """Test that retry prompt includes feedback text in XML format."""
        original = "Original prompt"
        previous = "feat: old message"
        feedback = "Make it more detailed"

        result = build_retry_prompt(original, previous, feedback)

        assert original in result
        assert "<retry>" in result
        assert "<previous_attempt>" in result
        assert previous in result
        assert "<feedback>" in result
        assert feedback in result
        assert "Please generate a new commit message" in result

    def test_build_retry_prompt_default_feedback(self):
        """Test retry prompt with default feedback."""
        original = "Original prompt"
        previous = "feat: old message"
        feedback = "用户不接受当前的 commit message"

        result = build_retry_prompt(original, previous, feedback)

        assert "<retry>" in result
        assert feedback in result
        assert "</feedback>" in result
