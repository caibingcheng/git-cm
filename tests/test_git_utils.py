"""Tests for git operations."""

import os
from pathlib import Path

import git
import pytest
from git import Repo

from git_cm.git_utils import (
    get_current_branch,
    check_user_in_history,
    commit_changes,
    find_agents_md,
    get_recent_commits,
    get_repo,
    get_staged_diff,
    get_user_config,
    grep_repo,
    has_staged_changes,
    is_git_repo,
    read_file,
    read_files_batch,
)


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


class TestGitRepoChecks:
    """Test git repository detection."""

    def test_is_git_repo_true(self, temp_git_repo):
        """Test detecting a valid git repo."""
        assert is_git_repo(temp_git_repo) is True

    def test_is_git_repo_false(self, tmp_path):
        """Test detecting a non-git directory."""
        non_git_dir = tmp_path / "not-a-repo"
        non_git_dir.mkdir()
        assert is_git_repo(non_git_dir) is False

    def test_get_repo(self, temp_git_repo):
        """Test getting repo object."""
        repo = get_repo(temp_git_repo)
        assert isinstance(repo, Repo)


class TestStagedChanges:
    """Test staged changes detection."""

    def test_no_staged_changes(self, temp_git_repo):
        """Test when there are no staged changes."""
        repo = get_repo(temp_git_repo)
        assert has_staged_changes(repo) is False

    def test_has_staged_changes(self, temp_git_repo):
        """Test when there are staged changes."""
        repo = get_repo(temp_git_repo)
        
        # Modify file and stage it
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified content")
        repo.index.add([str(test_file)])
        
        assert has_staged_changes(repo) is True


class TestUserConfig:
    """Test user configuration retrieval."""

    def test_get_user_config(self, temp_git_repo):
        """Test reading user name and email."""
        repo = get_repo(temp_git_repo)
        config = get_user_config(repo)
        
        assert config["name"] == "Test User"
        assert config["email"] == "test@example.com"


class TestCommitHistory:
    """Test commit history operations."""

    def test_get_recent_commits(self, temp_git_repo):
        """Test getting recent commit messages."""
        repo = get_repo(temp_git_repo)
        commits = get_recent_commits(repo, n=5)
        
        assert len(commits) == 1
        assert commits[0] == "initial commit"

    def test_get_recent_commits_multiple(self, temp_git_repo):
        """Test getting multiple recent commits."""
        repo = get_repo(temp_git_repo)
        
        # Create additional commits
        for i in range(3):
            test_file = temp_git_repo / f"file{i}.txt"
            test_file.write_text(f"content {i}")
            repo.index.add([str(test_file)])
            repo.index.commit(f"commit {i + 1}")
        
        commits = get_recent_commits(repo, n=5)
        
        assert len(commits) == 4  # initial + 3 new
        assert commits[0] == "commit 3"
        assert commits[3] == "initial commit"

    def test_check_user_in_history_true(self, temp_git_repo):
        """Test finding user in history."""
        repo = get_repo(temp_git_repo)
        result = check_user_in_history(repo, "Test User", "test@example.com")
        
        assert result is True

    def test_check_user_in_history_false(self, temp_git_repo):
        """Test not finding user in history."""
        repo = get_repo(temp_git_repo)
        result = check_user_in_history(repo, "Unknown User", "unknown@example.com")
        
        assert result is False


class TestStagedDiff:
    """Test staged diff retrieval."""

    def test_get_staged_diff(self, temp_git_repo):
        """Test getting staged diff."""
        repo = get_repo(temp_git_repo)
        
        # Modify file and stage it
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified content")
        repo.index.add([str(test_file)])
        
        diff = get_staged_diff(repo)
        
        assert "modified content" in diff
        assert "initial content" in diff

    def test_get_staged_diff_empty(self, temp_git_repo):
        """Test getting diff when nothing is staged."""
        repo = get_repo(temp_git_repo)
        diff = get_staged_diff(repo)
        
        # May be empty or contain diff stats
        assert isinstance(diff, str)


class TestCommit:
    """Test committing changes."""

    def test_commit_changes(self, temp_git_repo):
        """Test committing staged changes."""
        repo = get_repo(temp_git_repo)
        
        # Stage some changes
        test_file = temp_git_repo / "new_file.txt"
        test_file.write_text("new content")
        repo.index.add([str(test_file)])
        
        # Commit
        commit_changes(repo, "feat: add new file")
        
        # Verify commit was made
        latest_commit = list(repo.iter_commits("HEAD", max_count=1))[0]
        assert latest_commit.message.strip() == "feat: add new file"


class TestReadFile:
    """Test file reading functionality."""

    def test_read_file_success(self, temp_git_repo):
        """Test normal file reading."""
        repo = get_repo(temp_git_repo)
        
        # Create a test file
        test_file = temp_git_repo / "src" / "main.py"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("line1\nline2\nline3\n")
        
        result = read_file(repo, "src/main.py")
        
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_file_with_lines(self, temp_git_repo):
        """Test reading specific line range."""
        repo = get_repo(temp_git_repo)
        
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")
        
        result = read_file(repo, "test.txt", start_line=2, end_line=4)
        
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result
        assert "line1" not in result
        assert "line5" not in result
        assert "Lines 2-4 of 5" in result

    def test_read_file_outside_repo(self, temp_git_repo):
        """Test path outside repo returns security error."""
        repo = get_repo(temp_git_repo)
        
        result = read_file(repo, "../outside.txt")
        
        assert "Security error" in result

    def test_read_file_not_found(self, temp_git_repo):
        """Test file not found returns error."""
        repo = get_repo(temp_git_repo)
        
        result = read_file(repo, "nonexistent.py")
        
        assert "not found" in result

    def test_read_file_line_out_of_range(self, temp_git_repo):
        """Test line out of range."""
        repo = get_repo(temp_git_repo)
        
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("line1\nline2\n")
        
        result = read_file(repo, "test.txt", start_line=10)
        
        assert "exceeds file length" in result


class TestReadFilesBatch:
    """Test batch file reading."""

    def test_read_files_batch_ok(self, temp_git_repo):
        """Test normal batch reading."""
        repo = get_repo(temp_git_repo)
        
        test_file1 = temp_git_repo / "file1.txt"
        test_file1.write_text("content1")
        test_file2 = temp_git_repo / "file2.txt"
        test_file2.write_text("content2")
        
        requests = [
            {"path": "file1.txt"},
            {"path": "file2.txt"},
        ]
        
        results = read_files_batch(repo, requests, max_total_chars=1000)
        
        assert results["file1.txt"]["status"] == "ok"
        assert "content1" in results["file1.txt"]["content"]
        assert results["file2.txt"]["status"] == "ok"
        assert "content2" in results["file2.txt"]["content"]

    def test_read_files_batch_truncated(self, temp_git_repo):
        """Test cumulative budget truncation."""
        repo = get_repo(temp_git_repo)
        
        test_file = temp_git_repo / "large.txt"
        test_file.write_text("x" * 500)
        
        requests = [
            {"path": "large.txt"},
        ]
        
        results = read_files_batch(repo, requests, max_total_chars=250)
        
        assert results["large.txt"]["status"] == "truncated"
        assert "truncated from" in results["large.txt"]["content"]

    def test_read_files_batch_skipped(self, temp_git_repo):
        """Test budget exhausted skips remaining."""
        repo = get_repo(temp_git_repo)
        
        test_file1 = temp_git_repo / "file1.txt"
        test_file1.write_text("x" * 200)
        test_file2 = temp_git_repo / "file2.txt"
        test_file2.write_text("content2")
        
        requests = [
            {"path": "file1.txt"},
            {"path": "file2.txt"},
        ]
        
        results = read_files_batch(repo, requests, max_total_chars=150)
        
        assert results["file1.txt"]["status"] in ["ok", "truncated"]
        assert results["file2.txt"]["status"] == "skipped"
        assert "budget exhausted" in results["file2.txt"]["content"]


class TestFindAgentsMd:
    """Test AGENTS.md finding."""

    def test_find_agents_md_exists(self, temp_git_repo):
        """Test finding existing AGENTS.md."""
        repo = get_repo(temp_git_repo)
        
        agents_file = temp_git_repo / "AGENTS.md"
        agents_file.write_text("# Project Conventions\n\nUse semantic versioning.")
        
        result = find_agents_md(repo)
        
        assert "Project Conventions" in result
        assert "semantic versioning" in result

    def test_find_agents_md_not_exists(self, temp_git_repo):
        """Test missing AGENTS.md returns empty string."""
        repo = get_repo(temp_git_repo)
        
        result = find_agents_md(repo)
        
        assert result == ""

    def test_find_agents_md_truncated(self, temp_git_repo):
        """Test large AGENTS.md is truncated."""
        repo = get_repo(temp_git_repo)
        
        agents_file = temp_git_repo / "AGENTS.md"
        agents_file.write_text("x" * 4000)
        
        result = find_agents_md(repo)
        
        assert len(result) <= 3016  # 3000 + truncation message
        assert "[... truncated]" in result


class TestGrepRepo:
    """Test grep_repo functionality."""

    def test_grep_repo_found(self, temp_git_repo):
        """Test finding files matching pattern."""
        repo = get_repo(temp_git_repo)

        # Create test files with known content
        src_dir = temp_git_repo / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def hello(): pass\n")
        (src_dir / "utils.py").write_text("def helper(): pass\n")

        result = grep_repo(repo, "hello")
        assert "src/main.py" in result
        assert "src/utils.py" not in result

    def test_grep_repo_no_match(self, temp_git_repo):
        """Test no matches returns empty string."""
        repo = get_repo(temp_git_repo)

        result = grep_repo(repo, "nonexistent_pattern_xyz")
        assert result == ""

    def test_grep_repo_include_filter(self, temp_git_repo):
        """Test include filter restricts search."""
        repo = get_repo(temp_git_repo)

        src_dir = temp_git_repo / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("function test() {}\n")
        (src_dir / "main.js").write_text("function test() {}\n")

        result = grep_repo(repo, "function test", include="*.py")
        assert "src/main.py" in result
        assert "src/main.js" not in result

    def test_grep_repo_max_results(self, temp_git_repo):
        """Test max_results truncation."""
        repo = get_repo(temp_git_repo)

        # Create many files with same pattern
        for i in range(55):
            f = temp_git_repo / f"file_{i}.txt"
            f.write_text("match_this\n")

        result = grep_repo(repo, "match_this", max_results=50)
        lines = result.splitlines()
        # Last line should be truncation note
        assert "[Note:" in lines[-1]
        assert "total matches" in lines[-1]


class TestGetCurrentBranch:
    """Test current branch retrieval."""

    def test_get_current_branch(self, temp_git_repo):
        """Test getting current branch name."""
        repo = get_repo(temp_git_repo)
        branch = get_current_branch(repo)
        
        assert branch == "master"

    def test_get_current_branch_new_repo(self, tmp_path):
        """Test new repo without commits returns None."""
        repo_path = tmp_path / "new-repo"
        repo_path.mkdir()
        repo = Repo.init(repo_path)
        
        branch = get_current_branch(repo)
        
        assert branch == "master"
