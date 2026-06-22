"""Tests for git operations."""

import os
from pathlib import Path

import click
import git
import pytest
from git import Repo

from git_cm.git_utils import (
    abort_rebase,
    _build_editor_script,
    _unlink_safely,
    commit_changes,
    find_agents_md,
    get_commit_diff,
    get_current_branch,
    check_user_in_history,
    get_recent_commits,
    get_repo,
    get_staged_diff,
    get_unmerged_files,
    get_user_config,
    grep_repo,
    has_staged_changes,
    has_uncommitted_changes,
    is_commit_pushed,
    is_git_repo,
    is_rebasing,
    pop_stash,
    read_file,
    read_files_batch,
    rewrite_commit_message,
    stash_changes,
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


@pytest.fixture
def repo_with_merge_conflict(temp_git_repo):
    """Create a repo with an unresolved merge conflict."""
    repo = get_repo(temp_git_repo)
    base_file = temp_git_repo / "conflict.txt"
    base_file.write_text("base content")
    repo.index.add([str(base_file)])
    repo.index.commit("base")

    repo.create_head("feature")
    repo.heads.feature.checkout()
    base_file.write_text("feature content")
    repo.index.add([str(base_file)])
    repo.index.commit("feature change")

    repo.heads.master.checkout()
    base_file.write_text("master content")
    repo.index.add([str(base_file)])
    repo.index.commit("master change")

    try:
        repo.git.merge("feature")
    except Exception:
        pass

    return temp_git_repo


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


class TestBuildEditorScript:
    """Test _build_editor_script helper."""

    def test_build_editor_script_creates_executable_file(self):
        """Verify the returned path exists, is executable, and contains the template content."""
        template = "#!/usr/bin/env python3\nprint('hello')\n"
        path = _build_editor_script(template)

        assert os.path.exists(path)
        assert os.access(path, os.X_OK)
        assert Path(path).read_text() == template

        _unlink_safely(path)

    def test_build_editor_script_cleans_up_on_write_failure(self, monkeypatch):
        """Verify no temp file is left behind when chmod fails."""
        recorded_paths = []

        def failing_chmod(path, mode):
            recorded_paths.append(path)
            raise OSError("chmod failed")

        monkeypatch.setattr("git_cm.git_utils.os.chmod", failing_chmod)

        with pytest.raises(OSError, match="chmod failed"):
            _build_editor_script("content")

        assert len(recorded_paths) == 1
        assert not os.path.exists(recorded_paths[0])


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


class TestUnmergedEntries:
    """Test detection and handling of unmerged index entries."""

    def test_get_unmerged_files_clean(self, temp_git_repo):
        """Test clean repo returns empty list."""
        repo = get_repo(temp_git_repo)
        assert get_unmerged_files(repo) == []

    def test_get_unmerged_files_with_conflict(self, repo_with_merge_conflict):
        """Test unresolved merge conflict returns conflict file."""
        repo = get_repo(repo_with_merge_conflict)
        assert get_unmerged_files(repo) == ["conflict.txt"]

    def test_commit_changes_unmerged_entries(self, repo_with_merge_conflict, capsys):
        """Test commit raises ClickException with friendly message."""
        repo = get_repo(repo_with_merge_conflict)

        with pytest.raises(click.ClickException, match="Unmerged entries prevent commit"):
            commit_changes(repo, "feat: should fail")

        captured = capsys.readouterr()
        assert "Cannot commit because there are unmerged entries" in captured.err
        assert "conflict.txt" in captured.err


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


class TestHasUncommittedChanges:
    """Test detection of uncommitted changes."""

    def test_clean_repo(self, temp_git_repo):
        """Test clean repository returns False."""
        repo = get_repo(temp_git_repo)
        assert has_uncommitted_changes(repo) is False

    def test_staged_changes(self, temp_git_repo):
        """Test staged changes are detected."""
        repo = get_repo(temp_git_repo)
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified content")
        repo.index.add([str(test_file)])
        assert has_uncommitted_changes(repo) is True

    def test_untracked_files(self, temp_git_repo):
        """Test untracked files are detected."""
        repo = get_repo(temp_git_repo)
        (temp_git_repo / "untracked.txt").write_text("untracked")
        assert has_uncommitted_changes(repo) is True


class TestStashChanges:
    """Test stash and restore operations."""

    def test_stash_and_pop(self, temp_git_repo):
        """Test stashing and popping restores changes."""
        repo = get_repo(temp_git_repo)
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("stashed content")
        repo.index.add([str(test_file)])

        ref = stash_changes(repo, "test stash")
        assert ref is not None
        assert has_uncommitted_changes(repo) is False

        pop_stash(repo, ref)
        assert "stashed content" in test_file.read_text()

    def test_stash_nothing_returns_none(self, temp_git_repo):
        """Test stashing clean repo returns None."""
        repo = get_repo(temp_git_repo)
        ref = stash_changes(repo, "empty stash")
        assert ref is None

    def test_pop_stash_ref_changed(self, temp_git_repo):
        """Test pop detects unexpected stash stack changes."""
        repo = get_repo(temp_git_repo)
        f1 = temp_git_repo / "a.txt"
        f1.write_text("a")
        repo.index.add([str(f1)])
        ref1 = stash_changes(repo, "first")

        f2 = temp_git_repo / "b.txt"
        f2.write_text("b")
        repo.index.add([str(f2)])
        stash_changes(repo, "second")

        with pytest.raises(RuntimeError, match="Stash stack changed unexpectedly"):
            pop_stash(repo, ref1)

    def test_pop_stash_failure_message(self, temp_git_repo):
        """Test pop_stash includes recovery guidance when stash top changed."""
        repo = get_repo(temp_git_repo)
        f1 = temp_git_repo / "a.txt"
        f1.write_text("a")
        repo.index.add([str(f1)])
        ref1 = stash_changes(repo, "first")

        f2 = temp_git_repo / "b.txt"
        f2.write_text("b")
        repo.index.add([str(f2)])
        stash_changes(repo, "second")

        with pytest.raises(RuntimeError) as exc_info:
            pop_stash(repo, ref1)

        message = str(exc_info.value)
        assert "Expected" in message
        assert "found" in message
        assert "git stash list" in message
        assert "restore manually" in message


class TestCommitDiff:
    """Test retrieving a specific commit diff."""

    def test_get_commit_diff(self, temp_git_repo):
        """Test diff of a commit includes changes."""
        repo = get_repo(temp_git_repo)
        test_file = temp_git_repo / "test.txt"
        test_file.write_text("modified content")
        repo.index.add([str(test_file)])
        commit = repo.index.commit("modify file")

        diff = get_commit_diff(repo, commit.hexsha)

        assert "modified content" in diff
        assert "initial content" in diff

    def test_get_commit_diff_invalid_commit(self, temp_git_repo):
        """Test passing a non-existent commit sha returns empty string."""
        repo = get_repo(temp_git_repo)
        diff = get_commit_diff(repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        assert diff == ""


class TestIsCommitPushed:
    """Test detection of pushed commits."""

    def test_commit_pushed_true(self, temp_git_repo):
        """Test commit pushed to remote is detected."""
        repo = get_repo(temp_git_repo)
        remote_path = temp_git_repo.parent / "remote.git"
        remote_path.mkdir()
        Repo.init(remote_path, bare=True)
        repo.create_remote("origin", str(remote_path))
        repo.git.push("-u", "origin", "master")

        head_sha = repo.head.commit.hexsha
        assert is_commit_pushed(repo, head_sha) is True

    def test_commit_pushed_false(self, temp_git_repo):
        """Test local-only commit is not detected as pushed."""
        repo = get_repo(temp_git_repo)
        local_file = temp_git_repo / "local.txt"
        local_file.write_text("local")
        repo.index.add([str(local_file)])
        commit = repo.index.commit("local commit")

        assert is_commit_pushed(repo, commit.hexsha) is False

    def test_is_commit_pushed_invalid_commit(self, temp_git_repo):
        """Test passing a non-existent commit sha returns False."""
        repo = get_repo(temp_git_repo)
        result = is_commit_pushed(repo, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
        assert result is False


class TestRewriteCommitMessage:
    """Test rewriting commit messages."""

    def test_rewrite_head(self, temp_git_repo):
        """Test rewriting HEAD with amend."""
        repo = get_repo(temp_git_repo)
        new_file = temp_git_repo / "new.txt"
        new_file.write_text("content")
        repo.index.add([str(new_file)])
        repo.index.commit("old head message")

        rewrite_commit_message(repo, repo.head.commit.hexsha, "new head message")

        assert repo.head.commit.message.strip() == "new head message"

    def test_rewrite_intermediate_commit(self, temp_git_repo):
        """Test rewriting an intermediate commit."""
        repo = get_repo(temp_git_repo)
        for i in range(2):
            f = temp_git_repo / f"file{i}.txt"
            f.write_text(f"content {i}")
            repo.index.add([str(f)])
            repo.index.commit(f"commit {i}")

        commits = list(repo.iter_commits("HEAD", max_count=10))
        target = commits[1]
        old_head = commits[0]

        rewrite_commit_message(repo, target.hexsha, "rewritten middle")

        new_commits = list(repo.iter_commits("HEAD", max_count=10))
        messages = [c.message.strip() for c in new_commits]
        assert "rewritten middle" in messages
        assert new_commits[0].hexsha != old_head.hexsha

    def test_rewrite_root_commit(self, temp_git_repo):
        """Test rewriting the root commit."""
        repo = get_repo(temp_git_repo)
        root = list(repo.iter_commits("HEAD", max_count=10))[-1]

        rewrite_commit_message(repo, root.hexsha, "new root message")

        new_root = list(repo.iter_commits("HEAD", max_count=10))[-1]
        assert new_root.message.strip() == "new root message"
        assert new_root.hexsha != root.hexsha

    def test_rewrite_commit_message_aborts_rebase_on_conflict(self, temp_git_repo):
        """Test that rewrite_commit_message aborts an in-progress rebase on failure."""
        repo = get_repo(temp_git_repo)
        base_file = temp_git_repo / "file.txt"
        base_file.write_text("base")
        repo.index.add([str(base_file)])
        repo.index.commit("base")

        # Create branch with conflicting change
        repo.create_head("feature")
        repo.heads.feature.checkout()
        base_file.write_text("feature")
        repo.index.add([str(base_file)])
        repo.index.commit("feature change")

        # Return to master and make conflicting change
        repo.heads.master.checkout()
        base_file.write_text("master")
        repo.index.add([str(base_file)])
        master_commit = repo.index.commit("master change")

        # Start rebase that will conflict
        try:
            repo.git.rebase("master", "feature")
        except Exception:
            pass

        assert is_rebasing(repo) is True

        with pytest.raises(click.ClickException):
            rewrite_commit_message(repo, master_commit.hexsha, "rewritten master")

        assert is_rebasing(repo) is False


class TestRebaseState:
    """Test rebase state detection and abort."""

    def test_is_rebasing_and_abort(self, temp_git_repo):
        """Test detecting and aborting a rebase."""
        repo = get_repo(temp_git_repo)
        base_file = temp_git_repo / "file.txt"
        base_file.write_text("base")
        repo.index.add([str(base_file)])
        base_commit = repo.index.commit("base")

        # Create branch with conflicting change
        repo.create_head("feature")
        repo.heads.feature.checkout()
        base_file.write_text("feature")
        repo.index.add([str(base_file)])
        repo.index.commit("feature change")

        # Return to master and make conflicting change
        repo.heads.master.checkout()
        base_file.write_text("master")
        repo.index.add([str(base_file)])
        repo.index.commit("master change")

        # Start rebase that will conflict
        try:
            repo.git.rebase("master", "feature")
        except Exception:
            pass

        assert is_rebasing(repo) is True
        abort_rebase(repo)
        assert is_rebasing(repo) is False
