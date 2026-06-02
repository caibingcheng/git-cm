"""Git operations for git-cm."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import click
import git
from git import Repo


def is_git_repo(path: Path) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        Repo(path, search_parent_directories=True)
        return True
    except git.InvalidGitRepositoryError:
        return False


def get_repo(path: Path) -> Repo:
    """Get the git repository for the given path."""
    return Repo(path, search_parent_directories=True)


def has_staged_changes(repo: Repo) -> bool:
    """Check if there are staged changes in the repository."""
    try:
        return len(repo.index.diff("HEAD")) > 0 or bool(repo.index.diff(None))
    except Exception:
        # New repository without any commits (no HEAD)
        return len(repo.index.entries) > 0


def get_user_config(repo: Repo) -> dict:
    """Get user name and email from git config."""
    try:
        name = repo.config_reader().get_value("user", "name")
    except Exception:
        name = ""
    
    try:
        email = repo.config_reader().get_value("user", "email")
    except Exception:
        email = ""
    
    return {"name": name, "email": email}


def get_recent_commits(repo: Repo, n: int = 5) -> list:
    """Get the last n commit messages."""
    try:
        commits = list(repo.iter_commits("HEAD", max_count=n))
        return [commit.message.strip() for commit in commits]
    except Exception:
        return []


def check_user_in_history(repo: Repo, name: str, email: str) -> bool:
    """Check if the user's name or email appears in commit history."""
    try:
        for commit in repo.iter_commits("HEAD", max_count=50):
            commit_name = commit.author.name or ""
            commit_email = commit.author.email or ""
            if name == commit_name or email == commit_email:
                return True
        return False
    except Exception:
        return False


def get_staged_diff(repo: Repo) -> str:
    """Get the diff of staged changes."""
    try:
        diff = repo.git.diff("--cached")
        return diff
    except Exception:
        return ""


def commit_changes(repo: Repo, message: str) -> None:
    """Commit staged changes with the given message."""
    try:
        repo.index.commit(message)
        click.echo(f"Committed: {message}")
    except Exception as e:
        click.echo(f"Error committing changes: {e}", err=True)
        raise click.ClickException(str(e))


def read_file(repo: Repo, relative_path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Read file content from the repository working tree.

    Args:
        repo: GitPython Repo object
        relative_path: Path relative to repo root
        start_line: 1-based starting line number
        end_line: 1-based ending line number (None = end of file)

    Returns:
        File content or error message
    """
    try:
        file_path = Path(repo.working_tree_dir) / relative_path

        # Security check: path must be within repo
        try:
            file_path.resolve().relative_to(Path(repo.working_tree_dir).resolve())
        except ValueError:
            return f"[Security error: path '{relative_path}' is outside the repository]"

        if not file_path.exists():
            return f"[Error: File '{relative_path}' not found]"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Adjust line numbers
        if start_line < 1:
            start_line = 1
        if end_line is None or end_line > total_lines:
            end_line = total_lines
        if start_line > total_lines:
            return f"[Error: start_line {start_line} exceeds file length {total_lines}]"

        # Extract lines
        selected_lines = lines[start_line - 1:end_line]
        content = "".join(selected_lines)

        # Build result with notes
        result_parts = [content]
        notes = []
        if start_line > 1 or end_line < total_lines:
            notes.append(f"Lines {start_line}-{end_line} of {total_lines}")
        if notes:
            result_parts.append(f"[Note: {'; '.join(notes)}]")

        return "\n".join(result_parts)

    except Exception as e:
        return f"[Error reading '{relative_path}': {e}]"


def read_files_batch(repo: Repo, requests: List[Dict], max_total_chars: int = 4000) -> Dict[str, Dict[str, str]]:
    """Read multiple files with cumulative budget.

    Args:
        repo: GitPython Repo object
        requests: List of dicts with keys: path, start_line, end_line
        max_total_chars: Maximum total characters to read

    Returns:
        Dict mapping path to {"content": str, "status": str}
        Status: "ok", "truncated", "skipped", "error"
    """
    results: Dict[str, Dict[str, str]] = {}
    cumulative = 0
    budget_exhausted = False

    for req in requests:
        path = req["path"]
        if budget_exhausted:
            results[path] = {
                "content": "[Skipped: read budget exhausted]",
                "status": "skipped",
            }
            continue

        content = read_file(
            repo,
            path,
            req.get("start_line", 1),
            req.get("end_line"),
        )

        # Check if it's an error response
        if content.startswith("[Error:") or content.startswith("[Security error:"):
            results[path] = {"content": content, "status": "error"}
            continue

        # Check cumulative budget
        if cumulative + len(content) > max_total_chars:
            remaining = max_total_chars - cumulative
            if remaining > 100:
                truncated = content[:remaining]
                results[path] = {
                    "content": truncated + f"\n[Note: truncated from {len(content)} to {remaining} chars (budget limit)]",
                    "status": "truncated",
                }
                cumulative = max_total_chars
            else:
                results[path] = {
                    "content": "[Skipped: read budget exhausted]",
                    "status": "skipped",
                }
            budget_exhausted = True
        else:
            results[path] = {"content": content, "status": "ok"}
            cumulative += len(content)

    return results


def find_agents_md(repo: Repo) -> str:
    """Find and read AGENTS.md from repo root. Returns empty string if not found."""
    try:
        agents_path = Path(repo.working_tree_dir) / "AGENTS.md"
        if agents_path.exists():
            with open(agents_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Limit length
            max_len = 3000
            if len(content) > max_len:
                content = content[:max_len] + "\n[... truncated]"
            return content
    except Exception:
        pass
    return ""


def grep_repo(repo: Repo, pattern: str, include: Optional[str] = None, max_results: int = 50) -> str:
    """Search for pattern in repository files using grep.

    Args:
        repo: GitPython Repo object
        pattern: grep search pattern
        include: Optional file pattern (e.g., "*.py" or "*.{py,js}")
        max_results: Maximum number of file paths to return

    Returns:
        Newline-separated file paths (relative to repo root), or error message.
        Files outside the repository are filtered out with a note.
    """
    try:
        repo_path = Path(repo.working_tree_dir)

        cmd = ["grep", "-rl", "-I", "-E", pattern]
        if include:
            cmd.extend(["--include", include])
        cmd.append(".")

        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 1:
            # No matches
            return ""
        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"[Error: grep failed: {stderr}]"

        lines = result.stdout.strip().splitlines()
        repo_root = repo_path.resolve()
        valid_paths = []
        skipped_count = 0

        for line in lines:
            abs_path = (repo_root / line).resolve()
            try:
                rel_path = abs_path.relative_to(repo_root)
                valid_paths.append(str(rel_path))
            except ValueError:
                skipped_count += 1

        if skipped_count > 0:
            click.echo(
                f"Warning: {skipped_count} file(s) skipped (outside repository)",
                err=True,
            )

        if len(valid_paths) > max_results:
            truncated = valid_paths[:max_results]
            truncated.append(f"[Note: {len(valid_paths)} total matches, showing first {max_results}]")
            return "\n".join(truncated)

        return "\n".join(valid_paths)

    except subprocess.TimeoutExpired:
        return "[Error: grep timed out after 30s]"
    except FileNotFoundError:
        return "[Error: grep command not found]"
    except Exception as e:
        return f"[Error: {e}]"
