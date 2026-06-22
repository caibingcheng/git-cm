"""Git operations for git-cm."""

import configparser
import os
import subprocess
import tempfile
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
        return len(repo.index.diff("HEAD")) > 0
    except (git.BadName, git.BadObject):
        # New repository without any commits (no HEAD)
        return len(repo.index.entries) > 0


def get_user_config(repo: Repo) -> dict:
    """Get user name and email from git config."""
    try:
        name = repo.config_reader().get_value("user", "name")
    except (configparser.NoSectionError, configparser.NoOptionError):
        name = ""
    
    try:
        email = repo.config_reader().get_value("user", "email")
    except (configparser.NoSectionError, configparser.NoOptionError):
        email = ""
    
    return {"name": name, "email": email}


def get_recent_commits(repo: Repo, n: int = 5) -> list:
    """Get the last n commit messages."""
    try:
        commits = list(repo.iter_commits("HEAD", max_count=n))
        return [commit.message.strip() for commit in commits]
    except (git.BadName, git.BadObject):
        return []


def get_current_branch(repo: Repo) -> Optional[str]:
    """Get the current branch name.
    
    Returns None if no branch exists (e.g., new repo without commits).
    """
    try:
        return repo.active_branch.name
    except (TypeError, ValueError, git.InvalidGitRepositoryError):
        return None


def check_user_in_history(repo: Repo, name: str, email: str) -> bool:
    """Check if the user's name or email appears in commit history."""
    try:
        for commit in repo.iter_commits("HEAD", max_count=50):
            commit_name = commit.author.name or ""
            commit_email = commit.author.email or ""
            if name == commit_name or email == commit_email:
                return True
        return False
    except (git.BadName, git.BadObject):
        return False

def get_staged_files(repo: Repo) -> List[Dict[str, str]]:
    """Get list of staged files with their status and binary detection.
    
    Returns:
        List of dicts with keys: path, status, is_binary
        status: 'added', 'modified', 'deleted', 'renamed'
        is_binary: 'true' or 'false'
    """
    files = []
    
    try:
        # Get file statuses
        status_output = repo.git.diff("--cached", "--name-status")
        
        # Get binary detection (binary files show as "- -" in numstat)
        numstat_output = repo.git.diff("--cached", "--numstat")
        
        # Parse binary files
        binary_files = set()
        for line in numstat_output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "-" and parts[1] == "-":
                binary_files.add(parts[2])
        
        # Parse status
        for line in status_output.strip().splitlines():
            if not line.strip():
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            status_code = parts[0]
            path = parts[1]
            
            # Map status code
            if status_code == "A":
                status = "added"
            elif status_code == "M":
                status = "modified"
            elif status_code == "D":
                status = "deleted"
            elif status_code.startswith("R"):
                status = "renamed"
                # For renamed, path is the destination
                if len(parts) >= 3:
                    path = parts[2]
            else:
                status = "modified"
            
            is_binary = "true" if path in binary_files else "false"
            
            files.append({
                "path": path,
                "status": status,
                "is_binary": is_binary,
            })
    except Exception as e:
        click.echo(f"Warning: Failed to get staged files: {e}", err=True)
    
    return files

def get_staged_diff(repo: Repo) -> str:
    """Get the diff of staged changes with improved binary file handling."""
    try:
        diff = repo.git.diff("--cached")
        
        # Enhance binary file descriptions
        # Replace generic "Binary files differ" with explicit file paths
        lines = diff.split("\n")
        result_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if "Binary files differ" in line:
                # Try to extract file path from previous diff header line
                file_path = None
                for j in range(len(result_lines) - 1, -1, -1):
                    if result_lines[j].startswith("diff --git "):
                        parts = result_lines[j].split(" ")
                        if len(parts) >= 4:
                            # b/path/to/file
                            file_path = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                        break
                if file_path:
                    result_lines.append(f"[Binary file: {file_path}]")
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
            i += 1
        
        return "\n".join(result_lines)
    except Exception as e:
        click.echo(f"Warning: Failed to get staged diff: {e}", err=True)
        return ""


def get_unmerged_files(repo: Repo) -> List[str]:
    """Return paths of files that have unmerged index entries."""
    try:
        return sorted(
            {
                blob.path
                for entries in repo.index.unmerged_blobs().values()
                for _stage, blob in entries
            }
        )
    except Exception:
        return []


def commit_changes(repo: Repo, message: str) -> None:
    """Commit staged changes with the given message."""
    try:
        repo.index.commit(message)
        click.echo(f"Committed: {message}")
    except git.exc.UnmergedEntriesError as e:
        unmerged = get_unmerged_files(repo)
        files_list = "\n  ".join(unmerged) if unmerged else str(e)
        click.echo(
            "Error: Cannot commit because there are unmerged entries in the index.\n"
            "This usually happens during an unfinished merge or rebase.\n"
            "Unmerged files:\n  " + files_list + "\n"
            "Resolve the conflicts, run 'git add <file>', then commit.",
            err=True,
        )
        raise click.ClickException("Unmerged entries prevent commit")
    except (git.GitCommandError, OSError) as e:
        click.echo(f"Error committing changes: {e}", err=True)
        raise click.ClickException(str(e))


def read_file(
    repo: Repo,
    relative_path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> str:
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


def read_files_batch(
    repo: Repo,
    requests: List[Dict],
    max_total_chars: int = 4000,
) -> Dict[str, Dict[str, str]]:
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
                note = (
                    f"\n[Note: truncated from {len(content)} to "
                    f"{remaining} chars (budget limit)]"
                )
                results[path] = {
                    "content": truncated + note,
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


def grep_repo(
    repo: Repo,
    pattern: str,
    include: Optional[str] = None,
    max_results: int = 50,
) -> str:
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
            note = f"[Note: {len(valid_paths)} total matches, showing first {max_results}]"
            truncated.append(note)
            return "\n".join(truncated)

        return "\n".join(valid_paths)

    except subprocess.TimeoutExpired:
        return "[Error: grep timed out after 30s]"
    except FileNotFoundError:
        return "[Error: grep command not found]"
    except Exception as e:
        return f"[Error: {e}]"


def has_uncommitted_changes(repo: Repo) -> bool:
    """Check if there are staged, unstaged, or untracked changes."""
    try:
        return repo.is_dirty(index=True, working_tree=True, untracked_files=True)
    except git.GitCommandError:
        return False


def stash_changes(repo: Repo, message: str = "git-cm rewrite backup") -> Optional[str]:
    """Stash current changes (including untracked) and return the stash ref.

    Returns:
        Stash commit sha if a stash was created, None if there was nothing to stash.
    """
    if not has_uncommitted_changes(repo):
        return None

    try:
        output = repo.git.stash("push", "-u", "-m", message)
        if "No local changes to save" in output:
            return None
        return repo.git.rev_parse("refs/stash")
    except git.GitCommandError as e:
        click.echo(f"Error stashing changes: {e}", err=True)
        raise click.ClickException(str(e))


def pop_stash(repo: Repo, expected_ref: str) -> None:
    """Pop the stash, verifying it is still the expected ref on top."""
    try:
        current_top = repo.git.rev_parse("refs/stash")
    except git.GitCommandError as e:
        raise RuntimeError(f"Cannot locate stash to restore: {e}")

    if current_top != expected_ref:
        raise RuntimeError(
            f"Stash stack changed unexpectedly. "
            f"Expected {expected_ref}, found {current_top}. "
            f"Please run 'git stash list' and restore manually."
        )

    try:
        repo.git.stash("pop")
    except git.GitCommandError as e:
        raise RuntimeError(f"Failed to pop stash: {e}")


def is_rebasing(repo: Repo) -> bool:
    """Check whether a rebase is currently in progress."""
    git_dir = Path(repo.git_dir)
    return (git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists()


def abort_rebase(repo: Repo) -> None:
    """Abort an in-progress rebase, ignoring errors."""
    try:
        repo.git.rebase("--abort")
    except git.GitCommandError:
        pass


def get_commit_diff(repo: Repo, commit_sha: str) -> str:
    """Get the diff of a specific commit relative to its first parent."""
    try:
        diff = repo.git.show(
            commit_sha,
            "-p",
            "--format=",
            "--binary",
            "--find-renames",
        )
        return diff
    except git.GitCommandError as e:
        click.echo(f"Warning: Failed to get commit diff: {e}", err=True)
        return ""


def is_commit_pushed(repo: Repo, commit_sha: str) -> bool:
    """Check if a commit exists on any remote tracking branch."""
    try:
        output = repo.git.branch("-r", "--contains", commit_sha)
        return bool(output.strip())
    except git.GitCommandError:
        return False


def _unlink_safely(path: Optional[str]) -> None:
    """Remove a file, ignoring errors if it doesn't exist or is already removed."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _build_editor_script(content_template: str) -> str:
    """Create a temporary executable script and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        with f:
            f.write(content_template)
            path = f.name
        os.chmod(path, 0o755)
        return path
    except OSError:
        _unlink_safely(f.name)
        raise


def rewrite_commit_message(repo: Repo, commit_sha: str, new_message: str) -> None:
    """Rewrite the message of an arbitrary commit.

    Supports HEAD amend, intermediate commits via rebase, and root commits.
    """
    try:
        _do_rewrite_commit_message(repo, commit_sha, new_message)
    except Exception:
        if is_rebasing(repo):
            abort_rebase(repo)
        raise


def _do_rewrite_commit_message(repo: Repo, commit_sha: str, new_message: str) -> None:
    """Internal implementation of rewrite_commit_message."""
    commit = repo.commit(commit_sha)
    head_commit = repo.head.commit

    # HEAD: simple amend
    if commit == head_commit:
        try:
            repo.git.commit("--amend", "-m", new_message)
        except git.GitCommandError as e:
            click.echo(f"Error amending commit: {e}", err=True)
            raise click.ClickException(str(e))
        return

    parents = commit.parents

    # Root commit
    if not parents:
        try:
            new_sha = repo.git.commit_tree(commit.tree.hexsha, "-m", new_message)
            if head_commit == commit:
                repo.git.update_ref("HEAD", new_sha)
            else:
                repo.git.rebase("--onto", new_sha, "--root")
        except git.GitCommandError as e:
            click.echo(f"Error rewriting root commit: {e}", err=True)
            raise click.ClickException(str(e))
        return

    # Intermediate commit: interactive rebase with editor scripts
    parent_sha = parents[0].hexsha
    target_prefix = commit_sha[:7]
    seq_editor_script = None
    msg_editor_script = None

    try:
        seq_editor_script = _build_editor_script(
            f"""#!/usr/bin/env python3
import sys
path = sys.argv[1]
with open(path, "r") as f:
    lines = f.readlines()
with open(path, "w") as f:
    for line in lines:
        parts = line.split()
        if (
            len(parts) >= 2
            and parts[0] == "pick"
            and parts[1].startswith("{target_prefix}")
        ):
            f.write(line.replace("pick ", "reword ", 1))
        else:
            f.write(line)
"""
        )

        msg_editor_script = _build_editor_script(
            f"""#!/usr/bin/env python3
import sys
with open(sys.argv[1], "w") as f:
    f.write({repr(new_message)})
"""
        )

        env = os.environ.copy()
        env["GIT_SEQUENCE_EDITOR"] = seq_editor_script
        env["GIT_EDITOR"] = msg_editor_script

        subprocess.run(
            ["git", "rebase", "-i", parent_sha],
            cwd=str(repo.working_tree_dir),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        if is_rebasing(repo):
            abort_rebase(repo)
        stderr = e.stderr.strip() if e.stderr else str(e)
        click.echo(f"Error rewriting commit via rebase: {stderr}", err=True)
        raise click.ClickException(f"Failed to rewrite commit: {stderr}")
    except OSError as e:
        if is_rebasing(repo):
            abort_rebase(repo)
        click.echo(f"Error rewriting commit: {e}", err=True)
        raise click.ClickException(str(e))
    finally:
        _unlink_safely(seq_editor_script)
        _unlink_safely(msg_editor_script)
